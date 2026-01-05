#include <cutlass/fast_math.h>
#include <torch/extension.h>

#include <cute/layout.hpp>
#include <cute/tensor.hpp>

#include "../utils.h"

using namespace cute;

// Ref: https://github.com/xlite-dev/LeetCUDA
// Ref: https://github.com/izmttk/flash_attention_cute

// 为什么 一个 mma 的一个 row 是 t0-t4 四个线程，一个大的 MMA 在 N
// 方向有多个，但还是 t0-t4 个线程。 所以规约只需要规约四个线程就行了。
// ┌────────────────────────────────────────────────────────────────────┐
// │           沿 N 方向重复：64×8 的大 Tile (4 个 MMA 横向排列)            │
// ├────────────────────────────────────────────────────────────────────┤
// │                                                                    │
// │      MMA Tile 0       MMA Tile 1       MMA Tile 2       MMA Tile 3 │
// │    (col 0-7)        (col 8-15)       (col 16-23)      (col 24-31)  │
// │   ┌─────────────┬─────────────┬─────────────┬─────────────┐        │
// │   │T0 T1 T2 T3  │T0 T1 T2 T3  │T0 T1 T2 T3  │T0 T1 T2 T3  │ row 0  │
// │   │T0 T1 T2 T3  │T0 T1 T2 T3  │T0 T1 T2 T3  │T0 T1 T2 T3  │ row 1  │
// │   ├─────────────┼─────────────┼─────────────┼─────────────┤        │
// │   │T4 T5 T6 T7  │T4 T5 T6 T7  │T4 T5 T6 T7  │T4 T5 T6 T7  │ row 2  │
// │   │T4 T5 T6 T7  │T4 T5 T6 T7  │T4 T5 T6 T7  │T4 T5 T6 T7  │ row 3  │
// │   │    ...      │    ...      │    ...      │    ...      │  ...   │
// │   └─────────────┴─────────────┴─────────────┴─────────────┘        │
// │                                                                    │
// │   ▲▲▲ 关键观察 ▲▲▲                                                  │
// │   Row 0 的所有 32 个元素 (4 个 MMA × 8 列) 仍然由 T0,T1,T2,T3 持有！   │
// │   每个线程现在持有更多元素，但行归属不变！                               │
// │                                                                    │

template <typename T_, int BlockM_, int BlockN_, int HeadDim_, int NWarpsPerSM_>
struct FlashAttnConfig_ {
  using T = T_;

  // TODO: 定义 stride
  int stride_qb;
  int stride_kb;
  int stride_vb;
  int stride_ob;

  int stride_qh;
  int stride_kh;
  int stride_vh;
  int stride_oh;

  int stride_qm;
  int stride_km;
  int stride_vm;
  int stride_om;

  int stride_qk;
  int stride_kk;
  int stride_vk;
  int stride_ok;

  static constexpr int NWarpsPerSM = NWarpsPerSM_;
  static constexpr int NumThreads = NWarpsPerSM * 32;

  static constexpr int Block_M = BlockM_;
  static constexpr int Block_N = BlockN_;
  static constexpr int HeadDim = HeadDim_;

  // 每个线程一次读取 128bit 大小的数据，根据元素的类型，计算一次能多少个元素。
  static constexpr int GmemValsPerLoad = sizeof(uint128_t) / sizeof(T);
  // 一行需要多少个线程。
  static constexpr int GmemThreadsPerRow = HeadDim / GmemValsPerLoad;

  // 全局显存拷贝到共享显存中。
  using GmemCopyAtom =
      Copy_Atom<AutoVectorizingCopyWithAssumedAlignment<sizeof(uint128_t) * 8>,
                T>;

  // TODO: 使用 async
  // 第一个参数： 一个指令，一次拷贝多少数据
  // 第二个参数：线程的排列，每一个线程执行一次
  // 第三个参数：每个线程执行多少次指令
  using TiledCopyQKVO = decltype(make_tiled_copy(
      GmemCopyAtom{},
      make_layout(
          Shape<Int<NumThreads / GmemThreadsPerRow>, Int<GmemThreadsPerRow>>{},
          GenRowMajor{}),  // thr_layout
      make_layout(Shape<_1, Int<GmemValsPerLoad>>{}, GenRowMajor{})));

  static_assert(Int<NumThreads / GmemThreadsPerRow>::value <= Block_M,
                "NumThreads must be less than or equal to BlockQO");

  // LDSM will fit in the MMA_Atom.
  // TODO: 为什么要用 LDSM？
  using SmemCopyAtom = Copy_Atom<SM75_U32x4_LDSM_N, T>;

  // TODO: 为什么要用转置？
  using SmemCopyAtomTransposed = Copy_Atom<SM75_U16x8_LDSM_T, T>;

  // 把计算好的结果拷贝到全局显存。
  // TODO: 使用 tiled 是不是更快？
  using SmemCopyAtomO =
      Copy_Atom<AutoVectorizingCopyWithAssumedAlignment<sizeof(uint128_t) * 8>,
                T>;

  static_assert(std::is_same_v<T, half_t> || std::is_same_v<T, bfloat16_t>);

  // TODO: 为什么不用  SM80_16x8x16?
  // For simplicity, mnk == (16, 8, 8) is used: two MMAs will have the same
  // layout so that we don't need to adjust tSrS to fit in tOrS
  using MMA_Atom = std::conditional_t<std::is_same_v<T, half_t>,
                                      MMA_Atom<SM80_16x8x8_F32F16F16F32_TN>,
                                      MMA_Atom<SM80_16x8x8_F32BF16BF16F32_TN>>;

  using TiledMMA = decltype(make_tiled_mma(
      // thr_layout
      MMA_Atom{}, make_layout(Shape<Int<NWarpsPerSM>, _1, _1>{}, GenRowMajor{}),
      Tile<Int<16 * NWarpsPerSM>, _16, _16>{}));

  static_assert(
      16 * NWarpsPerSM <= Block_M && 16 <= Block_N && 16 <= HeadDim,
      "BlockQO, BlockKV, and HeadDim must be greater than or equal to "
      "16 * NWarpsPerSM, 16, and 16 respectively");

  // sanity checks
  static_assert(size(TiledMMA{}) == NumThreads &&
                size(TiledMMA{}) == size(TiledCopyQKVO{}));
};

template <typename FlashAttnConfig_>
__global__ void flash_attn_v2_kernel(typename FlashAttnConfig_::T* Q_ptr,
                                     typename FlashAttnConfig_::T* K_ptr,
                                     typename FlashAttnConfig_::T* V_ptr,
                                     typename FlashAttnConfig_::T* O_ptr,  //
                                     int B, int H, int N_QO_CTX, int N_KV_CTX,
                                     int D, float scaler) {
  using namespace cute;

  // TODO: static check

  using T = typename FlashAttnConfig_::T;
  // QO 的 block 长度是一样的， M 方向的
  // the length of kv is same.
  constexpr int Block_M = FlashAttnConfig_::Block_M;
  constexpr int Block_N = FlashAttnConfig_::Block_N;
  constexpr int HeadDim = FlashAttnConfig_::HeadDim;

  // the tiledCopy to copy global memory to shared memory
  // TODO: use async copy
  using TiledCopy = typename FlashAttnConfig_::TiledCopyQKVO;

  // copy shared memory to register
  using SmemCopyAtom = typename FlashAttnConfig_::SmemCopyAtom;
  // TODO: why transposed
  using SmemCopyAtom_T = typename FlashAttnConfig_::SmemCopyAtomTransposed;

  // first copy O to shared memory, it will be faster
  using SmemCopyAtomO = typename FlashAttnConfig_::SmemCopyAtomO;

  using TiledMMA = typename FlashAttnConfig_::TiledMMA;

  // x for B, y for H, z for N
  const int bb = blockIdx.x;
  const int bh = blockIdx.y;
  const int bn = blockIdx.z;

  // tx for block
  const int tx = threadIdx.x;

  // =================== 1 ===================
  // define the tensor of Q, K, V, O from the ptr
  // TODO: self-define shape for q, k, v, o
  auto Q = make_tensor(
      make_gmem_ptr(Q_ptr),
      make_layout(make_shape(B, H, N_QO_CTX, HeadDim), GenRowMajor{}));

  auto O = make_tensor(
      make_gmem_ptr(O_ptr),
      make_layout(make_shape(B, H, N_QO_CTX, HeadDim), GenRowMajor{}));

  auto K = make_tensor(
      make_gmem_ptr(K_ptr),
      make_layout(make_shape(B, H, N_KV_CTX, HeadDim), GenRowMajor{}));

  auto V = make_tensor(
      make_gmem_ptr(V_ptr),
      make_layout(make_shape(B, H, N_KV_CTX, HeadDim), GenRowMajor{}));

  // “从大矩阵 Q 中，锁定第 bx 个 Batch、第 by 个 Head，
  // 并沿着序列长度方向切出第 bz 块（大小为 BlockQO），最终剥离掉多余的维度，
  // 给我一个 [BlockQO, HeadDim] 的二维矩阵视图，命名为 gQ。”
  // shape of gQ and gO is (Block_M, HeadDim)
  // shape of gK and gV is (Block_N, HeadDim, Num_Blocks)
  auto gQ =
      local_tile(Q, make_shape(_1{}, _1{}, Int<Block_M>{}, Int<HeadDim>{}),
                 make_coord(bb, bh, bn, 0))(0, 0, _, _);
  auto gO =
      local_tile(Q, make_shape(_1{}, _1{}, Int<Block_M>{}, Int<HeadDim>{}),
                 make_coord(bb, bh, bn, 0))(0, 0, _, _);

  // 由于我们在 SeqLen 维度使用了 _，local_tile 返回的 Tensor
  // 维度会增加一维（或者说保留了一维的分块索引）。 返回的 Tensor
  // 逻辑形状结构如下:
  // [Batch(1), Head(1), Block_N, HeadDim, Num_Blocks]
  auto gK =
      local_tile(K, make_shape(_1{}, _1{}, Int<Block_N>{}, Int<HeadDim>{}),
                 make_coord(bb, bh, _, 0))(0, 0, _, _, _);
  auto gV =
      local_tile(V, make_shape(_1{}, _1{}, Int<Block_N>{}, Int<HeadDim>{}),
                 make_coord(bb, bh, _, 0))(0, 0, _, _, _);

  // =================== 2 ===================
  // define shared memory layout
  // TODO: use blow define type
  //   extern __shared__ unsigned char alignas(T) smem[];
  //   T* q_smem = reinterpret_cast<T*>(smem);
  //   T* k_smem = q_smem + cosize(SmemLayoutQ{});
  //   T* v_smem = k_smem + cosize(SmemLayoutK{});
  __shared__ T sQ_ptr[Block_M * HeadDim];
  __shared__ T sK_ptr[Block_N * HeadDim];
  __shared__ T sV_ptr[Block_N * HeadDim];

  // TODO: 显存布局没有考虑 bank conflict 问题, 应该自定义 swizzle 布局
  auto sQ = make_tensor(
      make_smem_ptr(sQ_ptr),
      make_layout(make_shape(Int<Block_M>{}, Int<HeadDim>{}), GenRowMajor{}));

  auto sK = make_tensor(
      make_smem_ptr(sK_ptr),
      make_layout(make_shape(Int<Block_N>{}, Int<HeadDim>{}), GenRowMajor{}));

  auto sV = make_tensor(
      make_smem_ptr(sV_ptr),
      make_layout(make_shape(Int<Block_N>{}, Int<HeadDim>{}), GenRowMajor{}));

  // TODO: 为什么需要转置? 暂时不是很懂
  auto sV_T = make_tensor(
      make_smem_ptr(sV_ptr),
      make_layout(make_shape(Int<HeadDim>{}, Int<Block_N>{}), GenRowMajor{}));

  // =================== 3 ===================
  // define tensor for current thread to copy global memory to shared memory
  TiledCopy tiled_copy;
  // 一个 tiled block，当前线程负责那几个元素。
  auto thr_copy = tiled_copy.get_slice(tx);

  // Copy: 一次拷贝多少个元素
  // BlockMCopy: M 方向执行多少次 copy
  // HeadDimCopy: K 方向执行多少次 coNy

  // 假设 Tile 是 128x64，128 个线程。总元素 8192。
  // 每个线程搬运 8192 / 128 = 64 个元素。
  // 如果使用 CP_ASYNC (一次搬 8 个)，每个线程执行 8 次指令。
  // 那么 tQgQ 和 tQsQ 的 Shape 看起来可能是 ((8), 4, 2) 或类似结构，表示：
  // (8): 一次指令搬 8 个。
  // 4: 在 M 维度上重复 4 次。
  // 2: 在 K 维度上重复 2 次。
  // (注：具体是 M 还是 K 维度重复，取决于 TiledCopy 的线程排布)

  auto tQgQ = thr_copy.partition_S(gQ);  // (Copy, BlockMCopy, HeadDimCopy)
  auto tQsQ = thr_copy.partition_D(sQ);  // (Copy, BlockMCopy, HeadDimCopy)

  // (Copy, BlockNCopy, HeadDimCopy, Num_Blocks)
  auto tKgK = thr_copy.partition_S(gK);
  auto tKsK = thr_copy.partition_D(sK);  // (Copy, BlockNCopy, HeadDimCopy)

  // (Copy, BlockNCopy, HeadDimCopy, Num_Blocks)
  auto tVgV = thr_copy.partition_S(gV);
  auto tVsV = thr_copy.partition_D(sV);  // (Copy, BlockNCopy, HeadDimCopy)

  // =================== 4 ===================
  // 定义每个线程负责的 tensor，这个 tensor 是在寄存器中的布局
  // 跟 mma 指令有关，每个线程负载加载部分 Q, K, V 元素到自己的寄存器中
  // 还有部分 S 和 O 在寄存器中，S 暂存 sore，O 是最后计算的值
  // 最后会统一拷贝到 shared memory
  //
  // tSrQ Shape: ((Atom_Val_A), Rep_M, Rep_K)
  // MMA (Atom_Val_A): 单次 MMA 指令中，一个线程负责的 A 矩阵元素个数（例如 FP16
  // 下 16x8x16 mma，A 矩阵由 4 个寄存器组成）。
  // MMA_QO (Rep_M): 整个 BlockQO 被 Atom 的 M 维度切分后的份数。如果
  // BlockQO=128, Atom_M=16, 则此处为  128 / 16 = 8. MMA_HEAD (Rep_K): HeadDim
  // 被 Atom 的 K 维度切分后的份数。
  //
  TiledMMA tiled_mma;
  auto thr_mma = tiled_mma.get_slice(tx);
  auto tSrQ = thr_mma.partition_fragment_A(sQ);  // (MMA, MMA_QO, MMA_HEAD)
  auto tSrK = thr_mma.partition_fragment_B(sK);  // (MMA, MMA_KV, MMA_HEAD)
  auto tSrS = partition_fragment_C(
      tiled_mma,
      Shape<Int<Block_M>, Int<Block_N>>{});  // (MMA, MMA_KV, MMA_HEAD);
  // TODO: why?
  auto tOrV_T =
      thr_mma.partition_fragment_B(sV_T);  // (MMA, MMA_Headdim, MMA_KV)
  auto tOrO = partition_fragment_C(
      tiled_mma,
      Shape<Int<Block_M>, Int<HeadDim>>{});  // (MMA, MMA_QO, MMA_Headdim)
  clear(tOrO);

  // =================== 5 ===================
  // 等内存拷贝完以后，需要构建 shared memory 拷贝到寄存器的 tiled_copy,
  // 这里有一个需要注意的点是：MMA 寄存器中每一个线程中的元素布局
  // 和 copy 到集群中的布局对应不上，所以需要使用 make_tiled_copy_A
  // 自动调整布局
  auto tiled_s2r_copy_Q = make_tiled_copy_A(SmemCopyAtom{}, tiled_mma);
  auto thr_s2r_copy_Q = tiled_s2r_copy_Q.get_slice(tx);
  auto tXsQ = thr_s2r_copy_Q.partition_S(sQ);

  // copy 和 mma 的视图是不一样的，tXrQ 和 tSrQ 是同一块寄存器的不同视图
  auto tXrQ = thr_s2r_copy_Q.retile_D(tSrQ);  // (CPY, MMA_QO, MMA_HEAD)

  auto tiled_s2r_copy_K = make_tiled_copy_B(SmemCopyAtom{}, tiled_mma);
  auto thr_s2r_copy_K = tiled_s2r_copy_K.get_slice(tx);
  auto tXsK = thr_s2r_copy_K.partition_S(sK);
  auto tXrK = thr_s2r_copy_K.retile_D(tSrK);  // (CPY, MMA_KV, MMA_HEAD)

  auto tiled_s2r_copy_V = make_tiled_copy_B(SmemCopyAtom_T{}, tiled_mma);
  auto thr_s2r_copy_V = tiled_s2r_copy_V.get_slice(tx);
  auto tXsV_T = thr_s2r_copy_V.partition_S(sV_T);
  auto tXrV_T = thr_s2r_copy_V.retile_D(tOrV_T);  // (CPY, MMA_Headdim, MMA_QO)

#ifdef FLASH_ATTN_MMA_DEBUG
  if (thread0()) {  // clang-format off
    print("NumThreads: "); print(FlashAttnConfig_::NumThreads); print("\n");
    print("tiled_mma: "); print(tiled_mma); print("\n");
    print("tiled_copy: "); print(tiled_copy); print("\n");
    print("GmemValsPerLoad: "); print(FlashAttnConfig_::GmemValsPerLoad); print("\n");
    print("GmemThreadsPerRow: "); print(FlashAttnConfig_::GmemThreadsPerRow); print("\n");
    print("gQ: "); print(gQ.layout()); print("\n");
    print("gK: "); print(gK.layout()); print("\n");
    print("gV: "); print(gV.layout()); print("\n");
    print("sQ: "); print(sQ.layout()); print("\n");
    print("sK: "); print(sK.layout()); print("\n");
    print("sV: "); print(sV.layout()); print("\n");

    print("tQgQ: "); print(tQgQ.layout()); print("\n");
    print("tQsQ: "); print(tQsQ.layout()); print("\n");
    print("tKsK: "); print(tKsK.layout()); print("\n");
    print("tKgK: "); print(tKgK.layout()); print("\n");
    print("tVsV: "); print(tVsV.layout()); print("\n");

    print("tSrQ: "); print(tSrQ.layout()); print("\n");
    print("tSrK: "); print(tSrK.layout()); print("\n");
    print("tSrS: "); print(tSrS.layout()); print("\n");
    print("tOrV_T: "); print(tOrV_T.layout()); print("\n");
    print("tOrO: "); print(tOrO.layout()); print("\n");

    print("tiled_s2r_copy_Q: "); print(tiled_s2r_copy_Q); print("\n");
    print("tXsQ: "); print(tXsQ.layout()); print("\n");
    print("tXrQ: "); print(tXrQ.layout()); print("\n");
    print("tiled_s2r_copy_K: "); print(tiled_s2r_copy_K); print("\n");
    print("tXsK: "); print(tXsK.layout()); print("\n");
    print("tXrK: "); print(tXrK.layout()); print("\n");
    print("tiled_s2r_copy_V: "); print(tiled_s2r_copy_V); print("\n");
    print("tXsV_T: "); print(tXsV_T.layout()); print("\n");
    print("tXrV_T: "); print(tXrV_T.layout()); print("\n");
  }  // clang-format on
#endif

  // =================== 6 ===================
  // 初始化 max 和 分母
  // TODO: for sm80 MMA, each thread owns 2 rows of C matrix, they are
  // [v0, v1]
  // ......
  // [v2, v3]
  // 单个 MMA 16x8x16 输出
  // ┌───────────────────────────────────────────────────────────────────┐
  // │              MMA 16x8x16 输出矩阵 (16行 x 8列)                      │
  // ├───────────────────────────────────────────────────────────────────┤
  // │                                                                   │
  // │            col0  col1  col2  col3  col4  col5  col6  col7         │
  // │          ┌─────┬─────┬─────┬─────┬─────┬─────┬─────┬─────┐        │
  // │   row 0  │ T0  │ T1  │ T2  │ T3  │ T0  │ T1  │ T2  │ T3  │        │
  // │   row 1  │ T0  │ T1  │ T2  │ T3  │ T0  │ T1  │ T2  │ T3  │        │
  // │          ├─────┼─────┼─────┼─────┼─────┼─────┼─────┼─────┤        │
  // │   row 2  │ T4  │ T5  │ T6  │ T7  │ T4  │ T5  │ T6  │ T7  │        │
  // │   row 3  │ T4  │ T5  │ T6  │ T7  │ T4  │ T5  │ T6  │ T7  │        │
  // │          ├─────┼─────┼─────┼─────┼─────┼─────┼─────┼─────┤        │
  // │   row 4  │ T8  │ T9  │ T10 │ T11 │ T8  │ T9  │ T10 │ T11 │        │
  // │   row 5  │ T8  │ T9  │ T10 │ T11 │ T8  │ T9  │ T10 │ T11 │        │
  // │          ├─────┼─────┼─────┼─────┼─────┼─────┼─────┼─────┤        │
  // │   ...    │ ... │ ... │ ... │ ... │ ... │ ... │ ... │ ... │        │
  // │          ├─────┼─────┼─────┼─────┼─────┼─────┼─────┼─────┤        │
  // │   row 14 │ T28 │ T29 │ T30 │ T31 │ T28 │ T29 │ T30 │ T31 │        │
  // │   row 15 │ T28 │ T29 │ T30 │ T31 │ T28 │ T29 │ T30 │ T31 │        │
  // │          └─────┴─────┴─────┴─────┴─────┴─────┴─────┴─────┘        │
  // │                                                                   │
  // │   关键观察: 每一行由 4 个连续线程 (T0-T3, T4-T7, ...) 共同持有         │
  // │                                                                   │
  // └───────────────────────────────────────────────────────────────────┘

  // pre_row_max 的 shape 为什么是 2 和 size<1>()
  // 2 是因为 单个 mma 的指令中，每个线程会参与 2 行元素的计算，如果在 N
  // 的方向有 重复，最终也会被规约。 Int<size<1>()>{tSrS}) 是因为在 M
  // 的方向会有多个重复的 MMA。所以 prev_row_max 的第一维表示的是在一个 tile
  // 中，当前在每一个 mma 中负责的两行最大值，第 2 维表示 在 M 方向有多个 mma。
  auto max_i = make_tensor<float>(make_shape(_2{}, Int<size<1>(tSrS)>{}));
  fill(max_i, -1e20);
  auto l_i = make_tensor<float>(make_shape(_2{}, Int<size<1>(tSrS)>{}));
  fill(l_i, 0);

  // =================== 7 ===================
  // 拷贝数据 Q，并做 scaler
  //  TODO: 没有做流水线
  copy(tiled_copy, tQgQ, tQsQ);
  for (int i = 0; i < size(tQsQ); ++i) {
    tQsQ(i) = static_cast<T>(scaler) * tQsQ(i);
  }
  __syncthreads();  // 至此，一个 Q 的 block 全局在 shared memory，而且已经
                    // scale
  copy(tiled_s2r_copy_Q, tXsQ, tXrQ);

  // =================== 8 ===================
  // 循环计算每一个 kv block
  for (int blkKVIdx = 0; blkKVIdx < size<2>(gK); ++blkKVIdx) {
    // TODO: mask masual

    // copy K into smem
    __syncthreads();
    copy(tiled_copy, tKgK(_, _, _, blkKVIdx), tKsK);
    __syncthreads();
    copy(tiled_s2r_copy_K, tKsK, tXrK);
#ifdef FLASH_ATTN_MMA_DEBUG
    if (thread0()) {  // clang-format off
      print("blkKVIdx: "); print(blkKVIdx); print("\n");
      print("tXrQ: "); print_tensor(tXrQ); print("\n");
      print("tSrQ: "); print_tensor(tSrQ); print("\n");
      print("tXrK: "); print_tensor(tXrK); print("\n");
      print("tSrK: "); print_tensor(tSrK); print("\n");
    }  // clang-format on
#endif
    // 每次循环必须清理 tSrS, tSrS 只计算一个 KV block。
    clear(tSrS);
    // 先计算 Q@K_T, tSrS 只
    gemm(tiled_mma, tSrQ, tSrK, tSrS);
#ifdef FLASH_ATTN_MMA_DEBUG
    if (thread0()) {  // clang-format off
      print("tSrS: "); print_tensor(tSrS); print("\n");
    }  // clang-format on
#endif
    // 先计算当前的最大 Max
    auto max_ij = make_fragment_like(max_i);
    fill(max_ij, -1e20);
    for (int val_idx = 0; val_idx < size<0>(tSrS); ++val_idx) {
      for (int row_idx = 0; row_idx < size<1>(tSrS); ++row_idx) {
        for (int col_idx = 0; col_idx < size<2>(tSrS); ++col_idx) {
          int max_row_idx = val_idx / 2;
          int max_col_idx = row_idx;
          max_ij(max_row_idx, max_col_idx) =
              max(max_ij(max_row_idx, max_col_idx),
                  tSrS(val_idx, row_idx, col_idx));
        }
      }
    }
#ifdef FLASH_ATTN_MMA_DEBUG
    if (thread0()) {  // clang-format off
      print("local max_ij: "); print_tensor(max_ij); print("\n");
    }  // clang-format on
#endif
    // 规约当前行最大的值(一个大的 tile 的一行只有 4 个线程，所以只需要 2 次 xor
    // 就能完成)
    for (int max_row_idx = 0; max_row_idx < size<0>(max_ij); ++max_row_idx) {
      for (int max_col_idx = 0; max_col_idx < size<1>(tSrS); ++max_col_idx) {
        max_ij(max_row_idx, max_col_idx) = max(
            max_ij(max_row_idx, max_col_idx),
            __shfl_xor_sync(0xffffffff, max_ij(max_row_idx, max_col_idx), 1));
        max_ij(max_row_idx, max_col_idx) = max(
            max_ij(max_row_idx, max_col_idx),
            __shfl_xor_sync(0xffffffff, max_ij(max_row_idx, max_col_idx), 2));
      }
    }
#ifdef FLASH_ATTN_MMA_DEBUG
    if (thread0()) {  // clang-format off
      print("quad max_ij: "); print_tensor(max_ij); print("\n");
    }  // clang-format on
#endif
    // 现在已经有当前 tile 的最大值，计算之前所有的 tile 的最大值
    // 为什么不更新全局最大值，是因为现在的最大值还在寄存器里面，计算是最快的。
    for (int max_row_idx = 0; max_row_idx < size<0>(max_ij); ++max_row_idx) {
      for (int max_col_idx = 0; max_col_idx < size<0>(max_ij); ++max_col_idx) {
        max_ij(max_row_idx, max_col_idx) = max(max_ij(max_row_idx, max_col_idx),
                                               max_i(max_row_idx, max_col_idx));
      }
    }
#ifdef FLASH_ATTN_MMA_DEBUG
    if (thread0()) {  // clang-format off
      print("max_ij: "); print_tensor(max_ij); print("\n");
    }  // clang-format on
#endif
    // 使用当前最大值计算补偿。 1 是最后的结果，2 是当前的分母。
    for (int val_idx = 0; val_idx < size<0>(tOrO); ++val_idx) {
      for (int row_idx = 0; row_idx < size<1>(tOrO); ++row_idx) {
        for (int col_idx = 0; col_idx < size<2>(tOrO); ++col_idx) {
          int max_row_idx = val_idx / 2;
          int max_col_idx = row_idx;
          tOrO(val_idx, row_idx, col_idx) *=
              exp(max_i(max_row_idx, max_col_idx) -
                  max_ij(max_row_idx, max_col_idx));
        }
      }
    }
    // 2. 补偿分母
    for (int max_row_idx = 0; max_row_idx < size<0>(max_ij); ++max_row_idx) {
      for (int max_col_idx = 0; max_col_idx < size<1>(max_ij); ++max_col_idx) {
        l_i(max_row_idx, max_col_idx) *= exp(max_i(max_row_idx, max_col_idx) -
                                             max_ij(max_row_idx, max_col_idx));
      }
    }

    // 3. sore 的 softmax 计算，分母的累加
    for (int val_idx = 0; val_idx < size<0>(tSrS); ++val_idx) {
      for (int row_idx = 0; row_idx < size<1>(tSrS); ++row_idx) {
        for (int col_idx = 0; col_idx < size<2>(tSrS); ++col_idx) {
          int max_row_idx = val_idx / 2;
          int max_col_idx = row_idx;
          tSrS(val_idx, row_idx, col_idx) =
              exp(tSrS(val_idx, row_idx, col_idx) -
                  max_ij(max_row_idx, max_col_idx));
          l_i(max_row_idx, max_col_idx) += tSrS(val_idx, row_idx, col_idx);
        }
      }
    }
#ifdef FLASH_ATTN_MMA_DEBUG
    if (thread0()) {  // clang-format off
      print("scaled tSrS: "); print_tensor(tSrS); print("\n");
      print("global_row_denominator: "); print_tensor(global_row_denominator); print("\n");
    }  // clang-format on
#endif
    // 现在用不着最大值了，在这里进行更新。
    // TODO: 循环 tSrS? why
    for (int max_row_idx = 0; max_row_idx < size<0>(max_ij); ++max_row_idx) {
      for (int max_col_idx = 0; max_col_idx < size<0>(max_ij); ++max_col_idx) {
        max_i(max_row_idx, max_col_idx) = max_ij(max_row_idx, max_col_idx);
      }
    }

    // TODO: 将 tSrS 的转成 T 的类型，因为 mma 返回的数据类型可能跟 T 对应不上
    auto tOrS = make_tensor<T>(tSrS.layout());
    for (int i = 0; i < size(tOrS); ++i) {
      tOrS(i) = static_cast<T>(tSrS(i));
    }

    // calculate numerator
    static_assert(tiled_mma.get_layoutA_TV() == tiled_mma.get_layoutC_TV(),
                  "This is only valid for atom mnk == (16, 8, 8), otherwise we "
                  "will have different A and C layout and need to adjust the "
                  "layout accordingly");

    // 等待所有线程都完成第一步。
    __syncthreads();
    // TODO: 应该使用流水线，现在才从全局拷贝到 shared
    // memory，中间会有很大的时间空隙。
    copy(tiled_copy, tVgV(_, _, _, blkKVIdx), tVsV);
    __syncthreads();

    // 这里使用转置的原因是 mma 指令第二个参数是一个转置的值，但是
    // S @ V 是不需要转置的，所以我们提前转置，再体用指令转回来，这里只是一个
    // view， 在物理上并没有真正的数据搬运，这就是 cute 的核心用法。
    copy(tiled_s2r_copy_V, tXsV_T, tXrV_T);
#ifdef FLASH_ATTN_MMA_DEBUG
    if (thread0()) {  // clang-format off
      print("tOrV_T: "); print_tensor(tOrV_T); print("\n");
    }  // clang-format on
#endif
    gemm(tiled_mma, tOrS, tOrV_T, tOrO);
#ifdef FLASH_ATTN_MMA_DEBUG
    if (thread0()) {  // clang-format off
      print("tOrO: "); print_tensor(tOrO); print("\n");
    }  // clang-format on
#endif
  }

  // 每一个线程都有自己的分母 fragment，现在把他们进行规约，形成最终的分母。
  // TODO: 循环 tSrS? why
  for (int row_idx = 0; row_idx < size<0>(max_i); ++row_idx) {
    for (int col_idx = 0; col_idx < size<0>(max_i); ++col_idx) {
      l_i(row_idx, col_idx) +=
          __shfl_xor_sync(0xffffffff, l_i(row_idx, col_idx), 1);
      l_i(row_idx, col_idx) +=
          __shfl_xor_sync(0xffffffff, l_i(row_idx, col_idx), 2);
    }
  }

  // eplilogue
  for (int val_idx = 0; val_idx < size<0>(tOrO); ++val_idx) {
    for (int row_idx = 0; row_idx < size<1>(tOrO); ++row_idx) {
      for (int col_idx = 0; col_idx < size<2>(tOrO); ++col_idx) {
        int l_row_idx = val_idx / 2;
        int l_col_idx = row_idx;
        tOrO(val_idx, row_idx, col_idx) /= l_i(l_row_idx, l_col_idx);
      }
    }
  }
#ifdef FLASH_ATTN_MMA_DEBUG
  if (thread0()) {  // clang-format off
    print("l_i: "); print_tensor(l_i); print("\n");
    print("tOrO: "); print_tensor(tOrO); print("\n");
  }  // clang-format on
#endif
  // 搬运最后的计算结果到全局显存中去，
  // TODO: 应该先搬运到 shared memory，然后在统一般到全局显存。
  auto tiled_r2s_copy_O = make_tiled_copy_C(SmemCopyAtomO{}, tiled_mma);
  auto thr_r2s_copy_O = tiled_r2s_copy_O.get_slice(tx);
  auto tXrO = thr_r2s_copy_O.retile_S(tOrO);
  auto tXgO = thr_r2s_copy_O.partition_D(gO);

  copy(tiled_r2s_copy_O, tXrO, tXgO);
}

// this kernel only implement limited functionality
static bool sanity_check(torch::Tensor Q, torch::Tensor K, torch::Tensor V,
                         torch::Tensor O) {
  const int bq = Q.size(0);  // B, H, N, d
  const int hq = Q.size(1);
  const int nq = Q.size(2);
  const int dq = Q.size(3);
  const int bk = K.size(0);  // B, H, N, d
  const int hk = K.size(1);
  const int nk = K.size(2);
  const int dk = K.size(3);
  const int bv = V.size(0);  // B, H, N, d
  const int hv = V.size(1);
  const int nv = V.size(2);
  const int dv = V.size(3);
  const int bo = O.size(0);  // B, H, N, d
  const int ho = O.size(1);
  const int no = O.size(2);
  const int do_ = O.size(3);
  if (!(bq == bk && bq == bv && bq == bo)) {
    printf("batch size mismatch: %d %d %d %d\n", bq, bk, bv, bo);
    fflush(stdout);
    return false;
  }
  if (!(hq == hk && hq == hv && hq == ho)) {
    printf("head size mismatch: %d %d %d %d\n", hq, hk, hv, ho);
    fflush(stdout);
    return false;
  }
  if (!(nq == nk && nq == nv && nq == no)) {
    printf("sequence length mismatch: %d %d %d %d, only self-attn is tested\n",
           nq, nk, nv, no);
    fflush(stdout);
    return false;
  }
  if (!(dq == dk && dq == dv && dq == do_)) {
    printf("hidden size mismatch: %d %d %d %d\n", dq, dk, dv, do_);
    fflush(stdout);
    return false;
  }
  return true;
}

template <int BlockQO, int BlockKV, int HeadDim, int NWarpsPerSM>
static void launch_kernel(torch::Tensor Q, torch::Tensor K, torch::Tensor V,
                          torch::Tensor O) {
  using config =
      FlashAttnConfig_<half_t, BlockQO, BlockKV, HeadDim, NWarpsPerSM>;

  assert(sanity_check(Q, K, V, O));
  const int b = Q.size(0);  // B, H, N, d
  const int h = Q.size(1);
  const int n = Q.size(2);
  const int d = Q.size(3);

  float scaler = 1.0 / sqrt(d);
  // TODO: ceil
  assert(n % BlockQO == 0);
  dim3 grid(b, h, n / BlockQO);
  dim3 block(size(config::NumThreads));
  flash_attn_v2_kernel<config><<<grid, block>>>(
      reinterpret_cast<half_t*>(Q.data_ptr()),
      reinterpret_cast<half_t*>(K.data_ptr()),
      reinterpret_cast<half_t*>(V.data_ptr()),
      reinterpret_cast<half_t*>(O.data_ptr()), b, h, n, n, d, scaler);
  CUDACHECK(cudaGetLastError());
}

void flash_attn_v2(torch::Tensor Q, torch::Tensor K, torch::Tensor V,
                   torch::Tensor O) {
  // TODO: 没有这个宏定义
  //   CHECK_TORCH_TENSOR_DTYPE(Q, torch::kHalf)  // Q [B,H,N,D]
  //   CHECK_TORCH_TENSOR_DTYPE(K, torch::kHalf)  // K [B,H,N,D]
  //  CHECK_TORCH_TENSOR_DTYPE(V, torch::kHalf)  // V [B,H,N,D]
  //    CHECK_TORCH_TENSOR_DTYPE(O, torch::kHalf)  // O [B,H,N,D]
  const int d = Q.size(3);

  // TODO: 最优的配置
  switch (d) {
    case 16:
      launch_kernel<128, 128, 16, 8>(Q, K, V, O);
      break;
    case 32:
      launch_kernel<128, 128, 32, 8>(Q, K, V, O);
      break;
    case 64:
      launch_kernel<128, 128, 64, 8>(Q, K, V, O);
      break;
    case 128:
      launch_kernel<64, 64, 128, 4>(Q, K, V, O);
      break;
    case 256:
      launch_kernel<32, 32, 256, 2>(Q, K, V, O);
      break;
    default:
      throw std::runtime_error("Unsupported headdim");
  }
}