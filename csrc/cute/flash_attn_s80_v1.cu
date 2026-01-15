#include <torch/all.h>
#include <torch/library.h>

#include <cutlass/fast_math.h>
#include <cute/layout.hpp>
#include <cute/tensor.hpp>

#include "../utils.h"

using namespace cute;

// Flash Attention v2: online softmax algorithm, O(N) memory instead of O(N^2)
// Key: Each row handled by 4 threads (T0-T3), reduction across these 4 threads
template <typename T_, int BlockQO_, int BlockKV_, int HeadDim_,
          int NWarpsPerSM_>
struct FlashAttnConfig {
  using T = T_;

  // TODO: Define stride parameters for flexible tensor layouts
  // int stride_qb;
  // int stride_kb;
  // int stride_vb;
  // int stride_ob;

  // int stride_qh;
  // int stride_kh;
  // int stride_vh;
  // int stride_oh;

  // int stride_qm;
  // int stride_km;
  // int stride_vm;
  // int stride_om;

  // int stride_qk;
  // int stride_kk;
  // int stride_vk;
  // int stride_ok;

  // Number of warps per SM and total threads per block
  static constexpr int NWarpsPerSM = NWarpsPerSM_;
  static constexpr int NumThreads = NWarpsPerSM * 32;  // 32 threads per warp

  // Block sizes for tiling
  static constexpr int BlockQO = BlockQO_;  // Block size for Q and O
  static constexpr int BlockKV = BlockKV_;  // Block size for K and V
  static constexpr int HeadDim = HeadDim_;  // Head dimension

  // Number of values each thread loads per instruction (128-bit aligned load)
  // Calculated based on element type: 128 bits / sizeof(T) elements
  static constexpr int GmemValsPerLoad = sizeof(uint128_t) / sizeof(T);
  // Number of threads needed per row to cover HeadDim elements
  static constexpr int GmemThreadsPerRow = HeadDim / GmemValsPerLoad;

  // Copy atom for global memory to shared memory transfer
  using GmemCopyAtom =
      Copy_Atom<AutoVectorizingCopyWithAssumedAlignment<sizeof(uint128_t) * 8>,
                T>;

  // Tiled copy: copy atom, thread layout, instructions per thread
  using TiledCopyQKVO = decltype(make_tiled_copy(
      GmemCopyAtom{},
      make_layout(
          Shape<Int<NumThreads / GmemThreadsPerRow>, Int<GmemThreadsPerRow>>{},
          GenRowMajor{}),
      make_layout(Shape<_1, Int<GmemValsPerLoad>>{}, GenRowMajor{})));

  static_assert(Int<NumThreads / GmemThreadsPerRow>::value <= BlockQO,
                "NumThreads must be less than or equal to BlockQO");

  // LDSM copy atom for shared memory to register (MMA-compatible format)
  using SmemCopyAtom = Copy_Atom<SM75_U32x4_LDSM_N, T>;

  // Transposed LDSM for V: MMA expects transposed B, but we compute S @ V
  using SmemCopyAtomTransposed = Copy_Atom<SM75_U16x8_LDSM_T, T>;

  // Copy atom for writing results to global memory
  using SmemCopyAtomO =
      Copy_Atom<AutoVectorizingCopyWithAssumedAlignment<sizeof(uint128_t) * 8>,
                T>;

  static_assert(std::is_same_v<T, half_t> || std::is_same_v<T, bfloat16_t>);

  // MMA atom: 16x8x8 for simplicity (same layout for Q@K^T and S@V)
  using MMA_Atom = std::conditional_t<std::is_same_v<T, half_t>,
                                      MMA_Atom<SM80_16x8x8_F32F16F16F32_TN>,
                                      MMA_Atom<SM80_16x8x8_F32BF16BF16F32_TN>>;

  // Tiled MMA: (16 * NWarpsPerSM) × 16 × 16 to cover BlockQO × BlockKV ×
  // HeadDim
  using TiledMMA = decltype(make_tiled_mma(
      MMA_Atom{}, make_layout(Shape<Int<NWarpsPerSM>, _1, _1>{}, GenRowMajor{}),
      Tile<Int<16 * NWarpsPerSM>, _16, _16>{}));

  static_assert(
      16 * NWarpsPerSM <= BlockQO && 16 <= BlockKV && 16 <= HeadDim,
      "BlockQO, BlockKV, and HeadDim must be greater than or equal to "
      "16 * NWarpsPerSM, 16, and 16 respectively");

  // Sanity checks: ensure thread counts match across all operations
  static_assert(size(TiledMMA{}) == NumThreads &&
                    size(TiledMMA{}) == size(TiledCopyQKVO{}),
                "Thread count mismatch between TiledMMA and TiledCopyQKVO");
};

// Flash Attention v2 kernel: O = softmax(Q @ K^T / sqrt(d)) @ V
// Uses online softmax to avoid storing full attention matrix
template <typename FlashAttnConfig_>
__global__ void flash_attn_v2_kernel(typename FlashAttnConfig_::T* Q_ptr,
                                     typename FlashAttnConfig_::T* K_ptr,
                                     typename FlashAttnConfig_::T* V_ptr,
                                     typename FlashAttnConfig_::T* O_ptr, int B,
                                     int H, int N_QO_CTX, int N_KV_CTX, int D,
                                     float scaler) {
  using namespace cute;

  // Extract data type from config
  using T = typename FlashAttnConfig_::T;

  // Block size for Q and O are the same (M dimension)
  // Block size for K and V are the same (KV dimension)
  constexpr int BlockQO = FlashAttnConfig_::BlockQO;
  constexpr int BlockKV = FlashAttnConfig_::BlockKV;
  constexpr int HeadDim = FlashAttnConfig_::HeadDim;

  // the tiledCopy to copy global memory to shared memory
  // TODO: use async copy
  using TiledCopy = typename FlashAttnConfig_::TiledCopyQKVO;

  // copy shared memory to register
  using SmemCopyAtom = typename FlashAttnConfig_::SmemCopyAtom;

  // TODO: Explain why transposed copy atom is needed
  using SmemCopyAtom_T = typename FlashAttnConfig_::SmemCopyAtomTransposed;

  // Copy atom for output: copy O to shared memory first for better performance
  using SmemCopyAtomO = typename FlashAttnConfig_::SmemCopyAtomO;

  using TiledMMA = typename FlashAttnConfig_::TiledMMA;

  // TODO: static check
  assert(HeadDim == D);

  // Block indices: x for batch (B), y for head (H), z for sequence block (N)
  const int bx = blockIdx.x;
  const int by = blockIdx.y;
  const int bz = blockIdx.z;

  // Thread index within the block
  const int tx = threadIdx.x;

  // Step 1: Define global memory tensors [Batch, Head, Sequence, HeadDim]
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

  // Extract local tiles: gQ/gO [BlockQO, HeadDim], gK/gV [BlockKV, HeadDim,
  // Num_Blocks]
  auto gQ =
      local_tile(Q, make_shape(_1{}, _1{}, Int<BlockQO>{}, Int<HeadDim>{}),
                 make_coord(bx, by, bz, 0))(0, 0, _, _);
  auto gO =
      local_tile(O, make_shape(_1{}, _1{}, Int<BlockQO>{}, Int<HeadDim>{}),
                 make_coord(bx, by, bz, 0))(0, 0, _, _);

  // Using _ in SeqLen preserves block index: [Batch(1), Head(1), BlockKV,
  // HeadDim, Num_Blocks]
  auto gK =
      local_tile(K, make_shape(_1{}, _1{}, Int<BlockKV>{}, Int<HeadDim>{}),
                 make_coord(bx, by, _, 0))(0, 0, _, _, _);
  auto gV =
      local_tile(V, make_shape(_1{}, _1{}, Int<BlockKV>{}, Int<HeadDim>{}),
                 make_coord(bx, by, _, 0))(0, 0, _, _, _);

  // __shared__ T psQ[BlockQO * HeadDim], psK[BlockKV * HeadDim],
  //     psV[BlockKV * HeadDim];

  // Step 2: Define shared memory layout for Q, K, V blocks
  __shared__ T sQ_ptr[BlockQO * HeadDim];
  __shared__ T sK_ptr[BlockKV * HeadDim];
  __shared__ T sV_ptr[BlockKV * HeadDim];
  auto sQ = make_tensor(
      make_smem_ptr(sQ_ptr),
      make_layout(make_shape(Int<BlockQO>{}, Int<HeadDim>{}), GenRowMajor{}));

  auto sK = make_tensor(
      make_smem_ptr(sK_ptr),
      make_layout(make_shape(Int<BlockKV>{}, Int<HeadDim>{}), GenRowMajor{}));

  auto sV = make_tensor(
      make_smem_ptr(sV_ptr),
      make_layout(make_shape(Int<BlockKV>{}, Int<HeadDim>{}), GenRowMajor{}));

  // Transposed view of V: view-only (no copy), MMA expects transposed B but we
  // compute S @ V
  auto sVt = make_tensor(
      make_smem_ptr(sV_ptr),
      make_layout(make_shape(Int<HeadDim>{}, Int<BlockKV>{}), GenColMajor{}));

  // Step 3: Define thread partitions for global->shared copy
  TiledCopy tiled_copy;
  auto thr_copy = tiled_copy.get_slice(tx);

  auto tQgQ = thr_copy.partition_S(gQ);  // (Copy, BlockQOCopy, HeadDimCopy)
  auto tQsQ = thr_copy.partition_D(sQ);  // (Copy, BlockQOCopy, HeadDimCopy)

  // (Copy, BlockKVCopy, HeadDimCopy, RestKV)
  auto tKgK = thr_copy.partition_S(gK);
  auto tKsK = thr_copy.partition_D(sK);  // (Copy, BlockKVCopy, HeadDimCopy)

  // (Copy, BlockKVCopy, HeadDimCopy, RestKV)
  auto tVgV = thr_copy.partition_S(gV);
  auto tVsV = thr_copy.partition_D(sV);

  // Step 4: Define register fragments for MMA operations
  // tSr*: register fragment from shared memory, tOr*: output fragment
  TiledMMA tiled_mma;
  auto thr_mma = tiled_mma.get_slice(tx);

  // Register fragments for Q and K (used in Q @ K^T computation)
  auto tSrQ =
      thr_mma.partition_fragment_A(sQ);  // Q fragment: (MMA, Rep_M, Rep_K)
  auto tSrK =
      thr_mma.partition_fragment_B(sK);  // K fragment: (MMA, Rep_KV, Rep_K)

  // Register fragment for attention scores S = Q @ K^T
  auto tSrS = partition_fragment_C(
      tiled_mma,
      Shape<Int<BlockQO>, Int<BlockKV>>{});  // S: (MMA, Rep_M, Rep_KV)

  // Register fragments for output computation O = S @ V
  // V is transposed because MMA expects transposed B operand, but we want S @ V
  auto tOrVt = thr_mma.partition_fragment_B(
      sVt);  // V^T fragment: (MMA, Rep_HeadDim, Rep_KV)
  auto tOrO = partition_fragment_C(
      tiled_mma, Shape<Int<BlockQO>, Int<HeadDim>>{});  // O fragment: (MMA,
                                                        // Rep_M, Rep_HeadDim)

  // Initialize output to zero (will accumulate results from each KV block)
  clear(tOrO);

  // Step 5: Define shared memory to register copy operations
  // Layout automatically adjusted to match MMA requirements
  auto tiled_s2r_copy_Q = make_tiled_copy_A(SmemCopyAtom{}, tiled_mma);
  auto thr_s2r_copy_Q = tiled_s2r_copy_Q.get_slice(tx);
  auto tXsQ = thr_s2r_copy_Q.partition_S(sQ);
  auto tXrQ = thr_s2r_copy_Q.retile_D(tSrQ);  // (CPY, MMA_QO, MMA_HEAD)

  auto tiled_s2r_copy_K = make_tiled_copy_B(SmemCopyAtom{}, tiled_mma);
  auto thr_s2r_copy_K = tiled_s2r_copy_K.get_slice(tx);
  auto tXsK = thr_s2r_copy_K.partition_S(sK);
  auto tXrK = thr_s2r_copy_K.retile_D(tSrK);  // (CPY, MMA_KV, MMA_HEAD)

  auto tiled_s2r_copy_V = make_tiled_copy_B(SmemCopyAtom_T{}, tiled_mma);
  auto thr_s2r_copy_V = tiled_s2r_copy_V.get_slice(tx);
  auto tXsVt = thr_s2r_copy_V.partition_S(sVt);
  auto tXrVt = thr_s2r_copy_V.retile_D(tOrVt);  // (CPY, MMA_Headdim, MMA_QO)

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
    print("tOrVt: "); print(tOrVt.layout()); print("\n");
    print("tOrO: "); print(tOrO.layout()); print("\n");

    print("tiled_s2r_copy_Q: "); print(tiled_s2r_copy_Q); print("\n");
    print("tXsQ: "); print(tXsQ.layout()); print("\n");
    print("tXrQ: "); print(tXrQ.layout()); print("\n");
    print("tiled_s2r_copy_K: "); print(tiled_s2r_copy_K); print("\n");
    print("tXsK: "); print(tXsK.layout()); print("\n");
    print("tXrK: "); print(tXrK.layout()); print("\n");
    print("tiled_s2r_copy_V: "); print(tiled_s2r_copy_V); print("\n");
    print("tXsVt: "); print(tXsVt.layout()); print("\n");
    print("tXrVt: "); print(tXrVt.layout()); print("\n");
  }  // clang-format on
#endif
  // Step 6: Initialize online softmax state
  // max_i: running maximum per row, l_i: running sum of exp(scores - max)
  // Shape: (_2{}, Rep_M) - each thread handles 2 rows per MMA
  auto max_i = make_tensor<float>(make_shape(_2{}, Int<size<1>(tSrS)>{}));
  fill(max_i, -1e20);
  auto l_i = make_tensor<float>(make_shape(_2{}, Int<size<1>(tSrS)>{}));
  fill(l_i, 0);

  // Step 7: Load and scale Q block: Q = Q / sqrt(head_dim)
  for (int i = 0; i < size(tQsQ); i++) {
    tQsQ(i) = static_cast<T>(scaler) * tQsQ(i);
  }
  __syncthreads();  // Ensure all threads finish loading and scaling Q

  // Step 8: Main loop - process each KV block
  // For each KV block: compute S = Q @ K^T, apply online softmax, accumulate O
  // += softmax(S) @ V
  copy(tiled_s2r_copy_Q, tXsQ, tXrQ);

  for (int blkKVIdx = 0; blkKVIdx < size<2>(gK); ++blkKVIdx) {
    // TODO: Implement causal masking (for autoregressive models)

    // Load current K block from global memory to shared memory
    __syncthreads();  // Ensure previous operations complete
    copy(tiled_copy, tKgK(_, _, _, blkKVIdx), tKsK);
    __syncthreads();  // Ensure all threads finish loading K

    // Copy K from shared memory to registers
    copy(tiled_s2r_copy_K, tXsK, tXrK);
#ifdef FLASH_ATTN_MMA_DEBUG
    if (thread0()) {  // clang-format off
      print("blkKVIdx: "); print(blkKVIdx); print("\n");
      print("tXrQ: "); print_tensor(tXrQ); print("\n");
      print("tSrQ: "); print_tensor(tSrQ); print("\n");
      print("tXrK: "); print_tensor(tXrK); print("\n");
      print("tSrK: "); print_tensor(tSrK); print("\n");
    }  // clang-format on
#endif
    // Clear attention scores for current KV block
    clear(tSrS);

    // Compute attention scores: S = Q @ K^T
    // Result stored in tSrS registers, shape: (MMA, Rep_M, Rep_KV)
    gemm(tiled_mma, tSrQ, tSrK, tSrS);
#ifdef FLASH_ATTN_MMA_DEBUG
    if (thread0()) {  // clang-format off
      print("tSrS: "); print_tensor(tSrS); print("\n");
    }  // clang-format on
#endif
    // Compute local maximum: max(v0, v1) and max(v2, v3) per thread
    auto max_ij = make_fragment_like(max_i);  // Max for current block
    fill(max_ij, -1e20);  // Initialize to very negative value
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
    // Reduce max across 4 threads per row using warp shuffle (XOR with 1, then
    // 2)
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
    // Combine max with previous blocks to get running maximum
    for (int max_row_idx = 0; max_row_idx < size<0>(max_ij); ++max_row_idx) {
      for (int max_col_idx = 0; max_col_idx < size<1>(max_ij); ++max_col_idx) {
        max_ij(max_row_idx, max_col_idx) = max(
            max_i(max_row_idx, max_col_idx), max_ij(max_row_idx, max_col_idx));
      }
    }
#ifdef FLASH_ATTN_MMA_DEBUG
    if (thread0()) {  // clang-format off
  print("max_ij: "); print_tensor(max_ij); print("\n");
}  // clang-format on
#endif
    // Online softmax compensation: rescale previous results using new max
    // Step 1: Compensate output O from previous KV blocks
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
    // Step 2: Compensate denominator (l_i) from previous iterations
    for (int max_row_idx = 0; max_row_idx < size<0>(max_ij); ++max_row_idx) {
      for (int max_col_idx = 0; max_col_idx < size<1>(max_ij); ++max_col_idx) {
        l_i(max_row_idx, max_col_idx) *= exp(max_i(max_row_idx, max_col_idx) -
                                             max_ij(max_row_idx, max_col_idx));
      }
    }

    // Step 3: Compute softmax: exp(S - max) and update denominator
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
      print("l_i: "); print_tensor(l_i); print("\n");
    }  // clang-format on
#endif
    // Update running maximum for next iteration
    for (int max_row_idx = 0; max_row_idx < size<0>(max_ij); ++max_row_idx) {
      for (int max_col_idx = 0; max_col_idx < size<1>(max_ij); ++max_col_idx) {
        max_i(max_row_idx, max_col_idx) = max_ij(max_row_idx, max_col_idx);
      }
    }

    // Convert softmax scores to output type (F32 -> half_t/bfloat16_t)
    auto tOrS = make_tensor<T>(tSrS.layout());
    for (int i = 0; i < size(tOrS); ++i) {
      tOrS(i) = static_cast<T>(tSrS(i));
    }

    // Compute O += softmax(S) @ V (accumulate contribution from current KV
    // block)
    static_assert(tiled_mma.get_layoutA_TV() == tiled_mma.get_layoutC_TV(),
                  "This is only valid for atom mnk == (16, 8, 8), otherwise we "
                  "will have different A and C layout and need to adjust the "
                  "layout accordingly");

    // Load V block from global memory to shared memory
    __syncthreads();
    copy(tiled_copy, tVgV(_, _, _, blkKVIdx), tVsV);
    __syncthreads();

    // Copy V from shared memory to registers (transposed view: MMA expects V^T
    // but we compute S @ V)
    copy(tiled_s2r_copy_V, tXsVt, tXrVt);
#ifdef FLASH_ATTN_MMA_DEBUG
    if (thread0()) {  // clang-format off
      print("tOrVt: "); print_tensor(tOrVt); print("\n");
    }  // clang-format on
#endif
    // Accumulate: O += softmax(S) @ V
    gemm(tiled_mma, tOrS, tOrVt, tOrO);
#ifdef FLASH_ATTN_MMA_DEBUG
    if (thread0()) {  // clang-format off
      print("tOrO: "); print_tensor(tOrO); print("\n");
    }  // clang-format on
#endif
  }
  // Reduce denominator across threads (warp shuffle: XOR with 1, then 2)
  for (int row_idx = 0; row_idx < size<0>(l_i); ++row_idx) {
    for (int col_idx = 0; col_idx < size<1>(l_i); ++col_idx) {
      l_i(row_idx, col_idx) +=
          __shfl_xor_sync(0xffffffff, l_i(row_idx, col_idx), 1);
      l_i(row_idx, col_idx) +=
          __shfl_xor_sync(0xffffffff, l_i(row_idx, col_idx), 2);
    }
  }

  // Epilogue: Final normalization O = O / sum(exp(S - max))
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
  // Copy final output from registers to global memory
  auto tiled_r2s_copy_O = make_tiled_copy_C(SmemCopyAtomO{}, tiled_mma);
  auto thr_r2s_copy_O = tiled_r2s_copy_O.get_slice(tx);
  auto tXrO = thr_r2s_copy_O.retile_S(tOrO);  // Retile to match copy layout
  auto tXgO =
      thr_r2s_copy_O.partition_D(gO);  // Partition global memory destination

  copy(tiled_r2s_copy_O, tXrO, tXgO);
}

// Sanity check: validates tensor dimensions match (self-attention only)
static bool sanity_check(torch::Tensor& Q, torch::Tensor& K, torch::Tensor& V,
                         torch::Tensor& O) {
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

// Launch kernel with specific configuration
template <int BlockQO, int BlockKV, int HeadDim, int NWarpsPerSM>
static void launch_kernel(torch::Tensor& Q, torch::Tensor& K, torch::Tensor& V,
                          torch::Tensor& O) {
  using config =
      FlashAttnConfig<half_t, BlockQO, BlockKV, HeadDim, NWarpsPerSM>;

  assert(sanity_check(Q, K, V, O));
  const int b = Q.size(0);  // Batch size
  const int h = Q.size(1);  // Number of heads
  const int n = Q.size(2);  // Sequence length
  const int d = Q.size(3);  // Head dimension

  // Compute scaling factor: 1/sqrt(head_dim) for attention
  float scaler = 1.0 / sqrt(d);

  // TODO: Use ceil division for non-divisible sequence lengths
  assert(n % BlockQO == 0);  // Sequence length must be divisible by block size

  // Grid configuration: (batch, head, sequence_blocks)
  dim3 grid(b, h, n / BlockQO);
  // Block configuration: total threads per block
  dim3 block(size(config::NumThreads));
  flash_attn_v2_kernel<config><<<grid, block>>>(
      reinterpret_cast<half_t*>(Q.data_ptr()),
      reinterpret_cast<half_t*>(K.data_ptr()),
      reinterpret_cast<half_t*>(V.data_ptr()),
      reinterpret_cast<half_t*>(O.data_ptr()), b, h, n, n, d, scaler);
  CUDACHECK(cudaGetLastError());
}

// Main entry point: selects kernel configuration based on head dimension
void flash_attn_v2_cute_v1(torch::Tensor& Q, torch::Tensor& K, torch::Tensor& V,
                           torch::Tensor& O) {
  // TODO: Add tensor dtype checks (macro not defined)
  //   CHECK_TORCH_TENSOR_DTYPE(Q, torch::kHalf)  // Q [B,H,N,D]
  //   CHECK_TORCH_TENSOR_DTYPE(K, torch::kHalf)  // K [B,H,N,D]
  //   CHECK_TORCH_TENSOR_DTYPE(V, torch::kHalf)  // V [B,H,N,D]
  //   CHECK_TORCH_TENSOR_DTYPE(O, torch::kHalf)  // O [B,H,N,D]
  const int d = Q.size(3);  // Head dimension

  // Select kernel configuration based on head dimension
  // Different configurations optimize for different head sizes
  // TODO: Optimize these configurations for different hardware (A100, H100,
  // etc.)
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

// Test main function for Flash Attention kernel
int main() {
  // Check if CUDA is available
  if (!torch::cuda::is_available()) {
    std::cerr << "CUDA is not available" << std::endl;
    return -1;
  }

  // Set test parameters
  int B = 1;  // Batch size
  int H = 1;  // Number of attention heads
  int N =
      128;  // Sequence length (must be multiple of BlockQO, checked by assert)
  int D = 32;  // Head dimension (must match one of the supported values)

  std::cout << "Running FlashAttention Cute Test with: "
            << "B=" << B << ", H=" << H << ", N=" << N << ", D=" << D
            << std::endl;

  auto options =
      torch::TensorOptions().dtype(torch::kHalf).device(torch::kCUDA);

  // Create input tensors
  auto Q = torch::randn({B, H, N, D}, options);
  auto K = torch::randn({B, H, N, D}, options);
  auto V = torch::randn({B, H, N, D}, options);
  auto O = torch::empty_like(Q);

  // Launch kernel
  try {
    flash_attn_v2_cute_v1(Q, K, V, O);

    // Synchronize device to ensure execution completes
    cudaDeviceSynchronize();
    std::cout << "Kernel executed successfully!" << std::endl;

    // Print a sample output value to verify execution
    std::cout << "Output[0][0][0][0]: " << O[0][0][0][0].item<float>()
              << std::endl;

  } catch (const std::exception& e) {
    std::cerr << "Error: " << e.what() << std::endl;
    return -1;
  }

  return 0;
}