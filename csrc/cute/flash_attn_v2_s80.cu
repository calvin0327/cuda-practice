#include <torch/all.h>
#include <torch/library.h>

#include <cutlass/fast_math.h>
#include <cute/layout.hpp>
#include <cute/tensor.hpp>

#include "../utils.h"

using namespace cute;

// Ref: https://github.com/xlite-dev/LeetCUDA
// Ref: https://github.com/izmttk/flash_attention_cute

// Key observation: In a single MMA operation, one row is handled by 4 threads
// (T0-T3). Even when multiple MMAs are tiled along the N dimension, each row is
// still handled by the same 4 threads. Therefore, reduction only needs to
// reduce across these 4 threads.
// ┌────────────────────────────────────────────────────────────────────┐
// │  Repeated along N dimension: 64×8 large Tile (4 MMAs horizontally) │
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
// └────────────────────────────────────────----------------────────────┘
// Configuration structure for Flash Attention kernel
// Template parameters:
//   T_: Data type (half_t or bfloat16_t)
//   BlockQO_: Block size for Q and O tensors along sequence dimension
//   BlockKV_: Block size for K and V tensors along sequence dimension
//   HeadDim_: Head dimension (typically 16, 32, 64, 128, or 256)
//   NWarpsPerSM_: Number of warps per streaming multiprocessor
template <typename T_, int BlockQO_, int BlockKV_, int HeadDim_,
          int NWarpsPerSM_, int NStage_>
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
  static constexpr int NStage = NStage_;

  // Block sizes for tiling
  static constexpr int BlockQO = BlockQO_;  // Block size for Q and O
  static constexpr int BlockKV = BlockKV_;  // Block size for K and V
  static constexpr int HeadDim = HeadDim_;  // Head dimension

  // Number of values each thread loads per instruction (128-bit aligned load)
  // Calculated based on element type: 128 bits / sizeof(T) elements
  static constexpr int GmemValsPerLoad = sizeof(uint128_t) / sizeof(T);
  // Number of threads needed per row to cover HeadDim elements
  static constexpr int GmemThreadsPerRow = HeadDim / GmemValsPerLoad;

  // Tiled copy: copy atom, thread layout, instructions per thread
  using TiledCopyQKVO = decltype(make_tiled_copy(
      Copy_Atom<SM80_CP_ASYNC_CACHEGLOBAL<uint128_t>, T>{},
      make_layout(
          Shape<Int<NumThreads / GmemThreadsPerRow>, Int<GmemThreadsPerRow>>{},
          GenRowMajor{}),  // thr_layout
      make_layout(Shape<_1, Int<GmemValsPerLoad>>{})));

  using SmemLayoutAtom = decltype(composition(
      Swizzle<3, 3, 3>{},
      make_layout(Shape<Int<8>, Int<HeadDim>>{}, GenRowMajor{})));

  using SmemLayoutQ = decltype(tile_to_shape(
      SmemLayoutAtom{}, make_shape(Int<BlockQO>{}, Int<HeadDim>{})));
  using SmemLayoutO = decltype(tile_to_shape(
      SmemLayoutAtom{}, make_shape(Int<BlockQO>{}, Int<HeadDim>{})));

  // Build mutil stage for K and V
  using SmemLayoutK = decltype(tile_to_shape(
      SmemLayoutAtom{},
      make_shape(Int<BlockKV>{}, Int<HeadDim>{}, Int<NStage>{})));
  using SmemLayoutV = decltype(tile_to_shape(
      SmemLayoutAtom{},
      make_shape(Int<BlockKV>{}, Int<HeadDim>{}, Int<NStage>{})));

  // Use GenColMajor to create transposed view of V
  using SmemLayoutAtomTranspose = decltype(composition(
      Swizzle<3, 3, 3>{},
      make_layout(Shape<Int<HeadDim>, Int<8>>{}, GenColMajor{})));

  // GenRowMajor refers to SmemLayoutAtomTranspose arrangement
  using SmemLayoutVt = decltype(tile_to_shape(
      SmemLayoutAtomTranspose{},
      make_shape(Int<HeadDim>{}, Int<BlockKV>{}, Int<NStage>{}),
      GenRowMajor{}));

  static_assert(Int<NumThreads / GmemThreadsPerRow>::value <= BlockQO,
                "NumThreads must be less than or equal to BlockQO");

  // LDSM copy atom for shared memory to register (MMA-compatible format)
  using SmemCopyAtom = Copy_Atom<SM75_U32x4_LDSM_N, T>;

  // Transposed LDSM for V: MMA expects transposed B, but we compute S @ V
  using SmemCopyAtomTransposed = Copy_Atom<SM75_U16x8_LDSM_T, T>;

  static_assert(std::is_same_v<T, half_t> || std::is_same_v<T, bfloat16_t>);

  // MMA atom: 16x8x8 for simplicity (same layout for Q@K^T and S@V)
  using MMA_Atom = std::conditional_t<std::is_same_v<T, half_t>,
                                      MMA_Atom<SM80_16x8x16_F32F16F16F32_TN>,
                                      MMA_Atom<SM80_16x8x16_F32BF16BF16F32_TN>>;

  // Tiled MMA: (16 * NWarpsPerSM) × 16 × 16 to cover BlockQO × BlockKV ×
  // HeadDim
  // TODO: optimzie?
  using TiledMMA = decltype(make_tiled_mma(
      MMA_Atom{}, make_layout(Shape<Int<NWarpsPerSM>, _1, _1>{}, GenRowMajor{}),
      Tile<Int<16 * NWarpsPerSM>, _16, _16>{}));

  using ToSmemCopyAtomO =
      Copy_Atom<AutoVectorizingCopyWithAssumedAlignment<sizeof(uint128_t) * 8>,
                T>;

  using ToGmemCopyAtomO = decltype(make_tiled_copy(
      Copy_Atom<AutoVectorizingCopyWithAssumedAlignment<sizeof(uint128_t) * 8>,
                T>{},
      make_layout(
          Shape<Int<NumThreads / GmemThreadsPerRow>, Int<GmemThreadsPerRow>>{},
          GenRowMajor{}),  // thr_layout
      make_layout(Shape<_1, Int<GmemValsPerLoad>>{})));

  static_assert(
      16 * NWarpsPerSM <= BlockQO && 16 <= BlockKV && 16 <= HeadDim,
      "BlockQO, BlockKV, and HeadDim must be greater than or equal to "
      "16 * NWarpsPerSM, 16, and 16 respectively");

  // Sanity checks: ensure thread counts match across all operations
  static_assert(size(TiledMMA{}) == NumThreads &&
                    size(TiledMMA{}) == size(TiledCopyQKVO{}),
                "Thread count mismatch between TiledMMA and TiledCopyQKVO");
};

// =================== Helper Functions for Flash Attention ===================

// Compute local maximum for current KV block from attention scores
// For each MMA, threads hold values v0, v1, v2, v3
// We compute max(v0, v1) and max(v2, v3) per thread first
template <typename TensorS, typename TensorMax>
__device__ void compute_local_max_and_scale(TensorS& tSrS, TensorMax& max_ij,
                                            const float scaler) {
  CUTE_UNROLL
  for (int val_idx = 0; val_idx < size<0>(tSrS); ++val_idx) {
    CUTE_UNROLL
    for (int row_idx = 0; row_idx < size<1>(tSrS); ++row_idx) {
      CUTE_UNROLL
      for (int col_idx = 0; col_idx < size<2>(tSrS); ++col_idx) {
        // Scale attention scores: S is F32 (no precision loss), reduces loop
        // iterations
        tSrS(val_idx, row_idx, col_idx) *= scaler;

        int max_row_idx = val_idx / 2;
        int max_col_idx = row_idx;
        max_ij(max_row_idx, max_col_idx) = max(max_ij(max_row_idx, max_col_idx),
                                               tSrS(val_idx, row_idx, col_idx));
      }
    }
  }
}

// Reduce maximum across 4 threads that share the same row
// Each row in MMA output is shared by 4 consecutive threads (T0-T3, T4-T7,
// etc.) Use warp shuffle with XOR pattern: first XOR with 1 (neighbors), then
// XOR with 2 This reduces 4 values to 1 in just 2 operations (log2(4) = 2)
template <typename TensorMax, typename TensorS>
__device__ void reduce_max_across_threads(TensorMax& max_ij,
                                          const TensorS& tSrS) {
  CUTE_UNROLL
  for (int max_row_idx = 0; max_row_idx < size<0>(max_ij); ++max_row_idx) {
    CUTE_UNROLL
    for (int max_col_idx = 0; max_col_idx < size<1>(tSrS); ++max_col_idx) {
      max_ij(max_row_idx, max_col_idx) =
          max(max_ij(max_row_idx, max_col_idx),
              __shfl_xor_sync(0xffffffff, max_ij(max_row_idx, max_col_idx), 1));
      max_ij(max_row_idx, max_col_idx) =
          max(max_ij(max_row_idx, max_col_idx),
              __shfl_xor_sync(0xffffffff, max_ij(max_row_idx, max_col_idx), 2));
    }
  }
}

// Combine max from current block with max from all previous blocks
// This gives us the running maximum across all KV blocks processed so far
template <typename TensorMaxI, typename TensorMaxIJ>
__device__ void combine_max_with_previous(const TensorMaxI& max_i,
                                          TensorMaxIJ& max_ij) {
  CUTE_UNROLL
  for (int max_row_idx = 0; max_row_idx < size<0>(max_ij); ++max_row_idx) {
    CUTE_UNROLL
    for (int max_col_idx = 0; max_col_idx < size<1>(max_ij); ++max_col_idx) {
      max_ij(max_row_idx, max_col_idx) = max(max_i(max_row_idx, max_col_idx),
                                             max_ij(max_row_idx, max_col_idx));
    }
  }
}

// Online softmax compensation: adjust previous results using new maximum
// When we find a new max, we need to rescale previous softmax values:
//   old_exp = exp(x - old_max)
//   new_exp = old_exp * exp(old_max - new_max) = exp(x - new_max)
template <typename TensorO, typename TensorMaxI, typename TensorMaxIJ>
__device__ void compensate_softmax(TensorO& tOrO, const TensorMaxI& max_i,
                                   const TensorMaxIJ& max_ij) {
  CUTE_UNROLL
  for (int val_idx = 0; val_idx < size<0>(tOrO); ++val_idx) {
    CUTE_UNROLL
    for (int row_idx = 0; row_idx < size<1>(tOrO); ++row_idx) {
      CUTE_UNROLL
      for (int col_idx = 0; col_idx < size<2>(tOrO); ++col_idx) {
        int max_row_idx = val_idx / 2;
        int max_col_idx = row_idx;
        tOrO(val_idx, row_idx, col_idx) *= exp(
            max_i(max_row_idx, max_col_idx) - max_ij(max_row_idx, max_col_idx));
      }
    }
  }
}

// Compensate denominator (l_i) from previous iterations
// The sum of exponentials also needs rescaling with the new max
template <typename TensorMaxLI, typename TensorMaxI, typename TensorMaxIJ>
__device__ void compensate_denominator(TensorMaxLI& l_i,
                                       const TensorMaxI& max_i,
                                       const TensorMaxIJ& max_ij) {
  CUTE_UNROLL
  for (int max_row_idx = 0; max_row_idx < size<0>(max_ij); ++max_row_idx) {
    CUTE_UNROLL
    for (int max_col_idx = 0; max_col_idx < size<1>(max_ij); ++max_col_idx) {
      l_i(max_row_idx, max_col_idx) *= exp(max_i(max_row_idx, max_col_idx) -
                                           max_ij(max_row_idx, max_col_idx));
    }
  }
}

// Compute softmax on current scores and update denominator
// Apply softmax: exp(S - max) and accumulate into denominator sum
// This is the standard softmax numerator computation
template <typename TensorS, typename TensorMaxLI, typename TensorMaxIJ>
__device__ void apply_softmax_and_update_denominator(
    TensorS& tSrS, TensorMaxLI& l_i, const TensorMaxIJ& max_ij) {
  CUTE_UNROLL
  for (int val_idx = 0; val_idx < size<0>(tSrS); ++val_idx) {
    CUTE_UNROLL
    for (int row_idx = 0; row_idx < size<1>(tSrS); ++row_idx) {
      CUTE_UNROLL
      for (int col_idx = 0; col_idx < size<2>(tSrS); ++col_idx) {
        int max_row_idx = val_idx / 2;
        int max_col_idx = row_idx;
        tSrS(val_idx, row_idx, col_idx) = exp(tSrS(val_idx, row_idx, col_idx) -
                                              max_ij(max_row_idx, max_col_idx));
        l_i(max_row_idx, max_col_idx) += tSrS(val_idx, row_idx, col_idx);
      }
    }
  }
}

// Update running maximum for next iteration
// Store max_ij into max_i so it's available for next KV block
template <typename TensorMaxI, typename TensorMaxIJ>
__device__ void update_running_max(TensorMaxI& max_i,
                                   const TensorMaxIJ& max_ij) {
  CUTE_UNROLL
  for (int max_row_idx = 0; max_row_idx < size<0>(max_ij); ++max_row_idx) {
    CUTE_UNROLL
    for (int max_col_idx = 0; max_col_idx < size<1>(max_ij); ++max_col_idx) {
      max_i(max_row_idx, max_col_idx) = max_ij(max_row_idx, max_col_idx);
    }
  }
}

// Reduce denominator across threads to get final sum for each row
// Each thread computed partial sums, now combine them using warp shuffle
// Same pattern as max reduction: XOR with 1, then XOR with 2
template <typename TensorMax>
__device__ void reduce_denominator(TensorMax& l_i) {
  CUTE_UNROLL
  for (int row_idx = 0; row_idx < size<0>(l_i); ++row_idx) {
    CUTE_UNROLL
    for (int col_idx = 0; col_idx < size<1>(l_i); ++col_idx) {
      l_i(row_idx, col_idx) +=
          __shfl_xor_sync(0xffffffff, l_i(row_idx, col_idx), 1);
      l_i(row_idx, col_idx) +=
          __shfl_xor_sync(0xffffffff, l_i(row_idx, col_idx), 2);
    }
  }
}

// Final normalization step
// Divide output by denominator to complete softmax: O = O / sum(exp(S - max))
// This is the final step of the softmax operation
template <typename TensorO, typename TensorMax>
__device__ void normalize_output(TensorO& tOrO, const TensorMax& l_i) {
  CUTE_UNROLL
  for (int val_idx = 0; val_idx < size<0>(tOrO); ++val_idx) {
    CUTE_UNROLL
    for (int row_idx = 0; row_idx < size<1>(tOrO); ++row_idx) {
      CUTE_UNROLL
      for (int col_idx = 0; col_idx < size<2>(tOrO); ++col_idx) {
        int l_row_idx = val_idx / 2;
        int l_col_idx = row_idx;
        tOrO(val_idx, row_idx, col_idx) /= l_i(l_row_idx, l_col_idx);
      }
    }
  }
}

// Flash Attention v2 kernel: O = softmax(Q @ K^T / sqrt(d)) @ V
// Uses online softmax to avoid storing full attention matrix
template <typename FlashAttnConfig_>
__global__ void flash_attn_v2_kernel(typename FlashAttnConfig_::T* Q_ptr,
                                     typename FlashAttnConfig_::T* K_ptr,
                                     typename FlashAttnConfig_::T* V_ptr,
                                     typename FlashAttnConfig_::T* O_ptr, int B,
                                     int H, int N_QO_CTX, int N_KV_CTX, int D,
                                     float scaler, bool is_causal) {
  // Extract data type from config
  using T = typename FlashAttnConfig_::T;

  // Block size for Q and O are the same
  // Block size for K and V are the same
  constexpr int BlockQO = FlashAttnConfig_::BlockQO;
  constexpr int BlockKV = FlashAttnConfig_::BlockKV;
  constexpr int HeadDim = FlashAttnConfig_::HeadDim;
  constexpr int NStage = FlashAttnConfig_::NStage;

  // the tiledCopy to copy global memory to shared memory
  using TiledCopy = typename FlashAttnConfig_::TiledCopyQKVO;

  using SmemLayoutQ = typename FlashAttnConfig_::SmemLayoutQ;
  using SmemLayoutO = typename FlashAttnConfig_::SmemLayoutO;
  using SmemLayoutK = typename FlashAttnConfig_::SmemLayoutK;
  using SmemLayoutV = typename FlashAttnConfig_::SmemLayoutV;
  using SmemLayoutVt = typename FlashAttnConfig_::SmemLayoutVt;

  // copy shared memory to register
  using SmemCopyAtom = typename FlashAttnConfig_::SmemCopyAtom;
  using SmemCopyAtom_T = typename FlashAttnConfig_::SmemCopyAtomTransposed;

  using TiledMMA = typename FlashAttnConfig_::TiledMMA;

  // Copy atom for output: copy O to shared memory first for better performance
  using ToSmemCopyAtomO = typename FlashAttnConfig_::ToSmemCopyAtomO;

  using ToTiledCopy = typename FlashAttnConfig_::ToGmemCopyAtomO;

  // static check
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

  // Step 2: Define shared memory layout for Q, K, V blocks
  extern __shared__ unsigned char smem[];
  T* sQ_ptr = reinterpret_cast<T*>(smem);
  T* sK_ptr = sQ_ptr + cosize(SmemLayoutQ{});
  T* sV_ptr = sK_ptr + cosize(SmemLayoutK{});
  auto sQ = make_tensor(make_smem_ptr(sQ_ptr), SmemLayoutQ{});
  auto sK = make_tensor(make_smem_ptr(sK_ptr), SmemLayoutK{});
  auto sV = make_tensor(make_smem_ptr(sV_ptr), SmemLayoutV{});

  // Reuse Q's memory after computation
  auto sO = make_tensor(sQ.data(), SmemLayoutO{});

  // Transposed view of V: view-only (no copy), MMA expects transposed B but we
  // compute S @ V
  auto sVt = make_tensor(make_smem_ptr(sV_ptr), SmemLayoutVt{});

  // Step 3: Define thread partitions for global->shared copy
  TiledCopy gmem_tiled_copy;
  auto gmem_thr_copy = gmem_tiled_copy.get_slice(tx);
  auto tQgQ = gmem_thr_copy.partition_S(gQ);
  auto tQsQ = gmem_thr_copy.partition_D(sQ);

  // (Copy, BlockKVCopy, HeadDimCopy, NumBlockKV)
  // (Copy, BlockKVCopy, HeadDimCopy, NumBlockKV, Stage)
  auto tKgK = gmem_thr_copy.partition_S(gK);
  auto tKsK = gmem_thr_copy.partition_D(sK);

  // (Copy, BlockKVCopy, HeadDimCopy, NumBlockKV)
  // (Copy, BlockKVCopy, HeadDimCopy, NumBlockKV, Stage)
  auto tVgV = gmem_thr_copy.partition_S(gV);
  auto tVsV = gmem_thr_copy.partition_D(sV);

  CUTE_STATIC_ASSERT_V(size<0>(tQgQ) == size<0>(tQsQ));
  CUTE_STATIC_ASSERT_V(size<1>(tQgQ) == size<1>(tQsQ));
  CUTE_STATIC_ASSERT_V(size<2>(tQgQ) == size<2>(tQsQ));
  CUTE_STATIC_ASSERT_V(size<0>(tKgK) == size<0>(tKsK));
  CUTE_STATIC_ASSERT_V(size<1>(tKgK) == size<1>(tKsK));
  CUTE_STATIC_ASSERT_V(size<2>(tKgK) == size<2>(tKsK));
  // tKgK has 4 dims (includes RestKV), tKsK has 3 dims, so skip size<3> check
  CUTE_STATIC_ASSERT_V(size<0>(tVgV) == size<0>(tVsV));
  CUTE_STATIC_ASSERT_V(size<1>(tVgV) == size<1>(tVsV));
  CUTE_STATIC_ASSERT_V(size<2>(tVgV) == size<2>(tVsV));
  // tVgV has 4 dims (includes RestKV), tVsV has 3 dims, so skip size<3> check

  // Step 4: Define register fragments for MMA operations
  // tSr*: register fragment from shared memory, tOr*: output fragment
  TiledMMA tiled_mma;
  auto thr_mma = tiled_mma.get_slice(tx);

  // Register fragments for Q and K (used in Q @ K^T computation)
  // Q fragment: (MMA, Rep_M, Rep_K)
  // K fragment: (MMA, Rep_KV, Rep_K, Stage)
  auto tSrQ = thr_mma.partition_fragment_A(sQ);
  auto tSrK = thr_mma.partition_fragment_B(sK);

  // Register fragment for attention scores S = Q @ K^T
  // S: (MMA, Rep_M, Rep_KV)
  auto tSrS =
      partition_fragment_C(tiled_mma, Shape<Int<BlockQO>, Int<BlockKV>>{});

  auto cS = make_identity_tensor(make_shape(Int<BlockQO>{}, Int<BlockKV>{}));
  auto tScS = thr_mma.partition_fragment_C(cS);

  // Register fragments for output computation O = S @ V
  // V is transposed because MMA expects transposed B operand, but we want S @ V
  // V^T fragment: (MMA, Rep_HeadDim, Rep_KV, Stage)
  auto tOrVt = thr_mma.partition_fragment_B(sVt);

  // O fragment: (MMA, Rep_M, Rep_HeadDim)
  auto tOrO =
      partition_fragment_C(tiled_mma, Shape<Int<BlockQO>, Int<HeadDim>>{});

  // Step 5: Define shared memory to register copy operations
  // Layout automatically adjusted to match MMA requirements
  auto tiled_s2r_copy_Q = make_tiled_copy_A(SmemCopyAtom{}, tiled_mma);
  auto thr_s2r_copy_Q = tiled_s2r_copy_Q.get_slice(tx);
  auto tXsQ = thr_s2r_copy_Q.partition_S(sQ);
  auto tXrQ = thr_s2r_copy_Q.retile_D(tSrQ);  // (CPY, MMA_QO, MMA_HEAD)

  auto tiled_s2r_copy_K = make_tiled_copy_B(SmemCopyAtom{}, tiled_mma);
  auto thr_s2r_copy_K = tiled_s2r_copy_K.get_slice(tx);
  auto tXsK = thr_s2r_copy_K.partition_S(sK);
  auto tXrK = thr_s2r_copy_K.retile_D(tSrK);  // (CPY, MMA_KV, MMA_HEAD, Stage)

  auto tiled_s2r_copy_V = make_tiled_copy_B(SmemCopyAtom_T{}, tiled_mma);
  auto thr_s2r_copy_V = tiled_s2r_copy_V.get_slice(tx);
  auto tXsVt = thr_s2r_copy_V.partition_S(sVt);
  auto tXrVt =
      thr_s2r_copy_V.retile_D(tOrVt);  // (CPY, MMA_Headdim, MMA_QO, Stage)

  // Step 7: Load and scale Q block: Q = Q / sqrt(head_dim)
  copy(gmem_tiled_copy, tQgQ, tQsQ);
  copy(gmem_tiled_copy, tKgK(_, _, _, 0), tKsK(_, _, _, 0));
  copy(gmem_tiled_copy, tVgV(_, _, _, 0), tVsV(_, _, _, 0));
  cp_async_fence();

  // Initialize output to zero (will accumulate results from each KV block)
  clear(tOrO);

#ifdef FLASH_ATTN_MMA_DEBUG
  if (thread0()) {  // clang-format off
    print("NumThreads: "); print(FlashAttnConfig_::NumThreads); print("\n");
    print("tiled_mma: "); print(tiled_mma); print("\n");
    print("tiled_copy: "); print(tiled_copy); print("\n");
    print("GmemValsPerLoad: "); print(FlashAttnConfig_::GmemValsPerLoad); print("\n");
    print("GmemThreadsPerRow: "); print(FlashAttnConfig_::GmemThreadsPerRow); print("\n");

    CUTE_PRINT("gQ", gQ.layout());
    CUTE_PRINT("gK", gK.layout());
    CUTE_PRINT("gV", gV.layout());
    CUTE_PRINT("sQ", sQ.layout());
    CUTE_PRINT("sK", sK.layout());
    CUTE_PRINT("sV", sV.layout());

    CUTE_PRINT("tQgQ", tQgQ.layout());
    CUTE_PRINT("tQsQ", tQsQ.layout());
    CUTE_PRINT("tKsK", tKsK.layout());
    CUTE_PRINT("tKgK", tKgK.layout());
    CUTE_PRINT("tVsV", tVsV.layout());

    CUTE_PRINT("tSrQ", tSrQ.layout());
    CUTE_PRINT("tSrK", tSrK.layout());
    CUTE_PRINT("tSrS", tSrS.layout());
    CUTE_PRINT("tOrVt", tOrVt.layout());
    CUTE_PRINT("tOrO", tOrO.layout());

    print("tiled_s2r_copy_Q: "); print(tiled_s2r_copy_Q); print("\n");
    CUTE_PRINT("tXsQ", tXsQ.layout());
    CUTE_PRINT("tXrQ", tXrQ.layout());

    print("tiled_s2r_copy_K: "); print(tiled_s2r_copy_K); print("\n");
    CUTE_PRINT("tXsK", tXsK.layout());
    CUTE_PRINT("tXrK", tXrK.layout());
    print("tiled_s2r_copy_V: "); print(tiled_s2r_copy_V);
    CUTE_PRINT("tXsVt", tXsVt.layout());
    CUTE_PRINT("tXrVt", tXrVt.layout());
  }  // clang-format on
#endif

  // Step 6: Initialize online softmax state
  // max_i: running maximum per row, l_i: running sum of exp(scores - max)
  // Shape: (_2{}, Rep_M) - each thread handles 2 rows per MMA
  // Single MMA 16x8x16 output matrix :
  // ┌───────────────────────────────────────────────────────────────────┐
  // │              MMA 16x8x16 Output Matrix (16 rows × 8 cols)         │
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
  // │   Key Observation: Each row is shared by 4 consecutive threads    │
  // │                    (T0-T3, T4-T7, ...)                            │
  // │                                                                   │
  // └───────────────────────────────────────────────────────────────────┘
  auto max_i = make_tensor<float>(make_shape(_2{}, Int<size<1>(tSrS)>{}));
  fill(max_i, -1e20);
  auto l_i = make_tensor<float>(make_shape(_2{}, Int<size<1>(tSrS)>{}));
  fill(l_i, 0);

  int k_block_max = size<2>(gK);
  if (is_causal) {
    k_block_max = (bz * BlockQO + BlockQO + BlockKV - 1) / BlockKV;
    k_block_max = min(k_block_max, (int)size<2>(gK));
  }

  for (int blkKVIdx = 0; blkKVIdx < k_block_max; ++blkKVIdx) {
    int smem_read_idx = blkKVIdx % NStage;
    int smem_write_idx = (blkKVIdx + 1) % NStage;

    // async copy the next K and V to smem
    if (blkKVIdx < size<2>(gK) - 1) {
      copy(gmem_tiled_copy, tKgK(_, _, _, blkKVIdx + 1),
           tKsK(_, _, _, smem_write_idx));
      copy(gmem_tiled_copy, tVgV(_, _, _, blkKVIdx + 1),
           tVsV(_, _, _, smem_write_idx));
      cp_async_fence();
    }

    // Clear attention scores for current KV block
    clear(tSrS);

    // wait current Q and K and K block from global memory to shared memory
    cp_async_wait<1>();
    __syncthreads();  // Ensure previous operations complete

    // Compute attention scores: S = Q @ K^T
    // Copy Q and K from shared memory to registers
    copy(tiled_s2r_copy_Q, tXsQ(_, _, 0), tXrQ(_, _, 0));
    copy(tiled_s2r_copy_K, tXsK(_, _, 0, smem_read_idx), tXrK(_, _, 0, 0));
    CUTE_UNROLL
    for (int blkIdx = 0; blkIdx < size<2>(tSrQ); blkIdx++) {
      if (blkIdx < size<2>(tSrQ) - 1) {
        copy(tiled_s2r_copy_Q, tXsQ(_, _, blkIdx + 1), tXrQ(_, _, blkIdx + 1));
        copy(tiled_s2r_copy_K, tXsK(_, _, blkIdx + 1, smem_read_idx),
             tXrK(_, _, blkIdx + 1, 0));
      }
      gemm(tiled_mma, tSrQ(_, _, blkIdx), tSrK(_, _, blkIdx, 0), tSrS);
    }

    if (is_causal) {
      int row_offset = bz * BlockQO;
      int col_offset = blkKVIdx * BlockKV;
      if (col_offset + BlockKV > row_offset) {
        CUTE_UNROLL
        for (int i = 0; i < size(tSrS); ++i) {
          int global_row = row_offset + get<0>(tScS(i));
          int global_col = col_offset + get<1>(tScS(i));
          if (global_col > global_row) {
            tSrS(i) = -INFINITY;
          }
        }
      }
    }

    // Compute local maximum for current KV block
    // For each MMA, threads hold values v0, v1, v2, v3
    // We compute max(v0, v1) and max(v2, v3) per thread first
    auto max_ij = make_fragment_like(max_i);  // Max for current block
    fill(max_ij, -1e20);

    compute_local_max_and_scale(tSrS, max_ij, scaler);

    // Reduce maximum across 4 threads that share the same row
    reduce_max_across_threads(max_ij, tSrS);

    // Combine max from current block with max from all previous blocks
    combine_max_with_previous(max_i, max_ij);

    // Online softmax compensation: adjust previous results using new maximum
    // When we find a new max, we need to rescale previous softmax values:
    //   old_exp = exp(x - old_max)
    //   new_exp = old_exp * exp(old_max - new_max) = exp(x - new_max)
    //
    compensate_softmax(tOrO, max_i, max_ij);
    compensate_denominator(l_i, max_i, max_ij);

    apply_softmax_and_update_denominator(tSrS, l_i, max_ij);
    update_running_max(max_i, max_ij);

    // Convert softmax scores to output type for matrix multiplication
    // MMA may return F32, but we need to convert to T (half_t/bfloat16_t)
    // for S
    // @ V. Additionally, for 16x8x16 atoms, the register layout of the
    // Accumulator (C) differs from the Operand (A). We must re-partition S
    // to match the A-layout.
    auto tOrS = make_fragment_like(thr_mma.partition_fragment_A(
        make_tensor(make_gmem_ptr((T*)nullptr),
                    make_shape(Int<BlockQO>{}, Int<BlockKV>{}))));
    copy(tSrS, tOrS);

    // Accumulate: O += softmax(S) @ V
    // This adds the attention-weighted values from current KV block to output
    copy(tiled_s2r_copy_V, tXsVt(_, _, 0, smem_read_idx), tXrVt(_, _, 0, 0));
    CUTE_UNROLL
    for (int blkIdx = 0; blkIdx < size<2>(tOrS); ++blkIdx) {
      if (blkIdx < size<2>(tOrS) - 1) {
        copy(tiled_s2r_copy_V, tXsVt(_, _, blkIdx + 1, smem_read_idx),
             tXrVt(_, _, blkIdx + 1, 0));
      }
      gemm(tiled_mma, tOrS(_, _, blkIdx), tOrVt(_, _, blkIdx, 0), tOrO);
    }

#ifdef FLASH_ATTN_MMA_DEBUG
    if (thread0()) {  // clang-format off
      print("blkKVIdx: "); print(blkKVIdx); print("\n");
      CUTE_PRINTTENSOR("tXrQ", tXrQ);
      CUTE_PRINTTENSOR("tSrQ", tSrQ);
      CUTE_PRINTTENSOR("tXrK", tXrK);
      CUTE_PRINTTENSOR("tSrK", tSrK);
      CUTE_PRINTTENSOR("tSrS", tSrS);
      CUTE_PRINTTENSOR("max_ij", max_ij);
      CUTE_PRINTTENSOR("l_i", l_i); 
      CUTE_PRINTTENSOR("tOrVt", tOrVt);
      CUTE_PRINTTENSOR("tOrO", tOrO); 
    }  // clang-format on
#endif
  }

  // Reduce denominator across threads to get final sum for each row
  // Each thread computed partial sums, now combine them using warp shuffle
  // Same pattern as max reduction: XOR with 1, then XOR with 2
  reduce_denominator(l_i);

  // Epilogue: Final normalization step
  // Divide output by denominator to complete softmax: O = O / sum(exp(S - max))
  // This is the final step of the softmax operation
  normalize_output(tOrO, l_i);
#ifdef FLASH_ATTN_MMA_DEBUG
  if (thread0()) {  // clang-format off
    print("l_i", l_i);
    print("tOrO", tOrO);
  }  // clang-format on
#endif

  __syncthreads();
  // first copying to shared memory, then batch writing to global memory This
  // would improve memory coalescing and reduce global memory transactions
  auto tiled_r2s_copy_O = make_tiled_copy_C(ToSmemCopyAtomO{}, tiled_mma);
  auto thr_r2s_copy_O = tiled_r2s_copy_O.get_slice(tx);

  auto tXrO = thr_r2s_copy_O.retile_S(tOrO);  // Retile to match copy layout
  auto tXsO = thr_r2s_copy_O.partition_D(sO);

  // Cute automatically converts FP32 (tXrO) to FP16 (tXsO)
  copy(tiled_r2s_copy_O, tXrO, tXsO);
  __syncthreads();

  // copy smem sO to gmem gO
  ToTiledCopy tiled_copy_to_gmem;
  auto thr_copy_to_gmem = tiled_copy_to_gmem.get_slice(tx);
  auto tOsO = thr_copy_to_gmem.partition_S(sO);
  auto tOgO = thr_copy_to_gmem.partition_D(gO);

  copy(tiled_copy_to_gmem, tOsO, tOgO);
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
template <int BlockQO, int BlockKV, int HeadDim, int NWarpsPerSM, int Nstage>
static void launch_kernel(torch::Tensor& Q, torch::Tensor& K, torch::Tensor& V,
                          torch::Tensor& O, bool is_causal) {
  using config =
      FlashAttnConfig<half_t, BlockQO, BlockKV, HeadDim, NWarpsPerSM, Nstage>;

  assert(sanity_check(Q, K, V, O));
  const int b = Q.size(0);  // Batch size
  const int h = Q.size(1);  // Number of heads
  const int n = Q.size(2);  // Sequence length
  const int d = Q.size(3);  // Head dimension

  // Compute scaling factor: 1/sqrt(head_dim) for attention
  float scaler = 1.0 / sqrt(d);

  // TODO: Support arbitrary N values
  assert(n % BlockQO == 0);  // Sequence length must be divisible by block size
  // In launch_kernel or the caller
  if (n % BlockQO != 0) {
    throw std::runtime_error(
        "Sequence length must be multiple of block size (" +
        std::to_string(BlockQO) +
        ") for this kernel (head_dim=" + std::to_string(d) + ")");
  }

  // Calculate shared memory size required by the kernel
  // sQ, sK, sV are needed. sO reuses sQ memory, sVt is a view of sV.
  using T = typename config::T;

  // Calculate size in elements (cosize returns the size of the domain)
  int size_q = cosize(typename config::SmemLayoutQ{});
  int size_k = cosize(typename config::SmemLayoutK{});
  int size_v = cosize(typename config::SmemLayoutV{});

  // Total shared memory bytes needed
  int smem_size = (size_q + size_k + size_v) * sizeof(T);

  int device_id;
  CUDACHECK(cudaGetDevice(&device_id));
  int max_smem_per_block_optin = 0;
  CUDACHECK(cudaDeviceGetAttribute(&max_smem_per_block_optin,
                                   cudaDevAttrMaxSharedMemoryPerBlockOptin,
                                   device_id));

  if (smem_size > max_smem_per_block_optin) {
    throw std::runtime_error(
        "Requesting shared memory size (" + std::to_string(smem_size) +
        ") exceeds device limit (" + std::to_string(max_smem_per_block_optin) +
        "). Try reducing BlockQO/BlockKV.");
  }

  // Check if we need to increase the shared memory limit (default is usually
  // 48KB)
  if (smem_size >= 48 * 1024) {
    CUDACHECK(cudaFuncSetAttribute(flash_attn_v2_kernel<config>,
                                   cudaFuncAttributeMaxDynamicSharedMemorySize,
                                   smem_size));
  }

  // Grid configuration: (batch, head, sequence_blocks)
  dim3 grid(b, h, n / BlockQO);
  // Block configuration: total threads per block
  dim3 block(config::NumThreads);

  // Pass smem_size as the 3rd argument
  flash_attn_v2_kernel<config>
      <<<grid, block, smem_size>>>(reinterpret_cast<half_t*>(Q.data_ptr()),
                                   reinterpret_cast<half_t*>(K.data_ptr()),
                                   reinterpret_cast<half_t*>(V.data_ptr()),
                                   reinterpret_cast<half_t*>(O.data_ptr()), b,
                                   h, n, n, d, scaler, is_causal);

  CUDACHECK(cudaGetLastError());
}

// Main entry point: selects kernel configuration based on head dimension
void flash_attn_v2_cute(torch::Tensor& Q, torch::Tensor& K, torch::Tensor& V,
                        torch::Tensor& O, bool is_causal = false) {
  CHECK_TORCH_TENSOR_DTYPE(Q, torch::kHalf)  // Q [B,H,N,D]
  CHECK_TORCH_TENSOR_DTYPE(K, torch::kHalf)  // K [B,H,N,D]
  CHECK_TORCH_TENSOR_DTYPE(V, torch::kHalf)  // V [B,H,N,D]
  CHECK_TORCH_TENSOR_DTYPE(O, torch::kHalf)  // O [B,H,N,D]

  const int d = Q.size(3);  // Head dimension

  // Select kernel configuration based on head dimension
  switch (d) {
    case 16:
      launch_kernel<128, 128, 16, 8, 2>(Q, K, V, O, is_causal);
      break;
    case 32:
      launch_kernel<128, 128, 32, 8, 2>(Q, K, V, O, is_causal);
      break;
    case 64:
      launch_kernel<128, 128, 64, 8, 2>(Q, K, V, O, is_causal);
      break;
    case 128:
      launch_kernel<64, 64, 128, 4, 2>(Q, K, V, O, is_causal);
      break;
    case 256:
      launch_kernel<32, 32, 256, 2, 2>(Q, K, V, O, is_causal);
      break;
    default:
      throw std::runtime_error("Unsupported headdim");
  }
}