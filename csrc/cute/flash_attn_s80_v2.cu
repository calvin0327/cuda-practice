#include <torch/all.h>
#include <torch/library.h>

#include <cutlass/fast_math.h>
#include <cute/layout.hpp>
#include <cute/tensor.hpp>

#include "../utils.h"

using namespace cute;

// Flash Attention v2 Implementation using CUTLASS Cute
// This implementation uses online softmax algorithm to compute attention scores
// without storing the full attention matrix, reducing memory usage from O(N^2)
// to O(N) for sequence length N.
//
// Algorithm overview:
//   1. Load Q block into shared memory and scale by 1/sqrt(head_dim)
//   2. For each KV block:
//      a. Compute S = Q @ K^T (attention scores)
//      b. Apply online softmax: update running max and sum
//      c. Compensate previous output and denominator using new max
//      d. Compute O += softmax(S) @ V
//   3. Normalize final output by dividing by denominator
//
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
  static constexpr int GmemThreadsPerRow = 64 / GmemValsPerLoad;

  // Copy atom for transferring data from global memory to shared memory
  using GmemCopyAtom = Copy_Atom<SM80_CP_ASYNC_CACHEGLOBAL<uint128_t>, T>;

  // Tiled copy configuration: 16x64
  //   First parameter: Copy atom - how much data per instruction
  //   Second parameter: Thread layout - how threads are arranged (each thread
  //      executes once)
  //   Third parameter: Number of instructions per thread
  using TiledCopyQKVO = decltype(make_tiled_copy(
      GmemCopyAtom{},
      make_layout(
          Shape<Int<NumThreads / GmemThreadsPerRow>, Int<GmemThreadsPerRow>>{},
          GenRowMajor{}),  // thr_layout
      make_layout(Shape<_1, Int<GmemValsPerLoad>>{})));

  // 8x64
  using SmemLayoutAtom = decltype(composition(
      Swizzle<3, 3, 3>{},
      make_layout(Shape<Int<8>, Int<64>>{}, GenRowMajor{})));

  using SmemLayoutQO = decltype(tile_to_shape(
      SmemLayoutAtom{}, make_shape(Int<BlockQO>{}, Int<HeadDim>{})));

  using SmemLayoutKV = decltype(tile_to_shape(
      SmemLayoutAtom{}, make_shape(Int<BlockKV>{}, Int<HeadDim>{})));

  // 64x8
  using SmemLayoutAtomTranspose = decltype(composition(
      Swizzle<3, 3, 3>{},
      make_layout(Shape<Int<64>, Int<8>>{}, GenColMajor{})));

  // TODO: GenRowMajor or GenColMajor?
  using SmemLayoutVt = decltype(tile_to_shape(
      SmemLayoutAtomTranspose{}, make_shape(Int<HeadDim>{}, Int<BlockKV>{}),
      GenRowMajor{}));

  static_assert(Int<NumThreads / GmemThreadsPerRow>::value <= BlockQO,
                "NumThreads must be less than or equal to BlockQO");

  // LDSM (Load Data from Shared Memory) copy atom optimized for MMA operations
  // LDSM is used because it provides efficient loading from shared memory
  // directly into registers in a format compatible with MMA instructions
  using SmemCopyAtom = Copy_Atom<SM75_U32x4_LDSM_N, T>;

  // Transposed LDSM copy atom for loading V matrix
  // V needs to be transposed because MMA instruction expects transposed B
  // operand for computing S @ V^T, but we want S @ V (no transpose needed)
  using SmemCopyAtomTransposed = Copy_Atom<SM75_U16x8_LDSM_T, T>;

  // Copy atom for writing computed results back to global memory
  // TODO: Would using tiled copy be faster?
  using SmemCopyAtomO =
      Copy_Atom<AutoVectorizingCopyWithAssumedAlignment<sizeof(uint128_t) * 8>,
                T>;

  static_assert(std::is_same_v<T, half_t> || std::is_same_v<T, bfloat16_t>);

  // MMA (Matrix Multiply-Accumulate) atom configuration
  // Using 16x8x8 instead of 16x8x16 for simplicity:
  //   - Both Q@K^T and S@V use the same MMA layout (16x8x8)
  //   - This avoids needing to adjust tSrS layout to fit tOrS
  //   - Trade-off: slightly less efficient than 16x8x16, but simpler code
  using MMA_Atom = std::conditional_t<std::is_same_v<T, half_t>,
                                      MMA_Atom<SM80_16x8x8_F32F16F16F32_TN>,
                                      MMA_Atom<SM80_16x8x8_F32BF16BF16F32_TN>>;

  // Tiled MMA: combines multiple MMA atoms to cover the full block
  // For SM75_U32x4_LDSM_N, we need at least 4 * 8x8 = 16x16 matrix
  // Layout: (NWarpsPerSM warps) × (1 warp) × (1 warp) along M/N/K dimensions
  // Tile size: (16 * NWarpsPerSM) × 16 × 16 to cover BlockQO × BlockKV ×
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

// =================== Helper Functions for Flash Attention ===================

// Compute local maximum for current KV block from attention scores
// For each MMA, threads hold values v0, v1, v2, v3
// We compute max(v0, v1) and max(v2, v3) per thread first
template <typename TensorS, typename TensorMax>
__device__ void compute_local_max(const TensorS& tSrS, TensorMax& max_ij) {
  CUTE_UNROLL
  for (int val_idx = 0; val_idx < size<0>(tSrS); ++val_idx) {
    CUTE_UNROLL
    for (int row_idx = 0; row_idx < size<1>(tSrS); ++row_idx) {
      CUTE_UNROLL
      for (int col_idx = 0; col_idx < size<2>(tSrS); ++col_idx) {
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

// Flash Attention v2 kernel implementation
// This kernel computes: O = softmax(Q @ K^T / sqrt(d)) @ V
// using online softmax algorithm to avoid storing full attention matrix
//
// Parameters:
//   Q_ptr, K_ptr, V_ptr, O_ptr: Pointers to Q, K, V, O tensors in global memory
//   B: Batch size
//   H: Number of attention heads
//   N_QO_CTX: Sequence length for Q and O
//   N_KV_CTX: Sequence length for K and V (can differ for cross-attention)
//   D: Head dimension
//   scaler: Scaling factor (typically 1/sqrt(head_dim))
template <typename FlashAttnConfig_>
__global__ void flash_attn_v2_kernel(typename FlashAttnConfig_::T* Q_ptr,
                                     typename FlashAttnConfig_::T* K_ptr,
                                     typename FlashAttnConfig_::T* V_ptr,
                                     typename FlashAttnConfig_::T* O_ptr, int B,
                                     int H, int N_QO_CTX, int N_KV_CTX, int D,
                                     float scaler) {
  // Extract data type from config
  using T = typename FlashAttnConfig_::T;

  // Block size for Q and O are the same (M dimension)
  // Block size for K and V are the same (KV dimension)
  constexpr int BlockQO = FlashAttnConfig_::BlockQO;
  constexpr int BlockKV = FlashAttnConfig_::BlockKV;
  constexpr int HeadDim = FlashAttnConfig_::HeadDim;

  // the tiledCopy to copy global memory to shared memory
  using TiledCopy = typename FlashAttnConfig_::TiledCopyQKVO;

  using SmemLayoutQO = typename FlashAttnConfig_::SmemLayoutQO;
  using SmemLayoutKV = typename FlashAttnConfig_::SmemLayoutKV;
  using SmemLayoutVt = typename FlashAttnConfig_::SmemLayoutVt;

  // copy shared memory to register
  using SmemCopyAtom = typename FlashAttnConfig_::SmemCopyAtom;
  using SmemCopyAtom_T = typename FlashAttnConfig_::SmemCopyAtomTransposed;

  using TiledMMA = typename FlashAttnConfig_::TiledMMA;

  // Copy atom for output: copy O to shared memory first for better performance
  using SmemCopyAtomO = typename FlashAttnConfig_::SmemCopyAtomO;

  // TODO: static check
  assert(HeadDim == D);

  // Block indices: x for batch (B), y for head (H), z for sequence block (N)
  const int bx = blockIdx.x;
  const int by = blockIdx.y;
  const int bz = blockIdx.z;

  // Thread index within the block
  const int tx = threadIdx.x;

  // Step 1: Define Global Memory Tensors
  // Create Cute tensor views from raw pointers with
  // row-major layout Shape: [Batch, Head, Sequence, HeadDim]
  // TODO: Self-define shape layouts for better memory access patterns
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

  // Extract local tiles from global tensors for this block:
  //   - gQ/gO: Extract one block of size [BlockQO, HeadDim] for current
  //   batch/head/seq_block
  //   - gK/gV: Extract all blocks along sequence dimension, resulting in
  //   [BlockKV, HeadDim, Num_Blocks] The _ in local_tile preserves the block
  //   index dimension for K/V
  //
  // Shape of gQ and gO: (BlockQO, HeadDim) - single 2D tile
  // Shape of gK and gV: (BlockKV, HeadDim, Num_Blocks) - multiple tiles along
  // sequence
  auto gQ =
      local_tile(Q, make_shape(_1{}, _1{}, Int<BlockQO>{}, Int<HeadDim>{}),
                 make_coord(bx, by, bz, 0))(0, 0, _, _);
  auto gO =
      local_tile(O, make_shape(_1{}, _1{}, Int<BlockQO>{}, Int<HeadDim>{}),
                 make_coord(bx, by, bz, 0))(0, 0, _, _);

  // Since we use _ in the SeqLen dimension, local_tile returns a Tensor
  // with an additional dimension (preserving the block index dimension).
  // The returned Tensor has the following logical shape:
  // [Batch(1), Head(1), BlockKV, HeadDim, Num_Blocks]
  auto gK =
      local_tile(K, make_shape(_1{}, _1{}, Int<BlockKV>{}, Int<HeadDim>{}),
                 make_coord(bx, by, _, 0))(0, 0, _, _, _);
  auto gV =
      local_tile(V, make_shape(_1{}, _1{}, Int<BlockKV>{}, Int<HeadDim>{}),
                 make_coord(bx, by, _, 0))(0, 0, _, _, _);

  // Step 2: Define Shared Memory Layout
  // Allocate shared memory buffers for Q, K, V blocks
  extern __shared__ unsigned char alignas(T) smem[];
  T* sQ_ptr = reinterpret_cast<T*>(smem);
  T* sK_ptr = sQ_ptr + cosize(SmemLayoutQO{});
  T* sV_ptr = sK_ptr + cosize(SmemLayoutKV{});

  // Create tensor views over shared memory with row-major layout
  auto sQ = make_tensor(make_smem_ptr(sQ_ptr), SmemLayoutQO{});
  auto sK = make_tensor(make_smem_ptr(sK_ptr), SmemLayoutKV{});
  auto sV = make_tensor(make_smem_ptr(sV_ptr), SmemLayoutKV{});

  // Create transposed view of V in shared memory
  // This is a view-only transformation (no data copy) - Cute's core feature
  // Needed because MMA instruction expects transposed B operand, but we compute
  // S @ V By transposing V in the view, MMA will transpose it back during
  // computation
  auto sVt = make_tensor(make_smem_ptr(sV_ptr), SmemLayoutVt{});

  // Step 3: Define Thread Partitions for Global->Shared
  // Each thread is responsible for copying a portion
  // of data from global to shared memory TiledCopy defines how threads
  // cooperate to copy the entire block
  TiledCopy gmem_tiled_copy;
  // Get this thread's portion of the copy operation
  auto gmem_thr_copy = gmem_tiled_copy.get_slice(tx);

  // Thread partition shape explanation:
  //   Copy: Number of elements copied per instruction
  //   BlockMCopy: Number of copy operations along M dimension
  //   HeadDimCopy: Number of copy operations along K dimension

  // Example: If Tile is 128x64 with 128 threads, total elements = 8192.
  // Each thread handles 8192 / 128 = 64 elements.
  // If using CP_ASYNC (8 elements per instruction), each thread executes 8
  // instructions. Then tQgQ and tQsQ shapes might look like ((8), 4, 2),
  // meaning:
  //   (8): 8 elements per instruction
  //   4: 4 repetitions along M dimension
  //   2: 2 repetitions along K dimension
  //   (Note: Which dimension repeats depends on TiledCopy thread layout)

  auto tQgQ =
      gmem_thr_copy.partition_S(gQ);  // (Copy, BlockQOCopy, HeadDimCopy)
  auto tQsQ =
      gmem_thr_copy.partition_D(sQ);  // (Copy, BlockQOCopy, HeadDimCopy)

  // (Copy, BlockKVCopy, HeadDimCopy, RestKV)
  auto tKgK = gmem_thr_copy.partition_S(gK);
  auto tKsK =
      gmem_thr_copy.partition_D(sK);  // (Copy, BlockKVCopy, HeadDimCopy)

  // (Copy, BlockKVCopy, HeadDimCopy, RestKV)
  auto tVgV = gmem_thr_copy.partition_S(gV);
  auto tVsV = gmem_thr_copy.partition_D(sV);

  // Step 4: Define Register-Resident Fragments
  // These are small tensor fragments stored in thread
  // registers for MMA operations Each thread holds a portion of Q, K, V, S
  // (scores), and O (output) in registers
  //
  // Fragment naming convention:
  //   tSr*: Register fragment from shared memory (S = shared, r = register)
  //   tOr*: Register fragment for output computation
  //   Shape: (MMA_atom_values, Repetitions_M, Repetitions_K)
  //
  // tSrQ Shape: ((Atom_Val_A), Rep_M, Rep_K)
  //   Atom_Val_A: A matrix elements per thread in one MMA (e.g., 4 for FP16
  //   16x8x8) Rep_M: BlockQO / Atom_M (e.g., 128/16 = 8 repetitions along M)
  //   Rep_K: HeadDim / Atom_K (e.g., 64/8 = 8 repetitions along K)
  TiledMMA tiled_mma;
  auto thr_mma = tiled_mma.get_slice(tx);

  // Register fragments for Q and K (used in Q @ K^T computation)
  // Q fragment: (MMA, Rep_M, Rep_K)
  // K fragment: (MMA, Rep_KV, Rep_K)
  auto tSrQ = thr_mma.partition_fragment_A(sQ);
  auto tSrK = thr_mma.partition_fragment_B(sK);

  // Register fragment for attention scores S = Q @ K^T
  auto tSrS = partition_fragment_C(
      tiled_mma,
      Shape<Int<BlockQO>, Int<BlockKV>>{});  // S: (MMA, Rep_M, Rep_KV)

  // Register fragments for output computation O = S @ V
  // V is transposed because MMA expects transposed B operand, but we want S @ V
  // V^T fragment: (MMA, Rep_HeadDim, Rep_KV)
  auto tOrVt = thr_mma.partition_fragment_B(sVt);

  // O fragment: (MMA, Rep_M, Rep_HeadDim)
  auto tOrO =
      partition_fragment_C(tiled_mma, Shape<Int<BlockQO>, Int<HeadDim>>{});

  // Initialize output to zero (will accumulate results from each KV block)
  clear(tOrO);

  // Step 7: Load and Scale Q Block
  // Load Q block from global memory to shared memory
  // performance
  copy(gmem_tiled_copy, tQgQ, tQsQ);
  copy(gmem_tiled_copy, tKgK(_, _, _, 0), tKsK);
  cp_async_fence();

  // Step 8: Main Loop - Process Each KV Block
  // Flash Attention processes K and V in blocks to reduce
  // memory usage For each KV block:
  //   1. Compute attention scores S = Q @ K^T
  //   2. Apply online softmax (update max and sum)
  //   3. Compensate previous output using new max
  //   4. Accumulate O += softmax(S) @ V

  // Step 5: Define Shared Memory to Register Copy Operations
  // Create tiled copy operations to transfer
  // data from shared memory to registers Critical: The layout in shared memory
  // doesn't match MMA register layout make_tiled_copy_A/B automatically adjusts
  // the layout during copy to match MMA requirements This is why we need
  // separate copy operations for Q, K, V
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
  // Step 6: Initialize Online Softmax State
  // Online softmax algorithm maintains running statistics:
  //   max_i: Running maximum for each row (used for numerical stability)
  //   l_i: Running sum of exp(scores - max) for each row (denominator)
  //
  // For SM80 MMA, each thread owns 2 rows of C matrix:
  // Shape: (_2{}, Rep_M) where:
  //   _2{}: Each thread handles 2 rows per MMA (v0, v1 and v2,v3)
  //   Rep_M: Number of MMA repetitions along M dimension
  //       [v0, v1]
  //       ......
  //       [v2, v3]
  // Single MMA 16x8x16 output matrix:
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
  // max_i shape explanation: (_2{}, Int<size<1>(tSrS){})
  //   First dimension (2): Each thread participates in 2 rows per MMA
  //   instruction.
  //   If there are repetitions along N dimension, they will be reduced.
  //   Second dimension (size<1>(tSrS)): Multiple repeated MMAs along M
  //   dimension. So max_i[0] stores max values for the 2 rows this thread
  //   handles in each MMA tile, and max_i[1] represents multiple MMAs along the
  //   M dimension.
  auto max_i = make_tensor<float>(make_shape(_2{}, Int<size<1>(tSrS)>{}));
  fill(max_i, -1e20);
  auto l_i = make_tensor<float>(make_shape(_2{}, Int<size<1>(tSrS)>{}));
  fill(l_i, 0);

  // Apply scaling factor: Q = Q / sqrt(head_dim)
  // This scaling is part of the attention formula: softmax(Q @ K^T / sqrt(d))
  // We scale Q here to avoid scaling during each Q@K^T computation
  cp_async_wait<0>();
  for (int i = 0; i < size(tQsQ); i++) {
    tQsQ(i) = static_cast<T>(scaler) * tQsQ(i);
  }
  __syncthreads();  // Ensure all threads finish loading and scaling Q

  for (int blkKVIdx = 0; blkKVIdx < size<2>(gK); ++blkKVIdx) {
    // TODO: Implement causal masking (for autoregressive models)

    // Clear attention scores for current KV block
    clear(tSrS);

    // wait current K block from global memory to shared memory
    cp_async_wait<0>();
    __syncthreads();  // Ensure previous operations complete

    // async copy current V to smem
    copy(gmem_tiled_copy, tVgV(_, _, _, blkKVIdx), tVsV);
    cp_async_fence();

    // Compute attention scores: S = Q @ K^T
    // Copy Q and K from shared memory to registers
    copy(tiled_s2r_copy_Q, tXsQ(_, _, 0), tXrQ(_, _, 0));
    copy(tiled_s2r_copy_K, tXsK(_, _, 0), tXrK(_, _, 0));
    CUTE_UNROLL
    for (int blkIdx = 0; blkIdx < size<2>(tSrQ); blkIdx++) {
      if (blkIdx < size<2>(tSrQ) - 1) {
        copy(tiled_s2r_copy_Q, tXsQ(_, _, blkIdx + 1), tXrQ(_, _, blkIdx + 1));
        copy(tiled_s2r_copy_K, tXsK(_, _, blkIdx + 1), tXrK(_, _, blkIdx + 1));
      }
      gemm(tiled_mma, tSrQ(_, _, blkIdx), tSrK(_, _, blkIdx), tSrS);
    }

    // wait block V to shared memory
    cp_async_wait<0>();
    __syncthreads();  // Ensure all threads finish loading K

    // async copy the next K to smem
    if (blkKVIdx < size(tQsQ)) {
      copy(gmem_tiled_copy, tKgK(_, _, _, blkKVIdx + 1), tKsK);
      cp_async_fence();
    }

    // Compute local maximum for current KV block
    // For each MMA, threads hold values v0, v1, v2, v3
    // We compute max(v0, v1) and max(v2, v3) per thread first
    auto max_ij = make_fragment_like(max_i);  // Max for current block
    fill(max_ij, -1e20);
    compute_local_max(tSrS, max_ij);

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
    // MMA may return F32, but we need to convert to T (half_t/bfloat16_t) for S
    // @ V
    auto tOrS = make_tensor<T>(tSrS.layout());
    for (int i = 0; i < size(tOrS); ++i) {
      tOrS(i) = static_cast<T>(tSrS(i));
    }

    // Compute attention-weighted values: O += softmax(S) @ V
    // This accumulates the contribution from current KV block to output
    // Assertion: This implementation assumes A and C have same layout
    // This is only true for 16x8x8 MMA atoms, not for 16x8x16
    static_assert(tiled_mma.get_layoutA_TV() == tiled_mma.get_layoutC_TV(),
                  "This is only valid for atom mnk == (16, 8, 8), otherwise we "
                  "will have different A and C layout and need to adjust the "
                  "layout accordingly");

    // Accumulate: O += softmax(S) @ V
    // This adds the attention-weighted values from current KV block to output
    copy(tiled_s2r_copy_V, tXsVt(_, _, 0), tXrVt(_, _, 0));
    CUTE_UNROLL
    for (int blkIdx = 0; blkIdx < size<2>(tOrS); ++blkIdx) {
      if (blkIdx < size<2>(tOrS) - 1) {
        copy(tiled_s2r_copy_V, tXsVt(_, _, blkIdx + 1),
             tXrVt(_, _, blkIdx + 1));
      }
      gemm(tiled_mma, tOrS(_, _, blkIdx), tOrVt(_, _, blkIdx), tOrO);
    }

#ifdef FLASH_ATTN_MMA_DEBUG
    if (thread0()) {  // clang-format off
      print("blkKVIdx: "); print(blkKVIdx); print("\n");
      print("tXrQ: "); print_tensor(tXrQ); print("\n");
      print("tSrQ: "); print_tensor(tSrQ); print("\n");
      print("tXrK: "); print_tensor(tXrK); print("\n");
      print("tSrK: "); print_tensor(tSrK); print("\n");
      print("tSrS: "); print_tensor(tSrS); print("\n");
      print("max_ij: "); print_tensor(max_ij); print("\n");
      print("l_i: "); print_tensor(l_i); print("\n");
      print("tOrVt: "); print_tensor(tOrVt); print("\n");
      print("tOrO: "); print_tensor(tOrO); print("\n");
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
    print("l_i: "); print_tensor(l_i); print("\n");
    print("tOrO: "); print_tensor(tOrO); print("\n");
  }  // clang-format on
#endif
  // Copy final output from registers to global memory
  // TODO: Optimize by first copying to shared memory, then batch writing to
  // global memory This would improve memory coalescing and reduce global memory
  // transactions
  auto tiled_r2s_copy_O = make_tiled_copy_C(SmemCopyAtomO{}, tiled_mma);
  auto thr_r2s_copy_O = tiled_r2s_copy_O.get_slice(tx);
  auto tXrO = thr_r2s_copy_O.retile_S(tOrO);  // Retile to match copy layout
  auto tXgO =
      thr_r2s_copy_O.partition_D(gO);  // Partition global memory destination

  copy(tiled_r2s_copy_O, tXrO, tXgO);
}

// Sanity check function: validates tensor dimensions match
// This kernel only implements limited functionality (self-attention with fixed
// block sizes) For cross-attention (N_QO != N_KV), additional modifications are
// needed
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
// Template parameters define the block sizes and thread organization
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

  // TODO: 支持任意 N 的值
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

// Main entry point for Flash Attention v2
// Selects appropriate kernel configuration based on head dimension
void flash_attn_v2_cute_v2(torch::Tensor& Q, torch::Tensor& K, torch::Tensor& V,
                           torch::Tensor& O) {
  CHECK_TORCH_TENSOR_DTYPE(Q, torch::kHalf)  // Q [B,H,N,D]
  CHECK_TORCH_TENSOR_DTYPE(K, torch::kHalf)  // K [B,H,N,D]
  CHECK_TORCH_TENSOR_DTYPE(V, torch::kHalf)  // V [B,H,N,D]
  CHECK_TORCH_TENSOR_DTYPE(O, torch::kHalf)  // O [B,H,N,D]

  const int d = Q.size(3);  // Head dimension

  // Select kernel configuration based on head dimension
  // Different configurations optimize for different head sizes
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
