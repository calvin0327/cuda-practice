#include <cuda.h>
#include <stdlib.h>
#include "util.h"

template <class ProblemShape, class CtaTiler, class TA, class AStride,
          class ASmemLayout, class AThreadLayout, class TB, class BStride,
          class BSmemLayout, class BThreadLayout, class TC, class CStride,
          class CSmemLayout, class CThreadLayout, class Alpha, class Beta>
__global__ static __launch_bounds__(decltype(size(
    CThreadLayout{}))::value) void gemm_device(ProblemShape shape_MNK,
                                               CtaTiler cta_tiler, TA const* A,
                                               AStride dA,
                                               ASmemLayout sA_layout,
                                               AThreadLayout tA, TB const* B,
                                               BStride dB,
                                               BSmemLayout sB_layout,
                                               BThreadLayout tB, TC* C,
                                               CStride dC, CSmemLayout,
                                               CThreadLayout tC, Alpha alpha,
                                               Beta beta) {
  // Define shapes (dynamic)
  auto M = int(m);
  auto N = int(n);
  auto K = int(k);
  auto prob_shape = make_shape(M, N, K);  // (M, N, K)

  // Preconditions
  CUTE_STATIC_ASSERT_V(rank(shape_MNK) == Int<3>{});  // (M, N, K)

  CUTE_STATIC_ASSERT_V(
      congruent(select<0, 2>(shape_MNK), dA));  // dA strides for shape MK
  CUTE_STATIC_ASSERT_V(
      congruent(select<1, 2>(shape_MNK), dB));  // dB strides for shape NK
  CUTE_STATIC_ASSERT_V(
      congruent(select<0, 1>(shape_MNK), dC));  // dC strides for shape MN

  // Represent the full tensors
  Tensor mA =
      make_tensor(make_gmem_ptr(A), select<0, 2>(shape_MNK), dA);  // (M,K)
  Tensor mB =
      make_tensor(make_gmem_ptr(B), select<1, 2>(shape_MNK), dB);  // (N,K)
  Tensor mC =
      make_tensor(make_gmem_ptr(C), select<0, 1>(shape_MNK), dC);  // (M,N)

  // Define NT strides (mixed)
  auto dA = make_stride(Int<1>{}, ldA);  // (dM, dK)
  auto dB = make_stride(Int<1>{}, ldB);  // (dN, dK)
  auto dC = make_stride(Int<1>{}, ldC);  // (dM, dN)

  // Define TN strides (mixed)
  //   auto dA = make_stride(ldA, Int<1>{});  // (dM, dK)
  //   auto dB = make_stride(ldB, Int<1>{});  // (dN, dK)
  //   auto dC = make_stride(Int<1>{}, ldC);  // (dM, dN)

  // Define CTA tile sizes (static)
  auto bM = Int<128>{};
  auto bN = Int<128>{};
  auto bK = Int<8>{};
  auto cta_tiler = make_shape(bM, bN, bK);  // (BLK_M, BLK_N, BLK_K)

  // Get the appropriate blocks for this threadblock
  auto cta_coord = make_coord(blockIdx.x, blockIdx.y, _);  // (m,n,k)
  Tensor gA = local_tile(mA, cta_tiler, cta_coord,
                         Step<_1, X, _1>{});  // (BLK_M,BLK_K,k)
  Tensor gB = local_tile(mB, cta_tiler, cta_coord,
                         Step<X, _1, _1>{});  // (BLK_N,BLK_K,k)
  Tensor gC =
      local_tile(mC, cta_tiler, cta_coord, Step<_1, _1, X>{});  // (BLK_M,BLK_N)

  // Use select<0,2> to use only the M- and K-modes of the tiler and coord
  Tensor gA = local_tile(mA, select<0, 2>(cta_tiler), select<0, 2>(cta_coord));

  // Define the smem layouts (static)
  auto sA = make_layout(make_shape(bM, bK));  // (m,k) -> smem_idx; m-major
  auto sB = make_layout(make_shape(bN, bK));  // (n,k) -> smem_idx; n-major

  // Preconditions
  static_assert(is_static<ASmemLayout>::value);
  static_assert(is_static<BSmemLayout>::value);
  static_assert(is_static<CSmemLayout>::value);

  CUTE_STATIC_ASSERT_V(size<0>(ASmemLayout{}) == size<0>(cta_tiler));  // BLK_M
  CUTE_STATIC_ASSERT_V(size<0>(CSmemLayout{}) == size<0>(cta_tiler));  // BLK_M
  CUTE_STATIC_ASSERT_V(size<0>(BSmemLayout{}) == size<1>(cta_tiler));  // BLK_N
  CUTE_STATIC_ASSERT_V(size<1>(CSmemLayout{}) == size<1>(cta_tiler));  // BLK_N
  CUTE_STATIC_ASSERT_V(size<1>(ASmemLayout{}) == size<2>(cta_tiler));  // BLK_K
  CUTE_STATIC_ASSERT_V(size<1>(BSmemLayout{}) == size<2>(cta_tiler));  // BLK_K

  // Shared memory buffers
  __shared__ TA smemA[cosize_v<ASmemLayout>];
  __shared__ TB smemB[cosize_v<BSmemLayout>];
  Tensor sA = make_tensor(make_smem_ptr(smemA), sA_layout);  // (BLK_M,BLK_K)
  Tensor sB = make_tensor(make_smem_ptr(smemB), sB_layout);  // (BLK_N,BLK_K)

    // Define thread layouts (static)
  auto tC = make_layout(make_shape(Int<16>{}, Int<16>{}));   // (m,n) -> thr_idx; m-major
    static_assert(is_static<CThreadLayout>::value);

  CUTE_STATIC_ASSERT_V(size(tC) == size(tA));                          // NumThreads

  CUTE_STATIC_ASSERT_V(size<0>(cta_tiler) % size<0>(tC) == Int<0>{});  // BLK_M / THR_M
  CUTE_STATIC_ASSERT_V(size<1>(cta_tiler) % size<1>(tC) == Int<0>{});  // BLK_N / THR_N

  // TUTORIAL: Example of a very simple compute mainloop
  //   copy(.) operates on the global and shared memory via the tA|tB
  //   partitioning gemm(.) operates on the shared and register memory via the
  //   tC partitioning

  auto K_TILE_MAX = size<2>(tAgA);

  for (int k_tile = 0; k_tile < K_TILE_MAX; ++k_tile) {
    // Copy gmem to smem with tA|tB thread-partitioned tensors
    copy(tAgA(_, _, k_tile), tAsA);  // A   (THR_M,THR_K) -> (THR_M,THR_K)
    copy(tBgB(_, _, k_tile), tBsB);  // B   (THR_N,THR_K) -> (THR_N,THR_K)

    cp_async_fence();    // Label the end of (potential) cp.async instructions
    cp_async_wait<0>();  // Sync on all (potential) cp.async instructions
    __syncthreads();     // Wait for all threads to write to smem

    // Compute gemm on tC thread-partitioned smem
    gemm(tCsA, tCsB, tCrC);  // (THR_M,THR_N) += (THR_M,BLK_K) * (THR_N,BLK_K)
    __syncthreads();         // Wait for all threads to read from smem
  }
}