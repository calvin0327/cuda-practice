#include <stdlib.h>
#include <cstdio>
#include <cuda.h>
#include <cuda_runtime.h>

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <cute/tensor.hpp>
#include <cute/util/print.hpp>

#include "../utils.h"

// Setup params for an NT HEMM
template <class T, class ProblemShape, class CtaTiler,  //
          class TiledCopyA, class TiledCopyB, class TiledMma>
__global__ void gemm_f16_stream_128x8_t16x16_kernel(T* C, T const* A,
                                                    T const* B,  // data
                                                    ProblemShape shape_MNK,
                                                    CtaTiler cta_tiler,
                                                    TiledCopyA tiled_copy_a,
                                                    TiledCopyB tiled_copy_b,
                                                    TiledMma tiled_mma) {
  using namespace cute;

  CUTE_STATIC_ASSERT_V(rank(shape_MNK) == Int<3>{});
  CUTE_STATIC_ASSERT_V(rank(cta_tiler) == Int<3>{});

  Tensor mA = make_tensor(make_gmem_ptr(A),
                          make_layout(select<0, 2>(shape_MNK)));  // M-major
  Tensor mB = make_tensor(make_gmem_ptr(B),
                          make_layout(select<1, 2>(shape_MNK)));  // N-major
  Tensor mC = make_tensor(make_gmem_ptr(C),
                          make_layout(select<0, 1>(shape_MNK)));  // M-major

  // tiled A, B, C, cta_tiler is (128, 128, 8)
  auto cta_coord = make_coord(blockIdx.x, blockIdx.y, _);  // (M, N, K)
  Tensor gA = local_tile(mA, cta_tiler, cta_coord, Step<_1, X, _1>());
  Tensor gB = local_tile(mB, cta_tiler, cta_coord, Step<X, _1, _1>());
  Tensor gC = local_tile(mC, cta_tiler, cta_coord, Step<_1, _1, X>());

  // smemA_size is 128 * 8, smemB_size is 128 * 8
  constexpr auto smemA_size = size<0>(cta_tiler) * size<2>(cta_tiler);
  constexpr auto smemB_size = size<1>(cta_tiler) * size<2>(cta_tiler);

  __shared__ T smemA[smemA_size];
  __shared__ T smemB[smemB_size];
  Tensor sA = make_tensor(make_smem_ptr(smemA),
                          make_layout(select<0, 2>(cta_tiler)));  // m-major
  Tensor sB = make_tensor(make_smem_ptr(smemB),
                          make_layout(select<1, 2>(cta_tiler)));  // n-major

  ThrCopy thr_copy_a = tiled_copy_a.get_slice(threadIdx.x);
  Tensor tAgA = thr_copy_a.partition_S(gA);
  Tensor tAsA = thr_copy_a.partition_D(sA);
  // Allocate registers same shape/layout as partitioned data
  Tensor tArA = make_fragment_like(tAsA);

  ThrCopy thr_copy_b = tiled_copy_b.get_slice(threadIdx.x);
  Tensor tBgB = thr_copy_b.partition_S(gB);
  Tensor tBsB = thr_copy_b.partition_D(sB);
  // Allocate registers same shape/layout as partitioned data
  Tensor tBrB = make_fragment_like(tBsB);

  CUTE_STATIC_ASSERT_V(size<1>(tAgA) == size<1>(tAsA));  // CPY_M
  CUTE_STATIC_ASSERT_V(size<1>(tAgA) == size<1>(tArA));  // CPY_M
  CUTE_STATIC_ASSERT_V(size<2>(tAgA) == size<2>(tAsA));  // CPY_K
  CUTE_STATIC_ASSERT_V(size<2>(tAgA) == size<2>(tArA));  // CPY_K
  CUTE_STATIC_ASSERT_V(size<1>(tBgB) == size<1>(tBsB));  // CPY_N
  CUTE_STATIC_ASSERT_V(size<1>(tBgB) == size<1>(tBrB));  // CPY_N
  CUTE_STATIC_ASSERT_V(size<2>(tBgB) == size<2>(tBsB));  // CPY_K
  CUTE_STATIC_ASSERT_V(size<2>(tBgB) == size<2>(tBrB));  // CPY_K

  copy(tiled_copy_a, tAgA(_, _, _, 0), tArA);
  copy(tiled_copy_b, tBgB(_, _, _, 0), tBrB);

  ThrMMA thr_mma = tiled_mma.get_slice(threadIdx.x);
  Tensor tCsA = thr_mma.partition_A(sA);
  Tensor tCsB = thr_mma.partition_B(sB);
  Tensor tCgC = thr_mma.partition_C(gC);

  Tensor tCrC = thr_mma.make_fragment_C(tCgC);

  CUTE_STATIC_ASSERT_V(shape(tCrC) == shape(tCgC));      // (MMA,MMA_M,MMA_N)
  CUTE_STATIC_ASSERT_V(size<1>(tCgC) == size<1>(tCsA));  // MMA_M
  CUTE_STATIC_ASSERT_V(size<2>(tCgC) == size<1>(tCsB));  // MMA_N
  CUTE_STATIC_ASSERT_V(size<2>(tCsA) == size<2>(tCsB));  // MMA_K

  // Clear the accumulators
  clear(tCrC);

  auto K_TILE_MAX = size<3>(tAgA);
  for (int k_tile = 0; k_tile < K_TILE_MAX; ++k_tile) {
    __syncthreads();
    copy(tArA, tAsA);
    copy(tBrB, tBsB);
    __syncthreads();

    // Copy gmem to rmem for k_tile+1 with tA|tB thread-partitioned tensors
    int k_tile_next = (k_tile + 1 < K_TILE_MAX) ? k_tile + 1 : k_tile;
    copy(tiled_copy_a, tAgA(_, _, _, k_tile_next), tArA);
    copy(tiled_copy_b, tBgB(_, _, _, k_tile_next), tBrB);

    __syncthreads();  // Wait for all threads to write

    // Compute gemm on tC thread-partitioned smem
    gemm(tiled_mma, tCsA, tCsB, tCrC);
  }

  // register to global memory
  copy(tCrC, tCgC);
}

int main() {
  using namespace cute;

  using T = cute::half_t;

  cudaEvent_t start, end;
  float elapsedTime;
  cudaEventCreate(&start);
  cudaEventCreate(&end);

  const int m = 1024 * 64;
  const int n = 128;
  const int k = 1024;

  thrust::host_vector<T> h_A(m * k);
  thrust::host_vector<T> h_B(n * k);
  thrust::host_vector<T> h_C(m * n);

  for (int i = 0; i < m * k; ++i) {
    h_A[i] = static_cast<T>(2 * (rand() / double(RAND_MAX)) - 1);
  }
  for (int i = 0; i < n * k; ++i) {
    h_B[i] = static_cast<T>(2 * (rand() / double(RAND_MAX)) - 1);
  }
  for (int i = 0; i < m * n; ++i) {
    h_C[i] = static_cast<T>(-1);
  }

  thrust::device_vector<T> d_A = h_A;
  thrust::device_vector<T> d_B = h_B;
  thrust::device_vector<T> d_C = h_C;

  // Define shapes (static)
  auto M = Int<m>{};
  auto N = Int<n>{};
  auto K = Int<k>{};
  auto proble_shape = make_shape(M, N, K);  // (M, N, K)

  // Define CTA tile sizes (static)
  auto bM = Int<128>{};
  auto bN = Int<128>{};
  auto bK = Int<8>{};
  auto cta_tiler = make_shape(bM, bN, bK);  // (BLK_M, BLK_N, BLK_K)

  TiledCopy copyA =
      make_tiled_copy(Copy_Atom<UniversalCopy<uint128_t>, T>{},  //
                      Layout<Shape<_32, _8>>{},                  //
                      Layout<Shape<_32, _8>>{});

  TiledCopy copyB =
      make_tiled_copy(Copy_Atom<UniversalCopy<uint128_t>, T>{},  //
                      Layout<Shape<_32, _8>>{},                  //
                      Layout<Shape<_32, _8>>{});

  TiledMMA mmaC = make_tiled_mma(UniversalFMA<T, T, T>{},  //
                                 Layout<Shape<_16, _16, _1>>{});

  cudaEventRecord(start);

  dim3 block(16 * 16);
  dim3 grid(size(ceil_div(M, bM)), size(ceil_div(N, bN)));
  gemm_f16_stream_128x8_t16x16_kernel<<<grid, block, 0, 0>>>(
      d_C.data().get(), d_A.data().get(), d_B.data().get(),  // data
      proble_shape, cta_tiler, copyA, copyB, mmaC);

  CUDACHECK(cudaGetLastError());
  CUDACHECK(cudaDeviceSynchronize());

  cudaEventRecord(end);
  cudaEventSynchronize(end);
  cudaEventElapsedTime(&elapsedTime, start, end);

  printf("Kernel execution time: %f ms\n", elapsedTime);

  CUDACHECK(cudaEventDestroy(start));
  CUDACHECK(cudaEventDestroy(end));

  return 0;
}
