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
template <class T, class ProblemShape, class CtaTiler>
__global__ void gemm_f16_sliced_128x8_t16x16_kernel(T* C, T const* A,
                                                    T const* B,
                                                    ProblemShape shape_MNK,
                                                    CtaTiler cta_tiler) {
  using namespace cute;

  CUTE_STATIC_ASSERT_V(rank(shape_MNK) == Int<3>{});
  CUTE_STATIC_ASSERT_V(rank(cta_tiler) == Int<3>{});

  // auto M = get<0>(shape_MNK);
  // auto N = get<1>(shape_MNK);
  // auto K = get<2>(shape_MNK);
  // auto m = get<0>(cta_tiler);
  // auto n = get<1>(cta_tiler);
  // auto k = get<2>(cta_tiler);

  // Tensor mA = make_tensor(make_gmem_ptr(A), select<0, 2>(shape_MNK),
  //                         make_stride(Int<1>{}, M));

  Tensor mA = make_tensor(make_gmem_ptr(A),
                          make_layout(select<0, 2>(shape_MNK)));  // M-major
  Tensor mB = make_tensor(make_gmem_ptr(B),
                          make_layout(select<1, 2>(shape_MNK)));  // N-major
  Tensor mC = make_tensor(make_gmem_ptr(C),
                          make_layout(select<0, 1>(shape_MNK)));  // M-major

  // if (thread0()) {
  // CUTE_PRINT("mA layout", mA.layout());
  // CUTE_PRINT("mB layout", mB.layout());
  // CUTE_PRINT("mC layout", mC.layout());
  // }

  // tiled A, B, C, cta_tiler is (128, 128, 8)
  auto cta_coord = make_coord(blockIdx.x, blockIdx.y, _);  // (M, N, K)
  Tensor gA = local_tile(mA, cta_tiler, cta_coord, Step<_1, X, _1>());
  Tensor gB = local_tile(mB, cta_tiler, cta_coord, Step<X, _1, _1>());
  Tensor gC = local_tile(mC, cta_tiler, cta_coord, Step<_1, _1, X>());

  // if (thread0()) {
  // CUTE_PRINT("gA layout", gA.layout());
  // CUTE_PRINT("gB layout", gB.layout());
  // CUTE_PRINT("gC layout", gC.layout());
  // }

  // smemA_size is 128 * 8, smemB_size is 128 * 8
  constexpr auto smemA_size = size<0>(cta_tiler) * size<2>(cta_tiler);
  constexpr auto smemB_size = size<1>(cta_tiler) * size<2>(cta_tiler);

  __shared__ T smemA[smemA_size];
  __shared__ T smemB[smemB_size];
  Tensor sA = make_tensor(make_smem_ptr(smemA),
                          make_layout(select<0, 2>(cta_tiler)));  // m-major
  Tensor sB = make_tensor(make_smem_ptr(smemB),
                          make_layout(select<1, 2>(cta_tiler)));  // n-major

  // if (thread0()) {
  // CUTE_PRINT("sA layout", sA.layout());
  // CUTE_PRINT("sB layout", sB.layout());
  // }

  // load 4 elements peer threa -> (128 * 8) / (32 * 8)
  auto load_thr_tile = make_layout(make_shape(Int<32>{}, Int<8>{}));
  Tensor tAgA = local_partition(gA, load_thr_tile, threadIdx.x);
  Tensor tAsA = local_partition(sA, load_thr_tile, threadIdx.x);

  Tensor tBgB = local_partition(gB, load_thr_tile, threadIdx.x);
  Tensor tBsB = local_partition(sB, load_thr_tile, threadIdx.x);

  // if (thread0()) {
  // CUTE_PRINT("tAgA layout", tAgA.layout());
  // CUTE_PRINT("tAsA layout", tAsA.layout());
  // CUTE_PRINT("tBgB layout", tBgB.layout());
  // CUTE_PRINT("tBsB layout", tBsB.layout());
  // }

  CUTE_STATIC_ASSERT_V(size<0>(tAgA) == size<0>(tAsA));  // Thr_m
  CUTE_STATIC_ASSERT_V(size<1>(tAgA) == size<1>(tAsA));  // Thr_k
  CUTE_STATIC_ASSERT_V(size<0>(tBgB) == size<0>(tBsB));  // Thr_n
  CUTE_STATIC_ASSERT_V(size<1>(tBgB) == size<1>(tBsB));  // Thr_k

  // comput 64 element of C -> (128 * 128) / (16 * 16)
  auto compute_thr_tile = make_layout(make_shape(Int<16>{}, Int<16>{}));
  Tensor tCsA = local_partition(sA, compute_thr_tile, threadIdx.x,
                                Step<_1, X>());  //(Thr_m, 8)
  Tensor tCsB = local_partition(sB, compute_thr_tile, threadIdx.x,
                                Step<X, _1>());  // (Thr_n, 8)
  Tensor tCgC = local_partition(gC, compute_thr_tile, threadIdx.x,
                                Step<_1, _1>());  // (8, 8)

  auto tCrC = make_tensor_like(tCgC);

  // if (thread0()) {
  // CUTE_PRINT("tCsA layout", tCsA.layout());
  // CUTE_PRINT("tCsB layout", tCsB.layout());
  // CUTE_PRINT("tCgC layout", tCgC.layout());
  // CUTE_PRINT("tCrC layout", tCrC.layout());
  // }

  CUTE_STATIC_ASSERT_V(size<0>(tCrC) == size<0>(tCgC));  // THR_M
  CUTE_STATIC_ASSERT_V(size<0>(tCrC) == size<0>(tCsA));  // THR_M
  CUTE_STATIC_ASSERT_V(size<1>(tCrC) == size<1>(tCgC));  // THR_N
  CUTE_STATIC_ASSERT_V(size<1>(tCrC) == size<0>(tCsB));  // THR_N
  CUTE_STATIC_ASSERT_V(size<1>(tCsA) == size<1>(tCsB));  // BLK_K

  // Clear the accumulators
  clear(tCrC);

  auto K_TILE_MAX = size<2>(tAgA);
  for (int k_tile = 0; k_tile < K_TILE_MAX; ++k_tile) {
    copy(tAgA(_, _, k_tile), tAsA);
    copy(tBgB(_, _, k_tile), tBsB);

    cp_async_fence();    // Label the end of (potential)
                         // cp.async instructions
    cp_async_wait<0>();  // Sync on all (potential)
                         // cp.async instructions
    __syncthreads();     // Wait for all threads to write
                         // to smem

    // Compute gemm on tC thread-partitioned smem
    gemm(tCsA, tCsB, tCrC);
    __syncthreads();  // Wait for all threads to read
                      // from smem
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

  cudaEventRecord(start);

  dim3 block(16 * 16);
  dim3 grid(size(ceil_div(M, bM)), size(ceil_div(N, bN)));
  gemm_f16_sliced_128x8_t16x16_kernel<<<grid, block>>>(
      d_C.data().get(), d_A.data().get(), d_B.data().get(),  // data
      proble_shape, cta_tiler);

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
