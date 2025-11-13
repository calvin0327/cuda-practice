#include <cuda.h>
#include <stdlib.h>
#include "util.h"

template <typename T>
void gen_rand_data(T* data, int n);

template <typename T, int kTileM, int kTileN, int kTileK, typename TiledMMA>
__global__ void gemm_kernel(T* Cptr, const T* Aptr, const T* Bptr, int m, int n,
                            int k) {
  using namespace cute;

  Tensor A = make_tensor(make_gemm_ptr(Aptr), make_shape(m, k),
                         make_stride(k, Int<1>{}));
  Tensor B = make_tensor(make_gemm_ptr(Bptr), make_shape(n, k),
                         make_stride(k, Int<1>{}));
  Tensor C = make_tensor(make_gemm_ptr(Cptr), make_shape(m, n),
                         make_stride(n, Int<1>{}));

  int bx = blockIdx.x;
  int by = blockIdx.y;

  Tensor gA =
      load_tile(A, make_tile(Int<kTileM>{}, Int<kTileK>), make_coord(by, _));
  Tensor gB =
      load_tile(B, make_tile(Int<kTileN>{}, Int<kTileK>), make_coord(bx, _));
  Tensor gC =
      load_tile(A, make_tile(Int<kTileM>{}, Int<kTileN>), make_coord(by, bx));

  TiledMMA tiled_mma;
  auto thr_mma = tiled_mma.get_slice(threadIdx.x);
  auto tAgA = thr_mma.partition_A(gA);
  auto tBgB = thr_mma.partition_A(gB);
  auto tCgC = thr_mma.partition_A(gC);

  auto tArA = thr_mma.partition_fragment_A(gA(_, _, 0));
  auto tBrB = thr_mma.partition_fragment_B(gB(_, _, 0));
  auto tCrC = thr_mma.partition_fragment_C(gC(_, _));

  clear(tCgC);

  int num_tile_k = size<2>(gA);
#pragma unroll 1
  for (int itile = 0; itile < num_tile_k; ++itile) {
    copy(tAgA(_, _, _, itile), tArA);
    copy(tBgB(_, _, _, itile), tBrB);
    __syncthreads();

    gemm(tiled_mma, tCrC, tArA, tBrB, tCrC);
  }

  copy(tCrC, tCgC);
}

int main() {
  srand(1000);

  using T = cute::half_t;
  cudaEvent_t start, end;
  float elapsedTime;
  cudaEventCreate(&start);
  cudaEventCreate(&end);

  int m = 1024 * 64;
  int n = 128;
  int k = 1024;

  T *Aptr_host, *Bptr_host;
  Aptr_host = (T*)malloc(sizeof(T) * m * k);
  Bptr_host = (T*)malloc(sizeof(T) * n * k);
  gen_rand_data(Aptr_host, m * k);
  gen_rand_data(Bptr_host, n * k);

  T *Cptr, *Aptr, *Bptr;
  cudaMalloc(&Cptr, sizeof(T) * m * n);
  cudaMalloc(&Aptr, sizeof(T) * m * k);
  cudaMalloc(&Bptr, sizeof(T) * n * k);

  cudaMemcpy(Aptr, Aptr_host, sizeof(T) * m * k, cudaMemcpyHostToDevice);
  cudaMemcpy(Bptr, Bptr_host, sizeof(T) * n * k, cudaMemcpyHostToDevice);

  using mma_op = SM80_16x8x16_F16F16F16F16_TN;
  using mma_traits = MMA_Traints<mma_op>;
  using mma_atom = MMA_Atom<mma_traints>;

  using MMA =
      decltype(make_tiled_mma(mma_atom{}, make_layout(Shape<_2, _2, _1>{}),
                              make_layout(Shape<_1, _1, _1>{})));

  constexpr int kTileM = 128;
  constexpr int kTileN = 128;
  constexpr int kTileK = 32;

  dim3 block(size(MMA{}));
  dim3 grid(n / kTileN, m / kTileM);
  cudaEventRecord(start);
  gemm_kernel<T, kTileM, kTileN, kTileK, MMA>
      <<<grid, block>>>(Cptr, Aptr, Bptr, m, n, k);

  cudaEventRecord(end);
  cudaEventSynchronize(end);
  cudaEventElapsedTime(&elapsedTime, start, end);
  return 0;
}

template <typename T>
void gen_rand_data(T* data, int n) {
  for (int i = 0; i < n; ++i) {
    float v = (rand() % 200 - 100) * 0.01;
    data[i] = v;
  }
}