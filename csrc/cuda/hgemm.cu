#include <cuda_runtime>
#include <float.h>
#include <torch/all.h>

__global__ void hgemm_navie_f16_kernel(half* a, half* b, half* c, int M, int K,
                                       int N) {
  int m = blockIdx.y * blockDim.y + threadIdx.y;
  int n = blockIdx.x * blockDim.x + threadIdx.x;

  if (m < M && n < N) {
    half sum = 0.0;
#pragma unroll
    for (int k = 0; k < K; ++k) {
      sum += a[m * K + k] * b[k * N + n];
    }
    c[m * K + n] = sum;
  }
}
