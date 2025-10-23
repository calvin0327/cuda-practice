#include <torch/all.h>
#include <torch/library.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include "utils.h"

__global__ void sgemm_naive_f32_kernel(float* a, float* b, float* c, int N,
                                       int M, int K) {
  // per threads calculation a small tile in c matrix.
  const int tx = blockDim.x * blockIdx.x + threadIdx.x;
  const int ty = blockDim.y * blockIdx.y + threadIdx.y;

  if (tx < N && ty < M) {
    float sum = 0.;
#pragma unroll
    for (int k = 0; k < K; k++) {
      sum += a[ty * K + k] * b[k * N + tx];
    }
    c[ty * N + ty] = sum;
  }
}

void sgemm_naive_f32(torch::Tensor& a, torch::Tensor& b, torch::Tensor& c) {
  TORCH_CHECK(a.options().dtype() == torch::kFloat,
              "tensor a type must be float");
  TORCH_CHECK(b.options().dtype() == torch::kFloat,
              "tensor b type must be float");
  TORCH_CHECK(c.options().dtype() == torch::kFloat,
              "tensor c type must be float");

  const int M = a.size(0);
  const int K = a.size(1);
  const int N = b.size(1);

  TORCH_CHECK(a.size(0) == M && a.size(1) == K,
              "tensor a shape must be [%d, %d]", M, K);
  TORCH_CHECK(b.size(0) == K && b.size(1) == N,
              "tensor a shape must be [%d, %d]", K, N);
  TORCH_CHECK(c.size(0) == M && c.size(1) == N,
              "tensor a shap must be [%d, %d]", M, N);

  constexpr int BM = 32;
  constexpr int BN = 32;

  dim3 block(BN, BM);
  dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM);

  sgemm_naive_f32_kernel<<<grid, block>>>(
      reinterpret_cast<float*>(a.data_ptr()),
      reinterpret_cast<float*>(b.data_ptr()),
      reinterpret_cast<float*>(c.data_ptr()), N, M, K);
}
