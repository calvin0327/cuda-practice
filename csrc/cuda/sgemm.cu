#include <torch/all.h>
#include <torch/library.h>
#include <float.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include "../utils.h"

#define FLOAT4(value) (reinterpret_cast<float4*>(&(value))[0])

__global__ void sgemm_naive_f32_kernel(float* a, float* b, float* c, int M,
                                       int K, int N) {
  // per threads calculation a small tile in c matrix.
  const int m = blockDim.y * blockIdx.y + threadIdx.y;
  const int n = blockDim.x * blockIdx.x + threadIdx.x;

  if (m < M && n < N) {
    float sum = 0.f;
#pragma unroll
    for (int k = 0; k < K; k++) {
      sum += a[m * K + k] * b[k * N + n];
    }
    c[m * N + n] = sum;
  }
}

template <const int BM = 32, const int BN = 32, const int BK = 32>
__global__ void sgemm_shared_f32_kernel(float* a, float* b, float* c, int M,
                                        int K, int N) {
  __shared__ float shared_a[BM][BK];
  __shared__ float shared_b[BK][BN];

  int shared_a_m = threadIdx.y;
  int shared_a_k = threadIdx.x;
  int shared_b_k = threadIdx.x;
  int shared_b_n = threadIdx.y;
  int global_c_m = blockIdx.y * BM + shared_a_m;
  int global_c_n = blockIdx.x * BN + shared_b_n;

  float sum = 0.f;
  for (int bk = 0; bk < (K + BK - 1) / BK; ++bk) {
    int global_a_k = bk * BK + shared_a_k;
    int global_a_idx = global_c_m * K + global_a_k;
    shared_a[shared_a_m][shared_a_k] =
        (global_c_m < M && global_c_n < N) ? a[global_a_idx] : 0.f;

    int global_b_k = bk * BK + shared_b_k;
    int global_b_idx = global_b_k * N + global_c_n;
    shared_b[shared_b_k][shared_b_n] =
        (global_c_m < M && global_c_n < N) ? b[global_b_idx] : 0.f;
    __syncthreads();

#pragma unroll
    for (int k = 0; k < BK; ++k) {
      sum += shared_a[shared_a_m][k] * shared_b[k][shared_b_n];
    }
    __syncthreads();
  }

  if (global_c_m < M && global_c_n < N) {
    int global_c_idx = global_c_m * N + global_c_n;
    c[global_c_idx] = sum;
  }
}

template <const int BM = 128, const int BN = 128, const int BK = 8,
          const int TM = 8, const int TN = 8>
__global__ void sgemm_t_8x8_shared_f32x4_kernel(float* a, float* b, float* c,
                                                int M, int K, int N) {
  __shared__ float s_a[BM][BK];
  __shared__ float s_b[BK][BN];

  int tid = threadIdx.y * blockDim.x + threadIdx.x;
  int shared_a_m = tid / 2;
  int shared_a_k = (tid % 2 == 0) ? 0 : 4;
  int shared_b_k = tid / 32;
  int shared_b_n = (tid % 32) * 4;
  int global_a_m = blockIdx.y * BM + shared_a_m;
  int global_b_n = blockIdx.x * BN + shared_b_n;

  float r_c[TM][TN] = {0.f};
  for (int bk = 0; bk < (K + BK - 1) / BK; ++bk) {
    int global_a_k = bk * BK + shared_a_k;
    int global_a_idx = global_a_m * K + global_a_k;
    if (global_a_m < M && global_a_k < K) {
      FLOAT4(s_a[shared_a_m][shared_a_k]) = FLOAT4(a[global_a_idx]);
    }

    int global_b_k = bk * BK + shared_b_k;
    int global_b_idx = global_b_k * N + global_b_n;
    if (global_b_n < N && global_b_k < K) {
      FLOAT4(s_b[shared_b_k][shared_b_n]) = FLOAT4(b[global_b_idx]);
    }
    __syncthreads();

#pragma unroll
    for (int m = 0; m < TM; ++m) {
      int shared_a_m = threadIdx.y * TM + m;
#pragma unroll
      for (int n = 0; n < TN; ++n) {
        int shared_b_n = threadIdx.x * TN + n;
#pragma unroll
        for (int k = 0; k < BK; ++k) {
          r_c[m][n] += s_a[shared_a_m][k] * s_b[k][shared_b_n];
        }
      }
    }
    __syncthreads();
  }

#pragma unroll
  for (int m = 0; m < TM; ++m) {
    int global_c_m = blockIdx.y * BM + threadIdx.y * TM + m;
    if (global_c_m < M) {
#pragma unroll
      for (int n = 0; n < TN; ++n) {
        int global_c_n = blockIdx.x * BN + threadIdx.x * TN + n;
        if (global_c_n < N) {
          int global_c_idx = global_c_m * N + global_c_n;
          c[global_c_idx] = r_c[m][n];
        }
      }
    }
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
      reinterpret_cast<float*>(c.data_ptr()), M, K, N);

  CUDACHECK(cudaGetLastError());
  CUDACHECK(cudaDeviceSynchronize());
}

void sgemm_shared_f32(torch::Tensor& a, torch::Tensor& b, torch::Tensor& c) {
  TORCH_CHECK(a.options().dtype() == torch::kFloat,
              "tensor a type must be float");
  TORCH_CHECK(b.options().dtype() == torch::kFloat,
              "tensor b type must be float");
  TORCH_CHECK(c.options().dtype() == torch::kFloat,
              "tensor c type must be float");

  const int M = a.size(0);
  const int K = a.size(1);
  const int N = b.size(1);

  TORCH_CHECK(b.size(0) == K && b.size(1) == N,
              "tensor a shape must be [%d, %d]", K, N);
  TORCH_CHECK(c.size(0) == M && c.size(1) == N,
              "tensor a shap must be [%d, %d]", M, N);

  constexpr int BM = 32;
  constexpr int BN = 32;
  constexpr int BK = 32;

  dim3 block(BN, BM);
  dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM);

  sgemm_shared_f32_kernel<BM, BN, BK>
      <<<grid, block>>>(reinterpret_cast<float*>(a.data_ptr()),
                        reinterpret_cast<float*>(b.data_ptr()),
                        reinterpret_cast<float*>(c.data_ptr()), M, K, N);

  CUDACHECK(cudaGetLastError());
  CUDACHECK(cudaDeviceSynchronize());
}

void sgemm_t_8x8_shared_f32x4(torch::Tensor& a, torch::Tensor& b,
                              torch::Tensor& c) {
  TORCH_CHECK(a.options().dtype() == torch::kFloat,
              "tensor a type must be float");
  TORCH_CHECK(b.options().dtype() == torch::kFloat,
              "tensor b type must be float");
  TORCH_CHECK(c.options().dtype() == torch::kFloat,
              "tensor c type must be float");

  const int M = a.size(0);
  const int K = a.size(1);
  const int N = b.size(1);

  TORCH_CHECK(b.size(0) == K && b.size(1) == N,
              "tensor a shape must be [%d, %d]", K, N);
  TORCH_CHECK(c.size(0) == M && c.size(1) == N,
              "tensor a shap must be [%d, %d]", M, N);

  constexpr int BM = 128;
  constexpr int BN = 128;
  constexpr int BK = 8;
  constexpr int TN = 8;
  constexpr int TM = 8;

  dim3 block(BN / TN, BM / TM);
  dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM);

  sgemm_t_8x8_shared_f32x4_kernel<BM, BN, BK, TM, TN>
      <<<grid, block>>>(reinterpret_cast<float*>(a.data_ptr()),
                        reinterpret_cast<float*>(b.data_ptr()),
                        reinterpret_cast<float*>(c.data_ptr()), M, K, N);

  CUDACHECK(cudaGetLastError());
  CUDACHECK(cudaDeviceSynchronize());
}
