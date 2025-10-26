#include <algorithm>
#include <float.h>
#include <stdio.h>
#include <stdlib.h>

#include <torch/all.h>
#include <torch/library.h>
#include <cuda_runtime.h>
#include <cuda.h>

#include "utils.h"

#define FLOAT4(value) (reinterpret_cast<float4*>(&(value))[0])

__global__ void relu_f32_kernel(float* x, float* y, int N) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < N) {
    y[idx] = fmaxf(0.0f, x[idx]);
  }
}

__global__ void relu_f32x4_kernel(float* x, float* y, int N) {
  int idx = (blockIdx.x * blockDim.x + threadIdx.x) * 4;
  if (idx < N) {
    float4 reg_x = FLOAT4(x[idx]);
    float4 reg_y;
    reg_y.x = fmaxf(0.0f, reg_x.x);
    reg_y.y = fmaxf(0.0f, reg_x.y);
    reg_y.z = fmaxf(0.0f, reg_x.z);
    reg_y.w = fmaxf(0.0f, reg_x.w);
    FLOAT4(y[idx]) = reg_y;
  }
}

#define TORCH_BINDING_RELU(packed_type, th_type, element_type, n_element) \
  void relu_##packed_type(torch::Tensor& x, torch::Tensor& y) {           \
    TORCH_CHECK(x.options().dtype() == (th_type),                         \
                "tensor x type must be ##th_type");                       \
    TORCH_CHECK(y.options().dtype() == (th_type),                         \
                "tensor y type must be ##th_type");                       \
    const int64_t N = x.numel();                                          \
    if (N == 0) return;                                                   \
                                                                          \
    dim3 block, grid;                                                     \
    const int ndim = x.dim();                                             \
    if (ndim == 2) {                                                      \
      const int S = x.size(0);                                            \
      const int K = x.size(1);                                            \
      if (K / (n_element) <= 1024) {                                      \
        block = dim3((K + (n_element) - 1) / (n_element));                \
        grid = dim3(S);                                                   \
      } else {                                                            \
        block = dim3((256 + (n_element) - 1) / (n_element));              \
        grid = dim3((N + 256 - 1) / 256);                                 \
      }                                                                   \
    } else {                                                              \
      block = dim3((256 + (n_element) - 1) / (n_element));                \
      grid = dim3((N + 256 - 1) / 256);                                   \
    }                                                                     \
    relu_##packed_type##_kernel<<<grid, block>>>(                         \
        reinterpret_cast<element_type*>(x.data_ptr()),                    \
        reinterpret_cast<element_type*>(y.data_ptr()), N);                \
    CUDACHECK(cudaGetLastError());                                        \
    CUDACHECK(cudaDeviceSynchronize());                                   \
  }

TORCH_BINDING_RELU(f32, torch::kFloat32, float, 1)
TORCH_BINDING_RELU(f32x4, torch::kFloat32, float, 4)
