#include <algorithm>

#include <torch/all.h>
#include <torch/library.h>
#include <float.h>
#include <cuda_fp16.h>
#include <cuda.h>
#include <cuda_runtime.h>

#include "utils.h"

#define ALPHA 1.0f

__device__ __forceinline__ float elu(float x) {
  return x > 0.f ? x : ALPHA * (expf(x) - 1.f);
}

__device__ __forceinline__ half elu_f16(half x) {
  return __hgt(x, __float2half(0.f))
             ? x
             : __hmul(__float2half(ALPHA), __hsub(hexp(x), __float2half(1.f)));
}

__global__ void elu_f32_kernel(float* x, float* y, int N) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < N) {
    y[idx] = elu(x[idx]);
  }
}

__global__ void elu_f16_kernel(half* x, half* y, int N) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < N) {
    y[idx] = elu_f16(x[idx]);
  }
}

#define TORCH_BINDING_ELU(packed_type, th_type, element_type, n_element) \
  void elu_##packed_type(torch::Tensor& x, torch::Tensor& y) {           \
    TORCH_CHECK(x.options().dtype() == (th_type),                        \
                "tensor x type must be ##th_type");                      \
    TORCH_CHECK(y.options().dtype() == (th_type),                        \
                "tensor y type must be ##th_type");                      \
    const int64_t N = x.numel();                                         \
    if (N == 0) return;                                                  \
                                                                         \
    dim3 block, grid;                                                    \
    const int ndim = x.dim();                                            \
    if (ndim == 2) {                                                     \
      const int S = x.size(0);                                           \
      const int K = x.size(1);                                           \
      if (K / (n_element) <= 1024) {                                     \
        block = dim3((K + (n_element) - 1) / (n_element));               \
        grid = dim3(S);                                                  \
      } else {                                                           \
        block = dim3((256 + (n_element) - 1) / (n_element));             \
        grid = dim3((N + 256 - 1) / 256);                                \
      }                                                                  \
    } else {                                                             \
      block = dim3((256 + (n_element) - 1) / (n_element));               \
      grid = dim3((N + 256 - 1) / 256);                                  \
    }                                                                    \
    elu_##packed_type##_kernel<<<grid, block>>>(                         \
        reinterpret_cast<element_type*>(x.data_ptr()),                   \
        reinterpret_cast<element_type*>(y.data_ptr()), N);               \
    CUDACHECK(cudaGetLastError());                                       \
    CUDACHECK(cudaDeviceSynchronize());                                  \
  }

TORCH_BINDING_ELU(f32, torch::kFloat32, float, 1)
TORCH_BINDING_ELU(f16, torch::kHalf, half, 1)
