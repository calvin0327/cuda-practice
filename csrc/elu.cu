// #include <algorithm>
// #include <cuda.h>
// #include <cuda_runtime.h>
// #include <cuda_fp16.h>
// #include <torch/extension.h>
// #include <torch/types.h>

// #define ALPHA 1.0f

// __device__ __forceinline__ float elu(float x) {
//   return x > 0.f ? x : ALPHA * (expf(x) - 1.f);
// }

// __device__ __forceinline__ half elu_f16(half x) {
//   return __hgt(x, __float2half(0.f))
//              ? x
//              : __hmul(__float2half(ALPHA), __hsub(hexp(x),
//              __float2half(1.f)));
// }

// __global__ void elu_f32_kernel(float* x, float* y, int N) {
//   int idx = blockIdx.x * blockDim.x + threadIdx.x;
//   if (idx < N) {
//     y[idx] = elu(x[idx]);
//   }
// }

// __global__ void elu_f16_kernel(half* x, half* y, int N) {
//   int idx = blockIdx.x * blockDim.x + threadIdx.x;
//   if (idx < N) {
//     y[idx] = elu_f16(x[idx]);
//   }
// }

// void elu_f32(torch::Tensor x, torch::Tensor y) {
//   const auto check_tensor = [](const torch::Tensor& tensor, const char* name)
//   {
//     if (tensor.options().dtype() != torch::kFloat32) {
//       throw std::runtime_error(std::string("Tensor ") + name +
//                                " must be Float32, got " +
//                                torch::toString(tensor.dtype()));
//     }
//     if (!tensor.is_contiguous()) {
//       throw std::runtime_error(std::string("Tensor ") + name +
//                                " must be contiguous");
//     }
//   };

//   check_tensor(x, "x");
//   check_tensor(y, "y");

//   const int64_t N = x.numel();
//   if (N == 0) return;

//   dim3 block, grid;
//   const int ndim = x.dim();

//   if (ndim == 2) {
//     const int64_t K = x.size(1);
//     if (K > 0 && K <= 1024) {
//       block = dim3(static_cast<unsigned int>(K));
//       grid = dim3(static_cast<unsigned int>(x.size(0)));
//     } else {
//       block = dim3(256);
//       grid = dim3((N + 256 - 1) / 256);
//     }
//   } else {
//     block = dim3(256);
//     grid = dim3((N + 256 - 1) / 256);
//   }

//   float* x_ptr = reinterpret_cast<float*>(x.data_ptr());
//   float* y_ptr = reinterpret_cast<float*>(y.data_ptr());
//   if (!x_ptr || !y_ptr) {
//     throw std::runtime_error("Tensor data pointer is null");
//   }

//   elu_f32_kernel<<<grid, block>>>(x_ptr, y_ptr, N);

//   cudaError_t err = cudaGetLastError();
//   if (err != cudaSuccess) {
//     throw std::runtime_error("Failed to launch relu_f32_kernel: " +
//                              std::string(cudaGetErrorString(err)));
//   }
// }
