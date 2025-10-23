// #include <algorithm>
// #include <cuda_fp16.h>
// #include <cuda_runtime.h>
// #include <float.h>
// #include <stdio.h>
// #include <stdlib.h>
// #include <torch/extension.h>
// #include <torch/types.h>
// #include <vector>

// #define FLOAT4(value) (reinterpret_cast<float4*>(&(value))[0])

// __global__ void relu_f32_kernel(float* x, float* y, int N) {
//   int idx = blockIdx.x * blockDim.x + threadIdx.x;
//   if (idx < N) {
//     y[idx] = fmaxf(0.0f, x[idx]);
//   }
// }

// __global__ void relu_f32x4_kernel(float* x, float* y, int N) {
//   int idx = (blockIdx.x * blockDim.x + threadIdx.x) * 4;
//   if (idx < N) {
//     float4 reg_x = FLOAT4(x[idx]);
//     float4 reg_y;
//     reg_y.x = fmaxf(0.0f, reg_x.x);
//     reg_y.y = fmaxf(0.0f, reg_x.y);
//     reg_y.z = fmaxf(0.0f, reg_x.z);
//     reg_y.w = fmaxf(0.0f, reg_x.w);
//     FLOAT4(y[idx]) = reg_y;
//   }
// }

// void relu_f32(torch::Tensor x, torch::Tensor y) {
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

//   relu_f32_kernel<<<grid, block>>>(x_ptr, y_ptr, N);

//   cudaError_t err = cudaGetLastError();
//   if (err != cudaSuccess) {
//     throw std::runtime_error("Failed to launch relu_f32_kernel: " +
//                              std::string(cudaGetErrorString(err)));
//   }
// }
