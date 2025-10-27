#include "ops.h"
#include "core/registration.h"

#include <Python.h>
#include <torch/library.h>
#include <torch/version.h>

TORCH_LIBRARY_EXPAND(TORCH_EXTENSION_NAME, ops) {
  ops.def("relu_f32(Tensor x, Tensor y) -> ()");
  ops.impl("relu_f32", torch::kCUDA, &relu_f32);
  ops.def("relu_f32x4(Tensor x, Tensor y) -> ()");
  ops.impl("relu_f32x4", torch::kCUDA, &relu_f32x4);

  ops.def("elu_f32(Tensor x, Tensor y) -> ()");
  ops.impl("elu_f32", torch::kCUDA, &elu_f32);
  ops.def("elu_f16(Tensor x, Tensor y) -> ()");
  ops.impl("elu_f16", torch::kCUDA, &elu_f16);

  ops.def("sgemm_naive_f32(Tensor a, Tensor b, Tensor c) -> ()");
  ops.impl("sgemm_naive_f32", torch::kCUDA, &sgemm_naive_f32);
  ops.def("sgemm_shared_f32(Tensor a, Tensor b, Tensor c) -> ()");
  ops.impl("sgemm_shared_f32", torch::kCUDA, &sgemm_shared_f32);
  ops.def("sgemm_t_8x8_shared_f32x4(Tensor a, Tensor b, Tensor c) -> ()");
  ops.impl("sgemm_t_8x8_shared_f32x4", torch::kCUDA, &sgemm_t_8x8_shared_f32x4);
}

REGISTER_EXTENSION(TORCH_EXTENSION_NAME)
