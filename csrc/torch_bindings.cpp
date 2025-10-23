#include "ops.h"
#include "core/registration.h"

#include <Python.h>
#include <torch/library.h>
#include <torch/version.h>

// // Defines the operators
// TORCH_LIBRARY(cuda_kernel, m) {
//   m.def("sgemm_naive_f32(Tensor a, Tensor b, Tensor c) -> ()");
// }

// // Registers CUDA implementations for mymuladd, mymul, myadd_out
// TORCH_LIBRARY_IMPL(cuda_kernel, CUDA, m) {
//   m.impl("sgemm_naive_f32", &sgemm_naive_f32);
// }

TORCH_LIBRARY_EXPAND(TORCH_EXTENSION_NAME, ops) {
  ops.def("sgemm_naive_f32(Tensor a, Tensor b, Tensor c) -> ()");
  ops.impl("sgemm_naive_f32", torch::kCUDA, &sgemm_naive_f32);
}

REGISTER_EXTENSION(TORCH_EXTENSION_NAME)
