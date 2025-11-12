#include <cutlass/cutlass.h>
#include <cutlass/gemm/device/gemm.h>
#include <iostream>
#include <vector>
#include <cmath>

#include "../utils.h"

void hgemm_cutlass_navie_f32(const float* A, const float* B, float* C, int M,
                             int N, int K, float alpha = 1.0f,
                             float beta = 0.0f) {
  using ElementA = float;
  using ElementB = float;
  using ElementC = float;
  using ElementAccumulator = float;

  using LayoutA = cutlass::layout::RowMajor;
  using LayoutB = cutlass::layout::RowMajor;
  using LayoutC = cutlass::layout::RowMajor;

  using GemmKernel = cutlass::gemm::device::Gemm<
      ElementA, LayoutA, ElementB, LayoutB, ElementC, LayoutC,
      ElementAccumulator, cutlass::arch::OpClassSimt, cutlass::arch::Sm70,
      cutlass::gemm::GemmShape<64, 64, 32>,
      cutlass::gemm::GemmShape<32, 32, 32>, cutlass::gemm::GemmShape<16, 8, 8>,
      cutlass::epilogue::thread::LinearCombination<
          ElementC, 1, ElementAccumulator, ElementAccumulator>,
      cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<2>>;

  cutlass::gemm::GemmCoord problem_size(M, N, K);

  LayoutA layout_A(K);
  LayoutB layout_B(N);
  LayoutC layout_C(N);

  cutlass::Status status =
      GemmKernel::invoke(problem_size, alpha, A, layout_A, B, layout_B, beta, C,
                         layout_C, C, layout_C);

  if (status != cutlass::Status::kSuccess) {
    std::cerr << "CUTLASS GEMM failed with error code: "
              << static_cast<int>(status) << std::endl;
    exit(EXIT_FAILURE);
  }
}

void hgemm_cutlass_navie_f32(torch::Tensor& a, torch::Tensor& b,
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