import os
import torch
import triton
import triton.language as tl

import utils


ref_lib = "cuBLAS" if utils.is_cuda() else "rocBLAS"
TORCH_HAS_FP8 = hasattr(torch, "float8_e5m2")


def get_triton_configs():
    return [
        triton.Config(
            {
                "BLOCK_SIZE_M": 128,
                "BLOCK_SIZE_N": 256,
                "BLOCK_SIZE_K": 64,
            },
            num_stages=3,
            num_warps=8,
        ),
        triton.Config(
            {
                "BLOCK_SIZE_M": 64,
                "BLOCK_SIZE_N": 256,
                "BLOCK_SIZE_K": 32,
            },
            num_stages=4,
            num_warps=4,
        ),
        triton.Config(
            {
                "BLOCK_SIZE_M": 128,
                "BLOCK_SIZE_N": 128,
                "BLOCK_SIZE_K": 32,
            },
            num_stages=4,
            num_warps=4,
        ),
    ]


@triton.autotune(configs=get_triton_configs(), key=["M", "N", "K"])
@triton.jit
def matmul_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    M,
    N,
    K,
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)

    m_offset = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    n_offset = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    k_offset = tl.arange(0, BLOCK_SIZE_K)

    # the shape of a_ptrs is [BLOCK_SIZE_M, BLOCK_SIZE_K]
    # the shape of b_ptrs is [BLOCK_SIZE_K, BLOCK_SIZE_N]
    a_ptrs = a_ptr + (m_offset[:, None] * stride_am + k_offset[None, :] * stride_ak)
    b_ptrs = b_ptr + (n_offset[None, :] * stride_bn + k_offset[:, None] * stride_bk)

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        # the shape of (mask=k_offset[None, :] < K - k * BLOCK_SIZE_K) is [1, BLOCK_SIZE_K]
        # the shape of (mask=k_offset[:, None] < K - k * BLOCK_SIZE_K) is [BLOCK_SIZE_K, 1]
        # mask_a: [[ True,  True,  True,  True,  True,  True,  True,  True,  True, .... True ]]
        # mask_b: [[True], [True], [True], [True], [True], [True],[True], [True] .... [True]]
        mask_a = (m_offset[:, None] < M) & k_offset[None, :] < K - k * BLOCK_SIZE_K
        mask_b = (n_offset[None, :] < N) & k_offset[:, None] < K - k * BLOCK_SIZE_K
        a = tl.load(a_ptrs, mask=mask_a, other=0.0)
        b = tl.load(b_ptrs, mask=mask_b, other=0.0)

        accumulator = tl.dot(a, b, accumulator)

        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk

    c = accumulator.to(tl.bfloat16)

    c_ptrs = c_ptr + (m_offset[:, None] * stride_cm + n_offset[None, :] * stride_cn)
    # the shape of (m_offset[:, None] < M) & (n_offset[None, :] < N) is [BLOCK_SIZE_M, BLOCK_SIZE_N]
    # [[ True,  True,  True, ...,  True,  True,  True],
    #   ...,
    #  [ True,  True,  True, ...,  True,  True,  True],
    #  [ True,  True,  True, ...,  True,  True,  True]]
    mask = (m_offset[:, None] < M) & (n_offset[None, :] < N)
    tl.store(c_ptrs, c, mask=mask)


def matmul(
    a: torch.Tensor,
    b: torch.Tensor,
) -> torch.tensor:
    assert a.shape[0] == b.shape[1], "imcompatible dimensions"
    assert a.is_contiguous(), "matrix a must be contiguous"
    M, K = a.shape
    K, N = b.shape

    c = torch.empty((M, N), device=a.device, dtype=torch.bfloat16)
    grid = lambda META: (
        triton.cdiv(M, META["BLOCK_SIZE_M"]),
        triton.cdiv(N, META["BLOCK_SIZE_N"]),
    )
    matmul_kernel[grid](
        a,
        b,
        c,
        M,
        N,
        K,
        a.stride(0),
        a.stride(1),
        b.stride(0),
        b.stride(1),
        c.stride(0),
        c.stride(1),
    )
    return c


def test():
    torch.manual_seed(0)
    a = torch.randn((512, 512), device="cuda", dtype=torch.bfloat16)
    b = torch.randn((512, 512), device="cuda", dtype=torch.bfloat16)

    triton_out = matmul(a, b)
    torch_out = torch.matmul(a, b)
    print(f"triton_output_with_fp16_inputs={triton_out}")
    print(f"torch_output_with_fp16_inputs={torch_out}")

    triton.testing.assert_close(
        triton_out, torch_out, rtol=1e-4, err_msg=f"Triton and Torch differ"
    )

    if TORCH_HAS_FP8 and utils.is_cuda():
        torch.manual_seed(0)
        a = torch.randn((512, 512), device="cuda", dtype=torch.bfloat16)
        b = torch.randn((512, 512), device="cuda", dtype=torch.bfloat16)
        a = a.to(torch.float8_e5m2)
        b = b.T
        b = b.to(torch.float8_e5m2)
        triton_out = matmul(a, b)
        torch_out = torch.matmul(a.to(torch.float16), b.to(torch.float16))
        triton.testing.assert_close(
            triton_out, torch_out, rtol=0.125, err_msg=f"Triton and Torch differ"
        )

    print("Correctness test completed!\n")


def get_perf_report_configs() -> []:
    configs = []
    for fp8_inputs in [False, True]:
        if fp8_inputs and (not TORCH_HAS_FP8 or not utils.is_cuda()):
            continue
        configs.append(
            triton.testing.Benchmark(
                x_names=[
                    "M",
                    "N",
                    "K",
                ],
                x_vals=[128 * i for i in range(2, 33)],
                line_arg="provider",
                line_vals=(["triton"] if fp8_inputs else [ref_lib.lower(), "triton"]),
                line_names=(["Triton"] if fp8_inputs else [ref_lib, "Triton"]),
                styles=[("green", "-"), ("blue", "-")],
                ylabel="TFLOPS",
                plot_name="matmul-performance-" + ("fp16" if not fp8_inputs else "fp8"),
                args={"fp8_inputs": fp8_inputs},
            )
        )
    return configs


@triton.testing.perf_report(get_perf_report_configs())
def benchmark(M, N, K, provider, fp8_inputs):
    a = torch.randn((M, K), device="cuda", dtype=torch.float16)
    b = torch.randn((K, N), device="cuda", dtype=torch.float16)
    if TORCH_HAS_FP8 and fp8_inputs:
        a = a.to(torch.float8_e5m2)
        b = b.T
        b = b.to(torch.float8_e5m2)
    quantiles = [0.5, 0.2, 0.8]
    if provider == ref_lib.lower():
        ms, min_ms, max_ms = triton.testing.do_bench(
            lambda: torch.matmul(a, b), quantiles=quantiles
        )
    if provider == "triton":
        ms, min_ms, max_ms = triton.testing.do_bench(
            lambda: matmul(a, b), quantiles=quantiles
        )
    perf = lambda ms: 2 * M * N * K * 1e-12 / (ms * 1e-3)
    return perf(ms), perf(max_ms), perf(min_ms)


if __name__ == "__main__":
    test()
    benchmark.run(show_plots=True, save_path=".", print_data=True)
