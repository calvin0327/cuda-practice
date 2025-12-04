import torch
import triton

from csrc.triton.flash_attn_v1 import attention_reference, flash_attn_v1


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["N"],
        x_vals=[2**i for i in range(2, 10)],
        line_arg="provider",
        line_vals=["pytoch attn", "triton flash-attn"],
        line_names=["Pytoch Attn", "Triton FlashAttn"],
        styles=[("green", "-"), ("blue", "-")],
        ylabel="TFLOPS",
        plot_name="flash-attention-performance",
        args={"B": 4, "H": 32, "D": 64, "causal": False},
    ),
)
def benchmark(B, H, N, D, provider, causal):
    q = torch.randn((B, H, N, D), device="cuda", dtype=torch.float32)
    k = torch.randn((B, H, N, D), device="cuda", dtype=torch.float32)
    v = torch.randn((B, H, N, D), device="cuda", dtype=torch.float32)

    # Calculate FLOPS: QK^T (B*H*N*N*D) + PV (B*H*N*N*D) = 2*B*H*N*N*D
    flops = 2 * B * H * N * N * D

    if provider == "pytoch attn":
        ms = triton.testing.do_bench(lambda: attention_reference(q, k, v, causal))
    elif provider == "triton flash-attn":
        ms = triton.testing.do_bench(lambda: flash_attn_v1(q, k, v, causal))

    tflops = flops / (ms * 1e-3) / 1e12
    return tflops


if __name__ == "__main__":
    benchmark.run(show_plots=True, print_data=True, save_path="./benchmark")
