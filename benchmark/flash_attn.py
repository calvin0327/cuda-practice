import torch
import triton

from ops import ops

from csrc.triton.flash_attn_v1 import attention_reference, flash_attn_v1
from csrc.triton.flash_attn_v2 import flash_attn_v2


def get_perf_config():
    batch, n_heads, head_dim = 4, 32, 64
    configs = []
    for causal in [True, False]:
        # for causal in [False]:
        configs.append(
            triton.testing.Benchmark(
                x_names=["N_CTX"],
                x_vals=[128 * i for i in range(2, 8)],
                # x_vals=[2**i for i in range(2, 6)],
                line_arg="provider",
                line_vals=[
                    "pytorch attn",
                    "triton flash-attn-v1",
                    "triton flash-attn-v2",
                    "cute flash-attn-v2",
                ],
                line_names=[
                    "Pytorch attn [FP16]",
                    "Triton flash-attn-v1 [FP16]",
                    "Triton flash-attn-v2 [FP16]",
                    "Cute flash-attn-v2 [FP16]",
                ],
                styles=[("red", "-"), ("blue", "-"), ("green", "-"), ("pink", "-")],
                ylabel="TFLOPS",
                plot_name=f"flash-attention-batch{batch}-head{n_heads}-d{head_dim}-causal={causal}",
                args={
                    "H": n_heads,
                    "B": batch,
                    "D": head_dim,
                    "causal": causal,
                },
            )
        )
    return configs


@triton.testing.perf_report(get_perf_config())
def benchmark(B, H, N_CTX, D, provider, causal):
    warmup = 25
    rep = 100
    dtype = torch.float16

    torch.manual_seed(20)

    q = torch.randn((B, H, N_CTX, D), dtype=dtype, device="cuda")
    k = torch.randn((B, H, N_CTX, D), dtype=dtype, device="cuda")
    v = torch.randn((B, H, N_CTX, D), dtype=dtype, device="cuda")

    if provider == "pytorch attn":
        fn = lambda: attention_reference(q, k, v, causal)
    elif provider == "triton flash-attn-v1":
        fn = lambda: flash_attn_v1(q, k, v, causal)
    elif provider == "triton flash-attn-v2":
        fn = lambda: flash_attn_v2(q, k, v, causal)
    elif provider == "cute flash-attn-v2":
        fn = lambda: ops.flash_attn_v2_cute(q, k, v, causal)
    else:
        raise ValueError(f"Unknown provider: {provider}")

    ms = triton.testing.do_bench(fn, warmup=warmup, rep=rep)

    # Attention: Q @ K^T (2*B*H*N*N*D) + Score @ V (2*B*H*N*N*D) = 4*B*H*N^2*D
    flops_per_matmul = 2.0 * B * H * N_CTX * N_CTX * D
    total_flops = 2 * flops_per_matmul  # Q@K^T + P@V

    if causal:
        total_flops *= 0.5

    # TFLOPS = FLOPs / (ms * 1e-3) / 1e12 = FLOPs / ms * 1e-9
    tflops = total_flops / ms * 1e-9
    return tflops


if __name__ == "__main__":
    benchmark.run(show_plots=True, print_data=True, save_path="./benchmark/artifacts")
