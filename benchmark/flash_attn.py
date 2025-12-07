import torch
import triton

from csrc.triton.flash_attn_v1 import attention_reference, flash_attn_v1
from csrc.triton.flash_attn_v2 import flash_attn_v2


def get_perf_config():
    batch, n_heads, head_dim = 4, 32, 64
    configs = []
    for causal in [True, False]:
        configs.append(
            triton.testing.Benchmark(
                x_names=["N_CTX"],
                x_vals=[2**i for i in range(2, 10)],
                line_arg="provider",
                line_vals=[
                    "pytorch attn",
                    "triton flash-attn-v1",
                    "triton flash-attn-v2",
                ],
                line_names=[
                    "Pytorch attn [FP16]",
                    "Triton flash-attn-v1 [FP16]",
                    "Triton flash-attn-v2 [FP16]",
                ],
                styles=[("red", "-"), ("blue", "-", "green", "-")],
                ylabel="ms",
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

    q = torch.randn((B, H, N_CTX, D), dtype=dtype, device="cuda", requires_grad=True)
    k = torch.randn((B, H, N_CTX, D), dtype=dtype, device="cuda", requires_grad=True)
    v = torch.randn((B, H, N_CTX, D), dtype=dtype, device="cuda", requires_grad=True)
    # sm_scale = 1.3
    if provider == "pytorch attn":
        fn = lambda: attention_reference(q, k, v, causal)
    elif provider == "triton flash-attn-v1":
        fn = lambda: flash_attn_v1(q, k, v, causal)
    elif provider == "triton flash-attn-v2":
        fn = lambda: flash_attn_v2(q, k, v, causal)

    ms = triton.testing.do_bench(fn, warmup=warmup, rep=rep)

    flops_per_matmul = 2.0 * B * H * N_CTX * N_CTX * D
    total_flops = 2 * flops_per_matmul
    if causal:
        total_flops *= 0.5
    return total_flops / ms * 1e-9


if __name__ == "__main__":
    benchmark.run(show_plots=True, print_data=True, save_path="./benchmark")
