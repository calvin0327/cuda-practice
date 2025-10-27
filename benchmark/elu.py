import time
from typing import Optional

import torch
from ops import ops

torch.set_grad_enabled(False)


def run_benchmark(
    perf_func: callable,
    x: torch.Tensor,
    tag: str,
    out: Optional[torch.Tensor] = None,
    warmup: int = 10,
    iters: int = 1000,
    show_all: bool = False,
):
    if out is not None:
        out.fill_(0)

    if out is not None:
        for i in range(warmup):
            perf_func(x, out)
    else:
        for i in range(warmup):
            _ = perf_func(x)

    torch.cuda.synchronize()

    start = time.time()

    if out is not None:
        for i in range(iters):
            perf_func(x, out)
    else:
        for i in range(iters):
            _ = perf_func(x)

    torch.cuda.synchronize()

    end = time.time()

    total_time = (end - start) * 1000
    mean_time = total_time / iters

    out_info = f"out_{tag}"

    out_val = out.flatten().detach().cpu().numpy().tolist()[:2]
    out_val = [round(v, 8) for v in out_val]
    out_val = [f"{v:<12}" for v in out_val]
    print(f"{out_info:>18}: {out_val}, time:{mean_time:.8f}ms")
    if show_all:
        print(out)
    return out, mean_time


Ss = [1024, 2048, 4096]
Ks = [1024, 2048, 4096]
SKs = [(S, K) for S in Ss for K in Ks]

for S, K in SKs:
    print("-" * 85)
    print(" " * 40 + f"S={S}, K={K}")
    x = torch.randn((S, K)).cuda().float().contiguous()
    y = torch.zeros_like(x).cuda().float().contiguous()
    run_benchmark(ops.elu_f32, x, "f32", y)
    print("-" * 85)
    x_f16 = x.half().contiguous()
    y_f16 = y.half().contiguous()
    run_benchmark(ops.elu_f16, x_f16, "f16", y_f16)
    print("-" * 85)
