import time
import torch
from typing import Optional

from ops import ops

MAX_TFLOPS = -1


def run_benchmark(
    perf_func: callable,
    a: torch.Tensor,
    b: torch.Tensor,
    tag: str,
    out: Optional[torch.Tensor] = None,
    stages: int = -1,
    swizzle: bool = False,
    swizzle_stride: int = 1,
    warmup: int = 2,
    iters: int = 20,
    show_all: bool = False,
):

    global MAX_TFLOPS

    M = a.size(0)
    K = a.size(1)
    N = b.size(1)

    if a.size(0) > 1024 or a.size(1) >= 1024 or b.size(1) > 1024:
        iters = 10

    if swizzle:
        # make swizzle stride as N/4 and multiples of 256
        swizzle_stride = int((int(N / 8) // 256) * 256)
        swizzle_stride = swizzle_stride if swizzle_stride >= 256 else 1
        swizzle = swizzle if swizzle_stride >= 256 else False
    else:
        swizzle_stride = 1  # means no thread block swizzle

    if stages:
        assert swizzle_stride is not None

    if out is not None:
        out.fill_(0)
    if out is not None:
        for i in range(warmup):
            if stages > 1:
                perf_func(a, b, out, stages, swizzle, swizzle_stride)
            else:
                perf_func(a, b, out)
    else:
        for i in range(warmup):
            _ = perf_func(a, b)

    torch.cuda.synchronize()
    start = time.time()
    # iters
    if out is not None:
        for i in range(iters):
            if stages > 1:
                perf_func(a, b, out, stages, swizzle, swizzle_stride)
            else:
                perf_func(a, b, out)
    else:
        for i in range(iters):
            out = perf_func(a, b)
    torch.cuda.synchronize()
    end = time.time()
    total_time = (end - start) * 1000  # ms
    mean_time = total_time / iters
    out_info = f"out_{tag}"
    out_val = out.flatten()[:2].detach().cpu().numpy().tolist()[:3]
    out_val = [round(v, 8) for v in out_val]
    out_val = [f"{v:<12}"[:10] for v in out_val]
    TFLOPS = (2 * M * N * K) * 1e-9 / (mean_time)
    mean_time = str(f"{mean_time:<12}")[:8]
    swizzle_stride = "NOOP" if swizzle_stride == 1 else swizzle_stride

    # caculate TFLOPS improved.
    if TFLOPS > MAX_TFLOPS:
        if MAX_TFLOPS > 0:
            improve = ((TFLOPS - MAX_TFLOPS) / MAX_TFLOPS) * 100
            improve = round(improve, 2)
        else:
            improve = 0
        MAX_TFLOPS = TFLOPS
        print(
            f"{out_info:>35}: {out_val}, time:{mean_time}ms, "
            f"swizzle: {swizzle_stride:<4}, TFLOPS: {TFLOPS:<6.2f}(+{improve:.2f}%)"
        )
    else:
        print(
            f"{out_info:>35}: {out_val}, time:{mean_time}ms, "
            f"swizzle: {swizzle_stride:<4}, TFLOPS: {TFLOPS:<6.2f}"
        )
    if show_all:
        print(out)
    return out, mean_time


Ms = [4096, 8192, 16384]
Ns = [4096, 8192, 16384]
Ks = [2048, 4096, 8192]
MAX_M, MAX_N, MAX_K = 16384, 16384, 8192
# pre allocate for fast profiling.
A = torch.randn((MAX_M, MAX_K), dtype=torch.float).cuda()
B = torch.randn((MAX_K, MAX_N), dtype=torch.float).cuda()
C = torch.randn((MAX_M, MAX_N), dtype=torch.float).cuda()
torch.cuda.synchronize()

MNKs = [(M, N, K) for M in Ms for N in Ns for K in Ks]
for M, N, K in MNKs:
    MAX_TFLOPS = -1
    print("-" * 130)
    print(" " * 55 + f"M={M}, N={N}, K={K}")
    a = A[:M, :K].contiguous()
    b = B[:K, :N].contiguous()
    c = C[:M, :N].contiguous()
    torch.cuda.synchronize()

    # CUDA Cores FP32
    # run_benchmark(lib.sgemm_naive_f32, a, b, "f32(naive)", c)
    run_benchmark(ops.sgemm_naive_f32, a, b, "sgemm_naive_f32", c)
    torch.cuda.synchronize()
    print("-" * 130)
