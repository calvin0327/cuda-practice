import triton
import torch
import triton.testing
import triton.language as tl
import utils

@triton.autotune(
    configs = [
        triton.Config({"BLOCK_SIZE": BLOCK_SIZE}) for BLOCK_SIZE in [32, 64]
    ],
    key = ["M", "N"],
    use_cuda_graph=utils.use_cuda_graph,
)
@triton.jit
def transpose_kernel(
    in_ptr: tl.tensor,
    out_ptr: tl.tensor,
    M: tl.int32,
    N: tl.int32,
    BLOCK_SIZE: tl.constexpr,
):
    by = tl.program_id(0) # 1dim represents y
    bx = tl.program_id(1) # 2dim represents x

    # if by is 0, the by_offset is (0, 1, 2, 3 .... 32)
    by_offset = by * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    bx_offset = bx * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)

    # calculate the offset of each element in the tile, it is to a matrix
    block_offset = by_offset[:, None] * N + bx_offset[None, :]
    mask = (by_offset[:, None] < M) & (bx_offset[None, :] < N) 

    block = tl.load(in_ptr + block_offset, mask=mask)
    trans_block = tl.trans(block)

    # calculate the offset of each element in the tile of b matrix
    trans_block_offset = bx_offset[:, None] * M + by_offset[None, :]
    tl.store(out_ptr + trans_block_offset, trans_block, mask=mask.T)

def triton_transpose(a: torch.Tensor) -> torch.Tensor:
    assert a.ndim == 2, f"only support 2D tensor transpose, current dim is {a.ndim}"
    m, n = a.shape
    dtype = a.dtype
    device = a.device
    
    b = torch.empty((n, m), dtype=dtype, device=device)

    grid = lambda META: (triton.cdiv(m, META["BLOCK_SIZE"]), triton.cdiv(n, META["BLOCK_SIZE"]))
    transpose_kernel[grid](a, b, m, n)
    return b

@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["size"],
        x_vals=[256, 512, 1024, 2048, 4096, 8192],
        x_log=True,
        line_arg="impl",
        line_vals=["triton", "pytorch"],
        line_names=["Triton (Tile=32)", "PyTorch Native"],
        ylabel="Latency",
        plot_name="transpose-perf-benchmark",
        args={"dtype": torch.float32},
    )
)
def benchmark(size, impl, dtype):
    device = torch.device("cuda")
    a = torch.randn(size, size, dtype=dtype, device=device)

    if impl == "triton":
        fn = lambda: triton_transpose(a)
    elif impl == "pytorch":
        fn = lambda: a.T.contiguous()

    mean_latency = triton.testing.do_bench(fn)
    return mean_latency

def test_correctness():
    test_shapes = [(32, 32), (33, 65), (1024, 2048), (1, 1000), (1000, 1)]
    for m, n in test_shapes:
        a = torch.randn(m, n, device="cuda")
        triton_out = triton_transpose(a)
        pytorch_out = a.T.contiguous()

        triton.testing.assert_close(
            triton_out, pytorch_out, rtol=1e-4,
            err_msg=f"Shape {m}x{n} mismatch"
        )
        print(f"✓ Shape {m}x{n} passed")
    print("Correctness test completed!\n")

if __name__ == "__main__":
    print("=== Testing Correctness ===")
    test_correctness()

    print("=== Running Performance Benchmark ===")
    benchmark.run(print_data=True, save_path=None)
