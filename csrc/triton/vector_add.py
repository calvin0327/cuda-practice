import time

import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[triton.Config({"BLOCK_SIZE": BLOCK_SIZE}) for BLOCK_SIZE in (32, 64, 128)],
    key=["N"],
)
@triton.jit
def vector_add_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    N,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)

    # BLOCK_SIZE is 32, the offsets is:
    # [ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16,
    #   17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31]
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    # [ True,  True,  True,  True,  True,  True,  True,  True,  True,
    #   True,  True,  True,  True,  True,  True,  True,  True,  True,
    #   True,  True,  True,  True,  True,  True,  True,  True,  True,
    #   True,  True,  True,  True,  True]
    mask = offsets < N

    a = tl.load(a_ptr + offsets, mask=mask)
    b = tl.load(b_ptr + offsets, mask=mask)
    c = a + b

    tl.store(c_ptr + offsets, c, mask=mask)


def triton_vector_add(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    assert a.shape == b.shape, "the shape of input tensor must be same"
    assert a.device == b.device, "input tensor must be the same device"
    assert a.dtype == b.dtype, "the type of input tensor must be same"
    assert a.ndim == 1, "only support 1 dim"

    # Configure block size and grid size
    n = a.shape[0]
    c = torch.empty_like(a)

    grid = lambda META: (triton.cdiv(n, META["BLOCK_SIZE"]),)

    # Launch Triton kernel
    vector_add_kernel[grid](
        a_ptr=a,
        b_ptr=b,
        c_ptr=c,
        N=n,
    )
    return c


def test_vector_add():
    # Test different vector sizes (small, medium, large)
    test_sizes = [1024, 1024 * 1024, 1024 * 1024 * 100]  # 1K, 1M, 100M
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if not torch.cuda.is_available():
        print(
            "Warning: No CUDA device detected, running on CPU (performance may be poor)"
        )

    for size in test_sizes:
        print(f"\n=== Testing vector size: {size:,} ===")

        # Generate random input vectors (float32 on target device)
        a = torch.randn(size, dtype=torch.float32, device=device)
        b = torch.randn(size, dtype=torch.float32, device=device)

        # 1. Correctness verification
        c_triton = triton_vector_add(a, b)
        c_torch = a + b
        max_error = torch.max(torch.abs(c_triton - c_torch))
        print(
            f"Correctness: {'Passed' if max_error < 1e-5 else 'Failed'}, Max error: {max_error:.6e}"
        )

        # 2. Performance comparison (with warmup)
        # Warmup runs to eliminate initialization overhead
        for _ in range(5):
            triton_vector_add(a, b)
            a + b
        torch.cuda.synchronize()  # Wait for all GPU operations to complete

        # Triton performance test
        start_time = time.time()
        for _ in range(100):
            triton_vector_add(a, b)
        torch.cuda.synchronize()
        triton_time = (time.time() - start_time) / 100  # Average time per run

        # PyTorch performance test
        start_time = time.time()
        for _ in range(100):
            a + b
        torch.cuda.synchronize()
        torch_time = (time.time() - start_time) / 100  # Average time per run

        # Calculate bandwidth (GB/s): each float32 element is 4B, total data: 3 * size * 4B (read a+b, write c)
        bandwidth_triton = (3 * size * 4) / (triton_time * 1e9)
        bandwidth_torch = (3 * size * 4) / (torch_time * 1e9)

        print(
            f"Triton average time: {triton_time:.4f} ms, Bandwidth: {bandwidth_triton:.2f} GB/s"
        )
        print(
            f"PyTorch average time: {torch_time:.4f} ms, Bandwidth: {bandwidth_torch:.2f} GB/s"
        )
        print(
            f"Performance ratio: Triton is {torch_time / triton_time:.2f}x faster than PyTorch"
        )


if __name__ == "__main__":
    test_vector_add()
