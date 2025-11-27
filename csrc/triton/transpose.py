import time

import torch
import triton
import triton.language as tl

@triton.jit
def transpose_kernel(
    in_ptr: tl.tensor,
    out_ptr: tl.tensor,
    M: tl.int32,
    N: tl.int32,
    TILE_SIZE: tl.constexpr,
):
    by = tl.program_id(0)
    bx = tl.program_id(1)

    by_offset = by * TILE_SIZE + tl.arange(0, TILE_SIZE)
    bx_offset = bx * TILE_SIZE + tl.arange(0, TILE_SIZE)

    block_offset = by_offset[:, None] * N + bx_offset[None, :]
    mask = (by_offset[:, None] < M) & (bx_offset[None, :] < N) 

    block = tl.load(in_ptr + block_offset, mask=mask)
    trans_block = tl.trans(block)

    trans_block_offset = bx_offset[:, None] * M + by_offset[None, :]
    tl.store(out_ptr+ trans_block_offset, trans_block, mask=mask.T)

def triton_transpose(a: torch.Tensor) -> torch.Tensor:
    assert a.ndim == 2, f"only support 2D tensor transpose, current dim is {a.ndim}"

    m, n = a.shape
    dtype = a.dtype
    device = a.device
    
    TILE_SIZE = 32
    b = torch.empty((m,n), dtype=dtype, device=device)

    grid_size = (triton.cdiv(m, TILE_SIZE), triton.cdiv(n, TILE_SIZE))
    transpose_kernel[grid_size](a, b, m, n, TILE_SIZE)
    return b

def test_triton_transpose():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float32
    
    if torch.cuda.is_available():
        print("\n3. Performance Test (GPU Only)...")
        perf_shapes = [
            (1024, 1024),    # 1Kx1K
            (4096, 4096),    # 4Kx4K
            (8192, 8192),    # 8Kx8K
            (16384, 8192),   # 16Kx8K
            (16384, 16384),   # 16Kx8K
        ]
        warmup_runs = 5
        test_runs = 100

        for (m, n) in perf_shapes:
            a = torch.randn(m, n, dtype=dtype, device=device)
            print(f"  Testing shape: {m}x{n}...")

            for _ in range(warmup_runs):
                triton_transpose(a)
                a.T.contiguous()
            torch.cuda.synchronize()

            start_time = time.time()
            for _ in range(test_runs):
                triton_out = triton_transpose(a)
            torch.cuda.synchronize()
            triton_avg_time = (time.time() - start_time) / test_runs

            start_time = time.time()
            for _ in range(test_runs):
                torch_out = a.T.contiguous()
            torch.cuda.synchronize()
            torch_avg_time = (time.time() - start_time) / test_runs

            data_size = 2 * m * n * dtype.itemsize 
            triton_bandwidth = data_size / (triton_avg_time * 1e9)  # GB/s
            torch_bandwidth = data_size / (torch_avg_time * 1e9)

            print(f"    Triton: {triton_avg_time:.4f} ms | Bandwidth: {triton_bandwidth:.2f} GB/s")
            print(f"    PyTorch: {torch_avg_time:.4f} ms | Bandwidth: {torch_bandwidth:.2f} GB/s")
            print(f"    Speedup: {torch_avg_time / triton_avg_time:.2f}x (Triton vs PyTorch)")

    print("\n=== All Tests Completed ===")

if __name__ == "__main__":
    test_triton_transpose()