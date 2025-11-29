import torch
import triton
import triton.language as tl
import triton.testing

import utils


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE": 16}, num_warps=2, num_stages=1),
        triton.Config({"BLOCK_SIZE": 32}, num_warps=4, num_stages=1),
        triton.Config({"BLOCK_SIZE": 64}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_SIZE": 64}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK_SIZE": 128}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK_SIZE": 128}, num_warps=8, num_stages=4),
    ],
    key=["M", "N"],
    use_cuda_graph=utils.use_cuda_graph,
)
@triton.jit
def transpose_kernel(
    in_ptr,
    out_ptr,
    M,
    N,
    stride_in_row,
    stride_in_col,
    stride_out_row,
    stride_out_col,
    BLOCK_SIZE: tl.constexpr,
):
    bm = tl.program_id(axis=0)  # 1dim represents y
    bn = tl.program_id(axis=1)  # 2dim represents x

    # if by is 0, the by_offset is [0, 1, 2, 3 .... 31]
    bm_offset = bm * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    bn_offset = bn * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)

    # calculate the offset of each element in the tile, it is to a matrix
    # BLOCK_SIZE is 32, N is 32, block_offset is:
    # [[   0,    1,    2, ...,   29,   30,   31],
    #  [  32,   33,   34, ...,   61,   62,   63],
    #  [  64,   65,   66, ...,   93,   94,   95],
    #  ...,
    #  [ 928,  929,  930, ...,  957,  958,  959],
    #  [ 960,  961,  962, ...,  989,  990,  991],
    #  [ 992,  993,  994, ..., 1021, 1022, 1023]]
    block_offset = (
        bm_offset[:, None] * stride_in_row + bn_offset[None, :] * stride_in_col
    )
    # the mask is:
    #  [[ True,  True,  True, ...,  True,  True,  True],
    #   [ True,  True,  True, ...,  True,  True,  True],
    #   [ True,  True,  True, ...,  True,  True,  True],
    #   ...,
    #   [ True,  True,  True, ...,  True,  True,  True],
    #   [ True,  True,  True, ...,  True,  True,  True],
    #   [ True,  True,  True, ...,  True,  True,  True]]
    mask = (bm_offset[:, None] < M) & (bn_offset[None, :] < N)

    block = tl.load(in_ptr + block_offset, mask=mask)
    trans_block = tl.trans(block)

    # calculate the offset of each element in the tile of b matrix
    trans_block_offset = (
        bn_offset[:, None] * stride_out_row + bm_offset[None, :] * stride_out_col
    )
    tl.store(out_ptr + trans_block_offset, trans_block, mask=mask.T)


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE": 32, "GROUP_SIZE_M": 8}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_SIZE": 64, "GROUP_SIZE_M": 8}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_SIZE": 64, "GROUP_SIZE_M": 8}, num_warps=8, num_stages=3),
        triton.Config(
            {"BLOCK_SIZE": 128, "GROUP_SIZE_M": 4}, num_warps=8, num_stages=3
        ),
        triton.Config(
            {"BLOCK_SIZE": 128, "GROUP_SIZE_M": 4}, num_warps=8, num_stages=4
        ),
    ],
    key=["M", "N"],
    use_cuda_graph=utils.use_cuda_graph,
)
@triton.jit
def transpose_cache_swizzle_kernel(
    in_ptr,
    out_ptr,
    M,
    N,
    stride_in_row,
    stride_in_col,
    stride_out_row,
    stride_out_col,
    BLOCK_SIZE: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE)

    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)

    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    # if by is 0, the by_offset is [0, 1, 2, 3 .... 31]
    bm_offset = pid_m * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    bn_offset = pid_n * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)

    # calculate the offset of each element in the tile, it is to a matrix
    # BLOCK_SIZE is 32, N is 32, block_offset is:
    # [[   0,    1,    2, ...,   29,   30,   31],
    #  [  32,   33,   34, ...,   61,   62,   63],
    #  [  64,   65,   66, ...,   93,   94,   95],
    #  ...,
    #  [ 928,  929,  930, ...,  957,  958,  959],
    #  [ 960,  961,  962, ...,  989,  990,  991],
    #  [ 992,  993,  994, ..., 1021, 1022, 1023]]
    block_offset = (
        bm_offset[:, None] * stride_in_row + bn_offset[None, :] * stride_in_col
    )
    # the mask is:
    #  [[ True,  True,  True, ...,  True,  True,  True],
    #   [ True,  True,  True, ...,  True,  True,  True],
    #   [ True,  True,  True, ...,  True,  True,  True],
    #   ...,
    #   [ True,  True,  True, ...,  True,  True,  True],
    #   [ True,  True,  True, ...,  True,  True,  True],
    #   [ True,  True,  True, ...,  True,  True,  True]]
    mask = (bm_offset[:, None] < M) & (bn_offset[None, :] < N)

    block = tl.load(in_ptr + block_offset, mask=mask)
    trans_block = tl.trans(block)

    # calculate the offset of each element in the tile of b matrix
    trans_block_offset = (
        bn_offset[:, None] * stride_out_row + bm_offset[None, :] * stride_out_col
    )
    tl.store(out_ptr + trans_block_offset, trans_block, mask=mask.T)


def triton_transpose(a: torch.Tensor) -> torch.Tensor:
    assert a.ndim == 2, "only support 2D tensor"

    M, N = a.shape

    b = torch.empty((N, M), dtype=a.dtype, device=a.device)

    stride_in_row, stride_in_col = a.stride()
    stride_out_row, stride_out_col = b.stride()

    grid = lambda META: (
        triton.cdiv(M, META["BLOCK_SIZE"]),
        triton.cdiv(N, META["BLOCK_SIZE"]),
    )

    transpose_kernel[grid](
        a, b, M, N, stride_in_row, stride_in_col, stride_out_row, stride_out_col
    )
    return b


def triton_transpose_cache_swizzle(a: torch.Tensor) -> torch.Tensor:
    assert a.ndim == 2, "only support 2D tensor"

    M, N = a.shape

    b = torch.empty((N, M), dtype=a.dtype, device=a.device)

    stride_in_row, stride_in_col = a.stride()
    stride_out_row, stride_out_col = b.stride()

    grid = lambda META: (
        triton.cdiv(M, META["BLOCK_SIZE"]) * triton.cdiv(N, META["BLOCK_SIZE"]),
    )

    transpose_cache_swizzle_kernel[grid](
        a, b, M, N, stride_in_row, stride_in_col, stride_out_row, stride_out_col
    )
    return b


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["size"],
        x_vals=[128 * i for i in range(2, 101)],
        x_log=False,
        line_arg="impl",
        line_vals=["triton", "triton_swizzle", "pytorch"],
        line_names=["Triton", "Triton (swizzle)", "PyTorch (Native)"],
        styles=[("blue", "-"), ("red", "-"), ("green", "-")],
        ylabel="Bandwidth (GB/s)",
        plot_name="transpose-performace",
        args={"dtype": torch.float32},
    )
)
def benchmark(size, impl, dtype):
    a = torch.randn(size, size, dtype=dtype, device="cuda")

    x_bytes = size * size * a.element_size()

    if impl == "triton":
        fn = lambda: triton_transpose(a)
    elif impl == "triton_swizzle":
        fn = lambda: triton_transpose_cache_swizzle(a)
    elif impl == "pytorch":
        fn = lambda: a.T.contiguous()
    else:
        raise ValueError(f"Unsupported impl: {impl}")

    ms = triton.testing.do_bench(fn, warmup=100, rep=500)

    gbps = lambda ms: 2 * x_bytes / (ms * 1e-3) * 1e-9
    return gbps(ms)


def test():
    print("Starting test...")
    torch.manual_seed(0)

    shapes = [
        (32, 32),
        (128, 128),  # Block size
        (31, 31),
        (33, 65),  #
        (10, 4096),
        (4096, 10),  #
    ]
    dtypes = [torch.float32, torch.float16]

    for dtype in dtypes:
        for M, N in shapes:
            a = torch.randn((M, N), dtype=dtype, device="cuda")
            triton_out = triton_transpose(a)
            triton_swizzle_out = triton_transpose_cache_swizzle(a)
            torch_out = a.T.contiguous()

            triton.testing.assert_close(
                triton_out,
                torch_out,
                rtol=1e-2,
                atol=1e-2,
                err_msg=f"Failed on shape {M}x{N} dtype {dtype} to trion",
            )

            triton.testing.assert_close(
                triton_swizzle_out,
                torch_out,
                rtol=1e-2,
                atol=1e-2,
                err_msg=f"Failed on shape {M}x{N} dtype {dtype} to triton swizzle",
            )

    print("✓ Basic shapes & dtypes passed")

    a_big = torch.randn((256, 256), device="cuda", dtype=torch.float32)
    a_slice = a_big[::2, ::2]

    triton_slice_out = triton_transpose(a_slice)
    triton_swizzle_slice_out = triton_transpose_cache_swizzle(a_slice)
    torch_slice_out = a_slice.T.contiguous()
    triton.testing.assert_close(triton_slice_out, torch_slice_out)
    triton.testing.assert_close(triton_swizzle_slice_out, torch_slice_out)

    print("✓ Non-contiguous (sliced) input passed")
    print("All tests completed!\n")


if __name__ == "__main__":
    test()

    print("--- Running Performance Benchmark ---")
    benchmark.run(print_data=True, save_path=".", show_plots=False)
