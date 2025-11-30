import torch
import triton
import triton.language as tl
from triton.runtime import driver


device = torch.cuda.current_device()
properties = driver.active.utils.get_device_properties(device)
NUM_SM = properties["multiprocessor_count"]
NUM_REGS = properties["max_num_regs"]
SIZE_SMEM = properties["max_shared_mem"]
WARP_SIZE = properties["warpSize"]
target = triton.runtime.driver.active.get_current_target()
kernels = {}


@triton.jit
def softmax_kernel(
    input_ptr,
    output_ptr,
    stride_input_row,
    stride_output_row,
    n_rows,
    n_cols,
    BLOCK_SIZE: tl.constexpr,
    num_stages: tl.constexpr,
):
    row_start = tl.program_id(axis=0)
    row_step = tl.num_programs(axis=0)

    for idx in tl.range(row_start, n_rows, row_step, num_stages=num_stages):
        offsets = tl.arange(0, BLOCK_SIZE)
        input_ptrs = input_ptr + idx * stride_input_row + offsets
        mask = offsets < n_cols

        row = tl.load(input_ptrs, mask=mask, other=-float("inf"))

        # exp(x - max(x)) / sum(exp(x - max(x)))
        row_minus_max = row - tl.max(row, axis=0)
        numerator = tl.exp(row_minus_max)
        denominator = tl.sum(numerator, axis=0)
        softmax_output = numerator / denominator

        output_ptrs = output_ptr + idx * stride_output_row + offsets
        tl.store(output_ptrs, softmax_output, mask=mask)


def triton_softmax(x):
    n_rows, n_cols = x.shape

    # The block size of each loop iteration is the smallest power
    # of two greater than the number of columns in `x`
    BLOCK_SIZE = triton.next_power_of_2(n_cols)
    num_warps = 8
    num_stages = 4 if SIZE_SMEM > 200000 else 2

    y = torch.empty_like(x)

    # autotune, the key is BLOCK_SIZE
    kernel, num_programs = kernels.get(BLOCK_SIZE, (None, 0))
    if kernel is None:
        kernel = softmax_kernel.warmup(
            x,
            y,
            x.stride(0),
            y.stride(0),
            n_rows,
            n_cols,
            BLOCK_SIZE=BLOCK_SIZE,
            num_stages=num_stages,
            num_warps=num_warps,
            grid=(1,),
        )
        kernel._init_handles()
        n_regs = kernel.n_regs
        size_smem = kernel.metadata.shared
        occupancy = NUM_REGS // (n_regs * WARP_SIZE * num_warps)
        occupancy = min(occupancy, SIZE_SMEM // size_smem)
        num_programs = NUM_SM * occupancy
        kernels[BLOCK_SIZE] = (kernel, num_programs)

    num_programs = min(num_programs, n_rows)

    softmax_kernel[(num_programs, 1, 1)](
        x,
        y,
        x.stride(0),
        y.stride(0),
        n_rows,
        n_cols,
        BLOCK_SIZE=BLOCK_SIZE,
        num_stages=num_stages,
    )
    return y


def naive_softmax(x):
    # x.max(dim=1) is (values, indices)
    x_max = x.max(dim=1)[0]

    z = x - x_max[:, None]
    numerator = torch.exp(z)

    # numerator.sum is [sum0, sum1, ...]
    denominator = numerator.sum(dim=1)

    ret = numerator / denominator[:, None]
    return ret


def test():
    torch.manual_seed(42)
    test_cases = [
        (1, 128),
        (2, 64),
        (128, 256),
        (256, 1024),
        (64, 1000),
        (100, 100),
    ]
    for M, N in test_cases:
        x = torch.randn(M, N, device="cuda", dtype=torch.float32)

        triton_out = triton_softmax(x)
        torch_out = torch.softmax(x, dim=1)

        triton.testing.assert_close(
            triton_out,
            torch_out,
            rtol=1e-4,
            atol=1e-4,
            err_msg=f"Shape ({M}, {N}) failed",
        )
        print(f"✓ Shape ({M:4d}, {N:4d}) passed")
    print("All correctness tests passed!\n")


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["N"],
        x_vals=[128 * i for i in range(2, 100)],
        line_arg="impl",
        line_vals=["triton", "naive", "torch"],
        line_names=["Triton", "Naive PyTorch", "torch.softmax"],
        styles=[("blue", "-"), ("red", "-"), ("green", "-")],
        ylabel="GB/s",
        plot_name="softmax-performance",
        args={"M": 4096, "dtype": torch.float32},
    )
)
def benchmark(M, N, impl, dtype):
    x = torch.randn(M, N, device="cuda", dtype=dtype)

    if impl == "triton":
        ms = triton.testing.do_bench(lambda: triton_softmax(x))
    elif impl == "naive":
        ms = triton.testing.do_bench(lambda: naive_softmax(x))
    elif impl == "torch":
        ms = triton.testing.do_bench(lambda: torch.softmax(x, dim=1))
    else:
        raise ValueError(f"Unknown impl: {impl}")

    gbps = lambda ms: 2 * x.nelement() * x.element_size() * 1e-9 / (ms * 1e-3)
    return gbps(ms)


if __name__ == "__main__":
    test()
    benchmark.run(print_data=True, save_path=".")
