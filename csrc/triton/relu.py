import torch
import triton
import triton.language as tl


@triton.jit
def relu_kernel(
    in_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(in_ptr + offsets, mask=mask)
    y = tl.maximum(x, 0.0)
    tl.store(out_ptr + offsets, y, mask=mask)


def triton_relu(x: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    n_elements = x.numel()

    BLOCK_SIZE = 1024

    grid = lambda meta: (triton.cdiv(n_elements, BLOCK_SIZE),)
    relu_kernel[grid](
        x,
        out,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return out


if __name__ == "__main__":
    x = torch.randn(1000000, device="cuda") * 10

    y_triton = triton_relu(x)
    y_torch = torch.relu(x)

    assert torch.allclose(y_triton, y_torch)
