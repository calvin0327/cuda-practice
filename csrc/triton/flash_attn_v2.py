import torch
import triton
import triton.language as tl


@triton.jit
def flash_attn_kernel_v1(
    Q_ptr,
    K_ptr,
    V_ptr,
    O_ptr,
    stride_qb,
    stride_qh,
    stride_qm,
    stride_qk,
    stride_kb,
    stride_kh,
    stride_kn,
    stride_kk,
    stride_vb,
    stride_vh,
    stride_vn,
    stride_vk,
    stride_ob,
    stride_oh,
    stride_om,
    stride_ok,
    sm_scale,
    N_CTX,  # sequence length
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
    IS_CAUSAL: tl.constexpr,
):
    pass


def flash_attn_v1(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
) -> torch.Tensor:
    b, h, n, d = q.shape
    assert q.dtype == k.dtype == v.dtype

    o = torch.empty_like(q)

    BLOCK_SIZE_M = 64
    BLOCK_SIZE_N = 64

    num_q_blocks = triton.cdiv(n, BLOCK_SIZE_M)
    grid = (b, h, num_q_blocks)
    flash_attn_kernel_v1[grid](
        q,
        k,
        v,
        q.stride(0),
        q.stride(1),
        q.stride(2),
        q.stride(3),
        k.stride(0),
        k.stride(1),
        k.stride(2),
        k.stride(3),
        v.stride(0),
        v.stride(1),
        v.stride(2),
        v.stride(3),
        o.stride(0),
        o.stride(1),
        o.stride(2),
        o.stride(3),
        N_CTX=n,
        dim
        
    )

    pass
