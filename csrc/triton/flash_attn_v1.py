import math
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
    HEAD_DIM: tl.constexpr,
    IS_CAUSAL: tl.constexpr,
):
    b_id = tl.program_id(0)
    h_id = tl.program_id(1)
    q_id = tl.program_id(2)

    q_offsets = q_id * BLOCK_M + tl.arange(0, BLOCK_M)
    q_mask = q_offsets < N_CTX

    q_ptrs = (
        Q_ptr
        + b_id * stride_qb
        + h_id * stride_qh
        + q_offsets[:, None] * stride_qm
        + tl.arange(0, HEAD_DIM)[None, :] * stride_qk
    )
    q = tl.load(q_ptrs, mask=q_mask[:, None], other=0)

    o_ptrs = (
        O_ptr
        + b_id * stride_ob
        + h_id * stride_oh
        + q_offsets[:, None] * stride_om
        + tl.arange(0, HEAD_DIM)[None, :] * stride_ok
    )

    m_i = tl.full([BLOCK_M], float("-inf"), dtype=tl.float32)
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    o_i = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)

    num_kv_blocks = tl.cdiv(N_CTX, BLOCK_N)

    for j in range(0, num_kv_blocks):
        kv_offsets = j * BLOCK_N + tl.arange(0, BLOCK_N)
        kv_mask = kv_offsets < N_CTX

        k_ptrs = (
            K_ptr
            + b_id * stride_kb
            + h_id * stride_kh
            + kv_offsets[:, None] * stride_kn
            + tl.arange(0, HEAD_DIM)[None, :] * stride_kk
        )
        k = tl.load(k_ptrs, mask=kv_mask[:, None], other=0.0)

        v_ptrs = (
            V_ptr
            + b_id * stride_vb
            + h_id * stride_vh
            + kv_offsets[:, None] * stride_vn
            + tl.arange(0, HEAD_DIM)[None, :] * stride_vk
        )
        v = tl.load(v_ptrs, mask=kv_mask[:, None], other=0.0)

        # shape: [BLOCK_M, BLOCK_N]
        s = tl.dot(q, tl.trans(k)) * sm_scale

        # Apply masks
        valid_mask = q_mask[:, None] & kv_mask[None, :]

        if IS_CAUSAL:
            causal_mask = q_offsets[:, None] >= kv_offsets[None, :]
            valid_mask = valid_mask & causal_mask

        s = tl.where(valid_mask, s, float("-inf"))

        m_new = tl.maximum(m_i, tl.max(s, axis=1))

        alpha = tl.exp(m_i - m_new)

        p = tl.exp(s - m_new[:, None])
        l_new = l_i * alpha + tl.sum(p, axis=1)
        pv = tl.dot(p.to(v.dtype), v)

        o_i = alpha[:, None] * o_i + pv

        # update the max and l
        l_i = l_new
        m_i = m_new

    o_i = o_i / l_i[:, None]
    tl.store(o_ptrs, o_i.to(O_ptr.dtype.element_ty), mask=q_mask[:, None])


def flash_attn_v1(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    causal: bool = True,
) -> torch.Tensor:
    assert q.dim() == 4, f"Expected 4D tensor, got {q.dim()}D"
    assert q.shape == k.shape == v.shape, "Q, K, V shapes must match"
    assert q.dtype in [torch.float16, torch.bfloat16, torch.float32]
    assert q.is_cuda, "Input must be on CUDA"

    B, H, N, D = q.shape

    scale = 1.0 / math.sqrt(D)

    o = torch.empty_like(q)

    BLOCK_SIZE_M = 64
    BLOCK_SIZE_N = 64

    num_q_blocks = triton.cdiv(N, BLOCK_SIZE_M)
    grid = (B, H, num_q_blocks)
    flash_attn_kernel_v1[grid](
        q,
        k,
        v,
        o,
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
        sm_scale=scale,
        N_CTX=N,
        HEAD_DIM=D,
        BLOCK_M=BLOCK_SIZE_M,
        BLOCK_N=BLOCK_SIZE_N,
        IS_CAUSAL=causal,
    )

    return o


def attention_reference(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    causal: bool = True,
) -> torch.Tensor:
    scale = 1.0 / math.sqrt(q.size(-1))
    s = torch.matmul(q, torch.transpose(k, -2, -1)) * scale

    if causal:
        N = q.size(-2)
        mask = torch.triu(
            torch.ones(N, N, device=q.device, dtype=torch.bool), diagonal=1
        )
        s.masked_fill_(mask, float("-inf"))

    p = torch.softmax(s, dim=-1)
    o = torch.matmul(p, v)
    return o


def test():
    print("-" * 60)
    print("FlashAttention Test")
    print("-" * 60)

    torch.manual_seed(42)

    test_configs = [
        # (B, H, N, D, causal)
        (1, 1, 64, 64, False),
        (1, 1, 128, 64, False),
        (2, 4, 256, 64, False),
        (2, 8, 512, 64, False),
        (1, 4, 1024, 64, False),
        (2, 8, 2048, 128, False),
        # Causal
        (1, 1, 64, 64, True),
        (2, 4, 256, 64, True),
        (2, 8, 512, 64, True),
        (1, 4, 1024, 128, True),
    ]

    for B, H, N, D, causal in test_configs:
        q = torch.randn((B, H, N, D), device="cuda", dtype=torch.float16)
        k = torch.randn((B, H, N, D), device="cuda", dtype=torch.float16)
        v = torch.randn((B, H, N, D), device="cuda", dtype=torch.float16)

        ref_o = attention_reference(q, k, v, causal)
        ref_o = ref_o.half()

        triton_o = flash_attn_v1(q, k, v, causal)

        max_diff = (ref_o - triton_o).abs().max().item()
        mean_diff = (ref_o - triton_o).abs().mean().item()

        status = "✓" if max_diff < 0.01 else "✗"
        causal_str = "causal" if causal else "full"

        print(
            f"{status} B={B}, H={H}, N={N:4d}, D={D:3d}, {causal_str:6s} | "
            f"max_diff={max_diff:.6f}, mean_diff={mean_diff:.6f}"
        )
    print("\ntest completed!")


if __name__ == "__main__":
    test()
