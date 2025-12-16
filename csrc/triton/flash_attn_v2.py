import math
import torch

import triton
import triton.language as tl

try:
    from csrc.triton.flash_attn_v1 import attention_reference
except ImportError:
    from flash_attn_v1 import attention_reference


configs = [
    triton.Config({"BLOCK_M": BM, "BLOCK_N": BN}, num_stages=s, num_warps=w)
    for BM in [64, 128]
    for BN in [32, 64]
    for s in ([3, 4, 7])
    for w in [4, 8]
]


def keep(conf):
    BLOCK_M = conf.kwargs["BLOCK_M"]
    BLOCK_N = conf.kwargs["BLOCK_N"]
    if BLOCK_M * BLOCK_N < 128 * 128 and conf.num_warps == 8:
        return False
    return True


# the shape of (Q K V) is [batch_size, head_size, n_ctx, head_dim]
@triton.autotune(list(filter(keep, configs)), key=["N_CTX", "HEAD_DIM"])
@triton.jit
def flash_attn_kernel_v2(
    Q_ptr,
    K_ptr,
    V_ptr,
    sm_scale,
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
    stride_vk,
    stride_vn,
    stride_ob,
    stride_oh,
    stride_om,
    stride_on,
    B,  # Batch size
    H,  # Head size
    N_CTX,  # N size
    HEAD_DIM: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    IS_CAUSAL: tl.constexpr,
):
    tl.static_assert(BLOCK_N <= HEAD_DIM)

    start_m = tl.program_id(0)  # tl.div(N_CTX, BLOCK_M)
    bh_size = tl.program_id(1)  # Batch * Head

    start_b = bh_size // H
    start_h = bh_size % H

    # q, k, v, o
    q_offsets = start_b.to(tl.int64) * stride_qb + start_h.to(tl.int64) * stride_qh
    k_offsets = start_b.to(tl.int64) * stride_kb + start_h.to(tl.int64) * stride_kh
    v_offsets = start_b.to(tl.int64) * stride_vb + start_h.to(tl.int64) * stride_vh
    o_offsets = start_b.to(tl.int64) * stride_ob + start_h.to(tl.int64) * stride_oh

    q_block_ptr = tl.make_block_ptr(
        base=Q_ptr + q_offsets,
        shape=(N_CTX, HEAD_DIM),
        strides=(stride_qm, stride_qk),
        offsets=(start_m * BLOCK_M, 0),
        block_shape=(BLOCK_M, HEAD_DIM),
        order=(1, 0),
    )

    # Load K as (N_CTX, HEAD_DIM) and transpose to (HEAD_DIM, BLOCK_N) for dot product
    k_block_ptr = tl.make_block_ptr(
        base=K_ptr + k_offsets,
        shape=(HEAD_DIM, N_CTX),
        strides=(stride_kk, stride_kn),
        offsets=(0, 0),
        block_shape=(HEAD_DIM, BLOCK_N),
        order=(0, 1),
    )

    v_block_ptr = tl.make_block_ptr(
        base=V_ptr + v_offsets,
        shape=(N_CTX, HEAD_DIM),
        strides=(stride_vk, stride_vn),
        offsets=(0, 0),
        block_shape=(BLOCK_N, HEAD_DIM),
        order=(1, 0),
    )

    o_block_ptr = tl.make_block_ptr(
        base=O_ptr + o_offsets,
        shape=(N_CTX, HEAD_DIM),
        strides=(stride_om, stride_on),
        offsets=(start_m * BLOCK_M, 0),
        block_shape=(BLOCK_M, HEAD_DIM),
        order=(1, 0),
    )

    # initialize m_i, l_i, acc
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)

    qk_scale = sm_scale
    q = tl.load(q_block_ptr)

    if IS_CAUSAL:
        # Causal Attention (IS_CAUSAL=True):
        # Q[i] Only for K[0:i]
        #      K[0]  K[1]  K[2]  K[3]  K[4]
        # Q[0]  ✓     ✗     ✗     ✗     ✗    ← hi = 0
        # Q[1]  ✓     ✓     ✗     ✗     ✗    ← hi = BLOCK_M
        # Q[2]  ✓     ✓     ✓     ✗     ✗    ← hi = 2*BLOCK_M
        # Q[3]  ✓     ✓     ✓     ✓     ✗    ← hi = 3*BLOCK_M
        # Q[4]  ✓     ✓     ✓     ✓     ✓    ← hi = 4*BLOCK_M
        lo, hi = 0, tl.minimum((start_m + 1) * BLOCK_M, N_CTX)
    else:
        # Full Attention (IS_CAUSAL=False):
        #      K[0]  K[1]  K[2]  K[3]  K[4]
        # Q[0]  ✓     ✓     ✓     ✓     ✓
        # Q[1]  ✓     ✓     ✓     ✓     ✓
        # Q[2]  ✓     ✓     ✓     ✓     ✓
        # Q[3]  ✓     ✓     ✓     ✓     ✓
        # Q[4]  ✓     ✓     ✓     ✓     ✓
        lo, hi = 0, N_CTX

    m_offsets = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    n_offsets = tl.arange(0, BLOCK_N)

    for start_n in tl.range(lo, hi, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)

        k = tl.load(k_block_ptr)  # (HEAD_DIM, BLOCK_N)
        qk = tl.dot(q, k)  # (BLOCK_M, BLOCK_n)

        if IS_CAUSAL:
            causal_mask = m_offsets[:, None] >= (start_n + n_offsets)
            qk = tl.where(causal_mask, qk, float("-inf"))

        s = qk * qk_scale

        m_ij = tl.maximum(tl.max(s, 1), m_i)
        p = tl.exp(s - m_ij[:, None])

        alpha = tl.exp(m_i - m_ij)

        # update m_i and l_i
        l_i = l_i * alpha + tl.sum(p, 1)
        m_i = m_ij

        # update acc
        acc = acc * alpha[:, None]
        v = tl.load(v_block_ptr, boundary_check=(0, 1))
        p = p.to(tl.float16)  # use tensor core
        acc = tl.dot(p, v, acc)

        # move k, v ptr
        k_block_ptr = tl.advance(k_block_ptr, (0, BLOCK_N))
        v_block_ptr = tl.advance(v_block_ptr, (BLOCK_N, 0))

    # epilogue
    l_i = tl.where(l_i == 0, 1.0, l_i)
    acc = acc / l_i[:, None]
    tl.store(o_block_ptr, acc.to(O_ptr.type.element_ty), boundary_check=(0, 1))


def flash_attn_v2(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    causal: bool = True,
) -> torch.Tensor:
    HEAD_DIM_Q, HEAD_DIM_K = q.shape[-1], k.shape[-1]
    # when v is in float8_e5m2 it is transposed.
    HEAD_DIM_V = v.shape[-1]

    assert HEAD_DIM_Q == HEAD_DIM_K and HEAD_DIM_K == HEAD_DIM_V
    assert HEAD_DIM_K in {16, 32, 64, 128, 256}
    o = torch.empty_like(q)

    grid = lambda args: (
        triton.cdiv(q.shape[2], args["BLOCK_M"]),  # N / BLOCKM
        q.shape[0] * q.shape[1],  # B * H
        1,
    )

    sm_scale = 1.0 / math.sqrt(q.shape[-1])

    flash_attn_kernel_v2[grid](
        q,
        k,
        v,
        sm_scale,
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
        B=q.shape[0],
        H=q.shape[1],
        N_CTX=q.shape[2],
        HEAD_DIM=HEAD_DIM_K,
        IS_CAUSAL=causal,
    )
    return o


# @pytest.mark.parametrize("Z, H, N_CTX, HEAD_DIM", [(1, 2, 1024, 64)])
# @pytest.mark.parametrize("causal", [True])
def test_op(Z, H, N_CTX, HEAD_DIM, causal, dtype=torch.float16):
    torch.manual_seed(20)
    q = torch.empty((Z, H, N_CTX, HEAD_DIM), dtype=dtype, device="cuda")
    k = torch.empty((Z, H, N_CTX, HEAD_DIM), dtype=dtype, device="cuda")
    v = torch.empty((Z, H, N_CTX, HEAD_DIM), dtype=dtype, device="cuda")

    ref_out = attention_reference(q, k, v, causal)
    tri_out = flash_attn_v2(q, k, v, causal).half()

    diff = (ref_out - tri_out).abs()
    max_diff = diff.max().item()
    mean_diff = diff.mean().item()
    num_large_diff = (diff > 0.01).sum().item()

    print(f"Max diff: {max_diff:.6f}")
    print(f"Mean diff: {mean_diff:.6f}")
    print(f"Number of elements with diff > 0.01: {num_large_diff}")
    print(
        f"Ref out stats: min={ref_out.min().item():.6f}, max={ref_out.max().item():.6f}, mean={ref_out.mean().item():.6f}"
    )
    print(
        f"Tri out stats: min={tri_out.min().item():.6f}, max={tri_out.max().item():.6f}, mean={tri_out.mean().item():.6f}"
    )
    assert torch.allclose(ref_out, tri_out, atol=1e-2, rtol=0)


if __name__ == "__main__":
    test_op(1, 2, 1024, 64, True)
