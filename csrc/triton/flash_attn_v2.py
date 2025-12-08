import math
import torch

import triton
import triton.language as tl

try:
    from csrc.triton.flash_attn_v1 import attention_reference
except ImportError:
    from flash_attn_v1 import attention_reference


# We don't run auto-tuning every time to keep the tutorial fast. Keeping
# the code below and commenting out the equivalent parameters is convenient for
# re-tuning.
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


@triton.autotune(list(filter(keep, configs)), key=["N_CTX", "HEAD_DIM"])
@triton.jit
def flash_attn_kernel_v2(
    Q,
    K,
    V,
    sm_scale,
    M,
    Out,
    stride_qz,
    stride_qh,
    stride_qm,
    stride_qk,
    stride_kz,
    stride_kh,
    stride_kn,
    stride_kk,
    stride_vz,
    stride_vh,
    stride_vk,
    stride_vn,
    stride_oz,
    stride_oh,
    stride_om,
    stride_on,
    Z,  # Batch size
    H,  # Head size
    N_CTX,  # N size
    HEAD_DIM: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    IS_CAUSAL: tl.constexpr,
):
    tl.static_assert(BLOCK_N <= HEAD_DIM)

    start_m = tl.program_id(0)
    off_hz = tl.program_id(1)

    off_z = off_hz // H  # B * H
    off_h = off_hz % H

    q_offset = off_z.to(tl.int64) * stride_qz + off_h.to(tl.int64) * stride_qh
    k_offset = off_z.to(tl.int64) * stride_kz + off_h.to(tl.int64) * stride_kh
    v_offset = off_z.to(tl.int64) * stride_vz + off_h.to(tl.int64) * stride_vh
    o_offset = off_z.to(tl.int64) * stride_oz + off_h.to(tl.int64) * stride_oh

    # block pointers
    Q_block_ptr = tl.make_block_ptr(
        base=Q + q_offset,
        shape=(N_CTX, HEAD_DIM),
        strides=(stride_qm, stride_qk),
        offsets=(start_m * BLOCK_M, 0),
        block_shape=(BLOCK_M, HEAD_DIM),
        order=(1, 0),
    )

    V_block_ptr = tl.make_block_ptr(
        base=V + v_offset,
        shape=(N_CTX, HEAD_DIM),
        strides=(stride_vk, stride_vn),
        offsets=(0, 0),
        block_shape=(BLOCK_N, HEAD_DIM),
        order=(1, 0),
    )

    K_block_ptr = tl.make_block_ptr(
        base=K + k_offset,
        shape=(HEAD_DIM, N_CTX),
        strides=(stride_kk, stride_kn),
        offsets=(0, 0),
        block_shape=(HEAD_DIM, BLOCK_N),
        order=(0, 1),
    )

    O_block_ptr = tl.make_block_ptr(
        base=Out + o_offset,
        shape=(N_CTX, HEAD_DIM),
        strides=(stride_om, stride_on),
        offsets=(start_m * BLOCK_M, 0),
        block_shape=(BLOCK_M, HEAD_DIM),
        order=(1, 0),
    )

    # initialize offsets
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)

    # initialize pointer to m and l
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32) + 1.0
    acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)

    # load scales
    qk_scale = sm_scale
    qk_scale *= 1.44269504  # 1/log(2)

    # load q: it will stay in SRAM throughout
    q = tl.load(Q_block_ptr)

    # range of values handled by this stage
    if IS_CAUSAL:
        lo, hi = 0, start_m * BLOCK_M
    else:
        lo, hi = 0, N_CTX

    K_block_ptr = tl.advance(K_block_ptr, (0, lo))
    V_block_ptr = tl.advance(V_block_ptr, (lo, 0))

    # loop over k, v and update accumulator
    for start_n in range(lo, hi, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        # -- compute qk ----
        k = tl.load(K_block_ptr)
        qk = tl.dot(q, k)

        m_ij = tl.maximum(m_i, tl.max(qk, 1) * qk_scale)
        qk = qk * qk_scale - m_ij[:, None]

        p = tl.math.exp2(qk)
        l_ij = tl.sum(p, 1)

        # -- update m_i and l_i
        alpha = tl.math.exp2(m_i - m_ij)
        l_i = l_i * alpha + l_ij

        # -- update output accumulator --
        acc = acc * alpha[:, None]

        # update acc
        v = tl.load(V_block_ptr)
        p = p.to(tl.float16)
        acc = tl.dot(p, v, acc)

        # update m_i and l_i
        m_i = m_ij
        V_block_ptr = tl.advance(V_block_ptr, (BLOCK_N, 0))
        K_block_ptr = tl.advance(K_block_ptr, (0, BLOCK_N))

    # epilogue
    m_i += tl.math.log2(l_i)
    acc = acc / l_i[:, None]
    m_ptrs = M + off_hz * N_CTX + offs_m
    tl.store(m_ptrs, m_i)
    tl.store(O_block_ptr, acc.to(Out.type.element_ty))


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

    M = torch.empty(
        (q.shape[0], q.shape[1], q.shape[2]), device=q.device, dtype=torch.float32
    )

    sm_scale = 1.0 / math.sqrt(q.shape[-1])

    flash_attn_kernel_v2[grid](
        q,
        k,
        v,
        sm_scale,
        M,
        o,  #
        q.stride(0),
        q.stride(1),
        q.stride(2),
        q.stride(3),  #
        k.stride(0),
        k.stride(1),
        k.stride(2),
        k.stride(3),  #
        v.stride(0),
        v.stride(1),
        v.stride(2),
        v.stride(3),  #
        o.stride(0),
        o.stride(1),
        o.stride(2),
        o.stride(3),  #
        q.shape[0],
        q.shape[1],  #
        N_CTX=q.shape[2],  #
        HEAD_DIM=HEAD_DIM_K,  #
        IS_CAUSAL=causal,  #
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
    test_op(1, 2, 1024, 64, False)
