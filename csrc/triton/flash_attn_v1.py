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
    # 定义三个维度的作用
    b_id = tl.program_id(0)
    h_id = tl.program_id(1)
    q_id = tl.program_id(2)

    # q 和 o 的地址定位
    q_offsets = q_id * BLOCK_M + tl.arange(0, BLOCK_M)
    q_mask = q_offsets < N_CTX

    q_ptrs = (
        Q_ptr
        + b_id * stride_qb
        + h_id * stride_qh
        + q_offsets[:, None] * stride_qm
        + tl.arange(0, HEAD_DIM)[None, :] * stride_qk
    )
    o_ptrs = (
        O_ptr
        + b_id * stride_ob
        + h_id * stride_oh
        + q_offsets[:, None] * stride_om
        + tl.arange(0, HEAD_DIM)[None, :] * stride_ok
    )

    # 加载 q to sram
    q = tl.load(q_ptrs, mask=q_mask[:, None], other=0)

    # 定义全局变量 max，l
    m_i = tl.full([BLOCK_M], float("inf"), dtype=tl.float32)
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    o_i = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)

    # 计算 kv 循环次数
    j = tl.cdiv(N_CTX, BLOCK_N)

    for j in range(j):
        k_offsets = j * BLOCK_N + tl.arange(0, BLOCK_N)
        mask = k_offsets < N_CTX

        k_ptrs = (
            K_ptr
            + b_id * stride_kb
            + h_id * stride_kh
            + k_offsets[:, None] * stride_kn
            + tl.range(0, HEAD_DIM)[None, :] * stride_kk
        )
        v_ptrs = (
            V_ptr
            + b_id * stride_vb
            + h_id * stride_vh
            + k_offsets[:, None] * stride_vn
            + tl.range(0, HEAD_DIM)[None, :] * stride_vk
        )
        k = tl.load(k_ptrs, mask=mask[:, None], other=0.0)
        v = tl.load(v_ptrs, mask=mask[:, None], other=0.0)

        # shape: [BLOCK_M, BLOCK_N]
        s = tl.dot(q, tl.trans(k)) * sm_scale

        # 计算当前最大值
        m_new = tl.maximum(m_i, tl.max(s, axis=1))

        # 计算差值
        alpha = tl.exp(m_i - m_new)

        # 计算分母
        l_new = l_i * alpha + tl.sum(tl.exp(s - m_new[:, None]), axis=1)

        # 计算分子
        p = tl.exp(s - m_new)
        pv = tl.dot(p.to(v.dtype), v)

        # 累积
        o_i = alpha[:, None] * o_i + pv

        l_i = l_new
        m_i = m_new

    # 归一化
    o_i = o_i / l_i[:, None]

    # store o to HBM
    tl.store(o_ptrs, o_i.to(O_ptr.dtype.element_ty), mask=q_mask[:, None])


def flash_attn_v1(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
) -> torch.Tensor:

    b, h, n, d = q.shape
    assert q.dtype == k.dtype == v.dtype

    scale = 1.0 / tl.sqrt(d)

    o = torch.empty_like(q)

    BLOCK_SIZE_M = 64
    BLOCK_SIZE_N = 64

    num_q_blocks = triton.cdiv(n, BLOCK_SIZE_M)
    grid = (b, h, num_q_blocks)
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
        N_CTX=n,
        HEAD_DIM=d,
        BLOCK_M=BLOCK_SIZE_M,
        BLOCK_N=BLOCK_SIZE_N,
    )
    return o
