import torch
from torch import Tensor


def relu_f32(x: Tensor, y: Tensor) -> Tensor:
    return torch.ops._C.relu_f32.default(x, y)


def relu_f32x4(x: Tensor, y: Tensor) -> Tensor:
    return torch.ops._C.relu_f32x4.default(x, y)


def elu_f32(x: Tensor, y: Tensor) -> Tensor:
    return torch.ops._C.elu_f32.default(x, y)


def elu_f16(x: Tensor, y: Tensor) -> Tensor:
    return torch.ops._C.elu_f16.default(x, y)


def sgemm_naive_f32(a: Tensor, b: Tensor, c: Tensor) -> Tensor:
    """Performs a GEMM operation: A @ B"""
    return torch.ops._C.sgemm_naive_f32.default(a, b, c)


def sgemm_shared_f32(a: Tensor, b: Tensor, c: Tensor) -> Tensor:
    """Performs a GEMM operation: A @ B"""
    return torch.ops._C.sgemm_shared_f32.default(a, b, c)


def sgemm_t_8x8_shared_f32x4(a: Tensor, b: Tensor, c: Tensor) -> Tensor:
    """Performs a GEMM operation: A @ B"""
    return torch.ops._C.sgemm_t_8x8_shared_f32x4.default(a, b, c)


def flash_attn_v2_cute_v1(Q: Tensor, K: Tensor, V: Tensor, causal: bool) -> Tensor:
    O = torch.zeros_like(Q)
    torch.ops._C.flash_attn_v2_cute_v1(Q, K, V, O)
    return O


def flash_attn_v2_cute_v2(Q: Tensor, K: Tensor, V: Tensor, causal: bool) -> Tensor:
    O = torch.zeros_like(Q)
    torch.ops._C.flash_attn_v2_cute_v2(Q, K, V, O)
    return O
