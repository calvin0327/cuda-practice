import torch
from torch import Tensor


def sgemm_naive_f32(a: Tensor, b: Tensor, c: Tensor) -> Tensor:
    """Performs a GEMM operation: A @ B"""
    return torch.ops._C.sgemm_naive_f32.default(a, b, c)
