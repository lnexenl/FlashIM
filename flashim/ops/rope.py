"""Rotary position embedding applied to Q and K."""

from __future__ import annotations

import torch
from torch import Tensor

from flashim.ops._registry import DeviceSpec, Op
from flashim.platforms import Feature


def _rope_qk_reference(q: Tensor, k: Tensor, sin: Tensor, cos: Tensor) -> tuple[Tensor, Tensor]:
    """Apply half-split RoPE to ``[B, H, N, D]`` q/k with ``[N', D]`` tables.

    q/k are cast to the table dtype, rotated, and cast back. The first
    ``N - N'`` tokens (e.g. CLS/storage tokens) are left untouched.
    """
    q_dtype, k_dtype = q.dtype, k.dtype
    q, k = q.to(sin.dtype), k.to(sin.dtype)
    prefix = q.shape[-2] - sin.shape[-2]
    if prefix < 0:
        raise ValueError(f"RoPE table length {sin.shape[-2]} exceeds token count {q.shape[-2]}")

    def apply(value: Tensor) -> Tensor:
        head, tail = value[:, :, :prefix, :], value[:, :, prefix:, :]
        first, second = tail.chunk(2, dim=-1)
        tail = tail * cos + torch.cat((-second, first), dim=-1) * sin
        return torch.cat((head, tail), dim=-2)

    return apply(q).to(q_dtype), apply(k).to(k_dtype)


rope_qk = Op("rope_qk", _rope_qk_reference)

_FLOATS = (torch.float16, torch.bfloat16, torch.float32)


def _triton_eligible(q: Tensor, k: Tensor, sin: Tensor, cos: Tensor) -> bool:
    if torch.is_grad_enabled() or q.ndim != 4 or k.shape != q.shape:
        return False
    if sin.ndim != 2 or sin.shape != cos.shape or sin.dtype != cos.dtype:
        return False
    if q.dtype not in _FLOATS or k.dtype not in _FLOATS or sin.dtype not in _FLOATS:
        return False
    # Low-precision tables require q/k already in the table dtype (no promotion).
    if sin.dtype != torch.float32 and (q.dtype != sin.dtype or k.dtype != sin.dtype):
        return False
    head_dim = q.shape[-1]
    return 0 < sin.shape[0] <= q.shape[2] and head_dim == sin.shape[1] and head_dim % 2 == 0


@rope_qk.register("triton", device=DeviceSpec("cuda"), requires=[Feature.TRITON], when=_triton_eligible)
def _rope_qk_triton(q: Tensor, k: Tensor, sin: Tensor, cos: Tensor) -> tuple[Tensor, Tensor]:
    from flashim.ops._triton.rope import rope_qk as kernel

    return kernel(q, k, sin, cos)
