"""LightGlue-specific fused ops (the INT8 transformer layer's glue, :mod:`flashim.models.lightglue.int8_glue`).

Both images' tokens are one ``[2 N, C]`` stream (image 0's rows, then image 1's):

``lightglue_qkv_rope`` -- the self block's ``Wqkv`` (INT8) with LightGlue's interleaved RoPE on q / k, scattered into
``[3, 2, H, N, D]`` (q, k, v as contiguous ``[B, H, N, D]``: what ``view / permute`` and ``rope_qk_interleaved``
produce, without the permuted BF16 copy and the separate RoPE pass).

``lightglue_swapped_linear`` -- the cross block's ``to_qk`` / ``to_v`` (INT8), also written with the two images swapped
(``y.view(2, N, C).flip(0)``: the keys / values the other image attends to), without the ``flip`` copy.
"""

from __future__ import annotations

import torch
from torch import Tensor

from flashim.ops._registry import Op
from flashim.ops.attention import _rope_qk_interleaved_reference
from flashim.ops.gemm import _int8_linear_reference


def _lightglue_qkv_rope_reference(
    a: Tensor, weight: Tensor, a_scale: Tensor, w_scale: Tensor, bias: Tensor | None, cos: Tensor, sin: Tensor,
    heads: int,
) -> Tensor:
    """``a`` INT8 ``[B * N, C]``, ``weight`` ``[3 C, C]``, ``cos`` / ``sin`` the rotary tables ``[B, 1, N, D]`` ->
    ``[3, B, H, N, D]`` BF16."""
    batch, tokens, dim = cos.shape[0], cos.shape[2], cos.shape[3]
    qkv = _int8_linear_reference(a, weight, a_scale, w_scale, bias).view(batch, tokens, 3, heads, dim)
    q, k, v = qkv.permute(2, 0, 3, 1, 4)
    q, k = _rope_qk_interleaved_reference(q, k, cos, sin)
    return torch.stack((q, k, v))


lightglue_qkv_rope = Op("lightglue_qkv_rope", _lightglue_qkv_rope_reference)


def _lightglue_swapped_linear_reference(
    a: Tensor, weight: Tensor, a_scale: Tensor, w_scale: Tensor, bias: Tensor | None, *, keep: bool = True
) -> tuple[Tensor | None, Tensor]:
    """``a`` INT8 ``[2 N, K]`` -> ``(y, swapped)``, ``y = int8_linear(a, ...)`` ``[2 N, M]`` BF16 (``None`` unless
    ``keep``) and ``swapped = y.view(2, N, M).flip(0).reshape(2 N, M)``."""
    y = _int8_linear_reference(a, weight, a_scale, w_scale, bias)
    swapped = y.view(2, -1, y.shape[-1]).flip(0).reshape(y.shape)
    return (y if keep else None), swapped


lightglue_swapped_linear = Op("lightglue_swapped_linear", _lightglue_swapped_linear_reference)
