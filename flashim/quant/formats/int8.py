"""Symmetric INT8: values in [-127, 127], per-output-channel weight scales, per-tensor static
activation scales (``x ~= q * scale``)."""

from __future__ import annotations

import torch
from torch import Tensor

INT8_MAX = 127.0
_EPS = torch.finfo(torch.float32).eps


def safe_scale(scale: Tensor) -> Tensor:
    """FP32 scale with non-finite / vanishing entries replaced by ``eps`` (all-zero tensors)."""
    scale = scale.float()
    return torch.where(torch.isfinite(scale) & (scale > _EPS), scale, torch.full_like(scale, _EPS))


def weight_scales(weight: Tensor) -> Tensor:
    """Per-output-channel scale ``amax / 127`` of an ``[N, K]`` weight."""
    return safe_scale(weight.float().abs().amax(dim=1) / INT8_MAX)


def quantize_symmetric(value: Tensor, scale: Tensor) -> Tensor:
    """``round(value / scale)`` (half to even) saturated to +-127, as int8 (FP32 arithmetic)."""
    return (value.float() / scale).round().clamp(-INT8_MAX, INT8_MAX).to(torch.int8)
