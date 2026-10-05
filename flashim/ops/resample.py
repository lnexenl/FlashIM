"""Resampling fused with what consumes it: bilinear upsampling (``align_corners=False``) of a coarse map joined to a
full-resolution one by concatenation or addition, in one pass (multi-scale decoders: DeDoDe / DaD / RIPE)."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor

from flashim.ops._registry import Op


def _upsample_concat_reference(features: Tensor, coarse: Tensor) -> Tensor:
    """``cat((features, upsample(coarse)), 1)``: ``coarse`` ``[B, Cc, h, w]`` upsampled bilinearly (``align_corners=False``,
    in FP32) to ``features``' ``[B, Cf, H, W]`` size and cast to its dtype -> ``[B, Cf + Cc, H, W]``."""
    upsampled = F.interpolate(coarse.float(), size=features.shape[-2:], mode="bilinear", align_corners=False)
    return torch.cat((features, upsampled.to(features.dtype)), dim=1)


upsample_concat = Op("upsample_concat", _upsample_concat_reference)


def _upsample_add_reference(coarse: Tensor, delta: Tensor) -> Tensor:
    """``upsample(coarse) + delta`` in FP32: ``coarse`` ``[B, C, h, w]`` upsampled bilinearly (``align_corners=False``) to
    ``delta``'s ``[B, C, H, W]`` size -> FP32."""
    upsampled = F.interpolate(coarse.float(), size=delta.shape[-2:], mode="bilinear", align_corners=False)
    return upsampled + delta.float()


upsample_add = Op("upsample_add", _upsample_add_reference)
