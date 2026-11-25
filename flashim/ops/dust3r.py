"""DUSt3R-specific op: fused DPT raw 4-channel output -> FP32 3D pointmap (exp depth) + confidence."""

from __future__ import annotations

import torch
from torch import Tensor

from flashim.ops._registry import Op


def _dust3r_pts3d_conf_reference(raw: Tensor, conf_offset: float = 1.0) -> tuple[Tensor, Tensor]:
    """Convert raw DPT head output ``[B, 4, H, W]`` into ``(pts3d [B, H, W, 3], conf [B, H, W])`` in FP32
    using DUSt3R's exponential depth parameterization (``unit(xyz) * expm1(||xyz||)``) and exponential confidence
    (``conf_offset + exp(c)``)."""
    fmap = raw.permute(0, 2, 3, 1).float()
    xyz = fmap[..., 0:3]
    d = xyz.norm(dim=-1, keepdim=True)
    pts3d = (xyz / d.clamp(min=1e-8)) * torch.expm1(d)
    conf = conf_offset + fmap[..., 3].exp()
    return pts3d, conf


dust3r_pts3d_conf = Op("dust3r_pts3d_conf", _dust3r_pts3d_conf_reference)
