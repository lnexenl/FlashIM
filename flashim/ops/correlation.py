"""Local correlation between a feature map and a warped window of another."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor

from flashim.ops._registry import Op


def local_window(radius: int, h: int, w: int, device: torch.device) -> Tensor:
    """``[1, K, 2]`` (x, y) offsets of a (2r+1)^2 window, one feature pixel apart."""
    ys, xs = torch.meshgrid(
        torch.linspace(-2 * radius / h, 2 * radius / h, 2 * radius + 1, device=device),
        torch.linspace(-2 * radius / w, 2 * radius / w, 2 * radius + 1, device=device),
        indexing="ij",
    )
    return torch.stack((xs, ys), dim=-1).reshape(1, (2 * radius + 1) ** 2, 2)


def _local_corr_reference(feature0: Tensor, feature1: Tensor, warp: Tensor, radius: int) -> Tensor:
    """Correlate ``feature0 [B,C,H,W]`` with ``feature1`` sampled around ``warp [B,H,W,2]``.

    Returns ``[B, K, H, W]`` with ``K = (2r+1)^2``; features are scaled by
    ``1/sqrt(C)``; sampling is bilinear, zeros padding, ``align_corners=False``.
    """
    batch, channels, h, w = feature0.shape
    window = local_window(radius, h, w, feature0.device)
    k = window.shape[1]
    corr = torch.empty((batch, k, h, w), device=feature0.device, dtype=feature0.dtype)
    for b in range(batch):
        coords = (warp[b, :, :, None] + window[:, None, None]).reshape(1, h, w * k, 2)
        sampled = F.grid_sample(
            feature1[b : b + 1], coords, padding_mode="zeros", align_corners=False, mode="bilinear"
        ).reshape(channels, h, w, k)
        corr[b] = (feature0[b, ..., None] / channels**0.5 * sampled).sum(dim=0).permute(2, 0, 1)
    return corr


local_corr = Op("local_corr", _local_corr_reference)
