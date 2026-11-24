"""Non-maximum suppression ops: strict window NMS (DISK) and two-round soft NMS (SuperPoint, ALIKE, ALIKED)."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor

from flashim.ops._registry import Op


def _strict_nms_mask_reference(heatmap: Tensor, window: int = 5, threshold: float | None = None) -> Tensor:
    """Boolean mask of pixels in ``heatmap`` ``[B, 1, H, W]`` strictly greater than every other pixel in their
    ``window x window`` neighbourhood (border of ``window // 2`` excluded), optionally also ``> threshold``."""
    r = window // 2
    h, w = heatmap.shape[-2:]
    if h <= 2 * r or w <= 2 * r:
        return torch.zeros_like(heatmap, dtype=torch.bool)
    centre = heatmap[..., r : h - r, r : w - r]
    keep = torch.ones_like(centre, dtype=torch.bool) if threshold is None else (centre > threshold)
    for dy in range(-r, r + 1):
        for dx in range(-r, r + 1):
            if dy or dx:
                keep &= centre > heatmap[..., r + dy : h - r + dy, r + dx : w - r + dx]
    mask = torch.zeros_like(heatmap, dtype=torch.bool)
    mask[..., r : h - r, r : w - r] = keep
    return mask


strict_nms_mask = Op("strict_nms_mask", _strict_nms_mask_reference)


def _simple_nms_reference(scores: Tensor, radius: int) -> Tensor:
    """SuperPoint / ALIKE / ALIKED two-round fast NMS over ``[B, H, W]`` or ``[B, 1, H, W]`` score maps."""
    def max_pool(x: Tensor) -> Tensor:
        return F.max_pool2d(x, kernel_size=radius * 2 + 1, stride=1, padding=radius)

    zeros = torch.zeros_like(scores)
    max_mask = scores == max_pool(scores)
    for _ in range(2):
        supp_mask = max_pool(max_mask.float()) > 0
        supp_scores = torch.where(supp_mask, zeros, scores)
        new_max_mask = supp_scores == max_pool(supp_scores)
        max_mask = max_mask | (new_max_mask & (~supp_mask))
    return torch.where(max_mask, scores, zeros)


simple_nms = Op("simple_nms", _simple_nms_reference)
