"""2D non-maximum suppression over fixed-size pixel boxes."""

from __future__ import annotations

import torch
import torchvision.ops
from torch import Tensor

from flashim.ops._registry import Op


def _box_nms_2d_reference(
    scores_hw: Tensor, threshold: float, size: int = 4, iou_threshold: float = 0.1
) -> tuple[Tensor, Tensor]:
    """Greedy box NMS on a 2D pixel score map ``[H, W]`` with ``size x size`` boxes centered at each pixel.

    Filters pixels with ``scores_hw >= threshold``, forms boxes ``[y - size/2, x - size/2, y + size/2, x + size/2]``,
    and applies :func:`torchvision.ops.nms` at ``iou_threshold``. Returns ``(kpts_xy [K, 2], scores [K])`` (FP32).
    """
    ys, xs = torch.where(scores_hw >= threshold)
    if ys.numel() == 0:
        return scores_hw.new_zeros((0, 2), dtype=torch.float32), scores_hw.new_zeros((0,), dtype=torch.float32)
    vals = scores_hw[ys, xs].float()
    half = size / 2.0
    yf, xf = ys.float(), xs.float()
    boxes = torch.stack((yf - half, xf - half, yf + half, xf + half), dim=-1)
    keep = torchvision.ops.nms(boxes, vals, iou_threshold)
    kpts = torch.stack((xf[keep], yf[keep]), dim=-1)
    return kpts, vals[keep]


box_nms_2d = Op("box_nms_2d", _box_nms_2d_reference)
