"""RaCo keypoint selection: NMS + top-k of the detector logits, soft-argmax refinement, ranker sampling."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor

from flashim.utils import device_constant


def detect_candidates(logits: Tensor, count: int, radius: int, temperature: float) -> Tensor:
    """Top-``count`` local maxima of ``logits`` ``[B, 1, H, W]`` (window ``radius``), refined by a softmax-weighted
    offset over that window. Returns ``(x, y)`` keypoints ``[B, count, 2]`` in pixel-index coordinates.

    The spatial softmax over the image is strictly monotonic, so maxima and their order are selected on the
    logits directly; the refinement uses the same logits (zero-padded window), as in the reference."""
    b, _, _, w = logits.shape
    nms = F.max_pool2d(logits, radius, stride=1, padding=radius // 2)
    scores = torch.where(logits == nms, logits, torch.full_like(logits, -torch.inf)).flatten(1)
    index = scores.topk(count, dim=1).indices  # [B, count]
    keypoints = torch.stack((index % w, index // w), dim=-1).to(logits.dtype)

    patches = F.unfold(logits, kernel_size=radius, padding=radius // 2)  # [B, radius**2, H*W]
    patches = patches.gather(2, index[:, None].expand(b, radius**2, -1))
    probability = F.softmax(patches / temperature, dim=1)
    steps = torch.linspace(-(radius - 1) / 2, (radius - 1) / 2, radius, device=logits.device)
    grid_y, grid_x = torch.meshgrid(steps, steps, indexing="ij")
    offsets = torch.stack((grid_x, grid_y), dim=-1).reshape(radius**2, 2)
    return keypoints + torch.einsum("bkn,kd->bnd", probability, offsets)


def sample(feature_map: Tensor, keypoints: Tensor) -> Tensor:
    """Bilinear samples (border-clamped, ``align_corners=True``) of a single-channel ``[B, 1, H, W]`` map at
    pixel-index ``keypoints`` ``[B, N, 2]`` -> ``[B, N]``."""
    h, w = feature_map.shape[-2:]
    scale = device_constant([w - 1, h - 1], keypoints)
    grid = (2 * keypoints / scale - 1).unsqueeze(2)
    sampled = F.grid_sample(feature_map, grid, mode="bilinear", padding_mode="border", align_corners=True)
    return sampled[:, 0, :, 0]
