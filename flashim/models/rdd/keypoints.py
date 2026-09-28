"""Sparse-feature helpers: soft keypoint detection, descriptor sampling and dual-softmax matching."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor


def simple_nms(scores: Tensor, radius: int) -> Tensor:
    """Non-maximum suppression with a ``(2 * radius + 1)`` window (SuperPoint-style, two suppression rounds)."""

    def max_pool(x: Tensor) -> Tensor:
        return F.max_pool2d(x, kernel_size=radius * 2 + 1, stride=1, padding=radius)

    zeros = torch.zeros_like(scores)
    max_mask = scores == max_pool(scores)
    for _ in range(2):
        suppressed = max_pool(max_mask.float()) > 0
        supp_scores = torch.where(suppressed, zeros, scores)
        new_max = supp_scores == max_pool(supp_scores)
        max_mask = max_mask | (new_max & ~suppressed)
    return torch.where(max_mask, scores, zeros)


def detect_keypoints(
    score_map: Tensor, top_k: int, radius: int = 2, temperature: float = 0.1
) -> tuple[Tensor, Tensor]:
    """Top-``top_k`` NMS maxima of ``score_map`` ``[B, 1, H, W]`` refined by a local soft-argmax.

    Returns keypoints ``[B, K, 2]`` as ``(x, y)`` in pixel-index coordinates (integer = pixel center) and
    the bilinearly sampled score at each keypoint ``[B, K]``.
    """
    b, _, h, w = score_map.shape
    nms = simple_nms(score_map, radius)
    nms[:, :, : radius + 1, :] = 0
    nms[:, :, :, : radius + 1] = 0
    nms[:, :, h - radius :, :] = 0
    nms[:, :, :, w - radius :] = 0
    index = torch.topk(nms.reshape(b, -1), top_k).indices  # [B, K]
    ys, xs = index // w, index % w

    size = 2 * radius + 1
    steps = torch.arange(size, device=score_map.device)
    padded = F.pad(score_map[:, 0], (radius,) * 4)
    batch = torch.arange(b, device=score_map.device)[:, None, None, None]
    patch = padded[batch, (ys[..., None] + steps)[..., :, None], (xs[..., None] + steps)[..., None, :]]
    patch = patch.reshape(b, top_k, size * size)
    weights = ((patch - patch.amax(-1, keepdim=True)) / temperature).exp()
    offsets = (steps - radius).float()
    grid_x, grid_y = offsets.repeat(size), offsets.repeat_interleave(size)
    residual = torch.stack((weights @ grid_x, weights @ grid_y), dim=-1) / weights.sum(-1, keepdim=True)
    keypoints = torch.stack((xs, ys), dim=-1).float() + residual

    norm = keypoints / keypoints.new_tensor([w - 1, h - 1]) * 2 - 1
    scores = F.grid_sample(score_map, norm[:, None], mode="bilinear", align_corners=True)[:, 0, 0]
    return keypoints, scores


def sample_descriptors(features: Tensor, keypoints: Tensor, height: int, width: int) -> Tensor:
    """Bicubic samples of ``features`` ``[B, C, h, w]`` at pixel-index ``keypoints`` of a ``height x width``
    image, L2-normalized: ``[B, K, C]``."""
    scale = torch.tensor([width, height], device=keypoints.device, dtype=keypoints.dtype)
    grid = ((2.0 * keypoints + 1.0) / scale - 1.0)[:, :, None].to(features.dtype)
    sampled = F.grid_sample(features, grid, mode="bicubic", align_corners=False)
    return F.normalize(sampled.permute(0, 2, 3, 1).squeeze(-2), dim=-1)


def dual_softmax_match(
    desc0: Tensor, desc1: Tensor, inv_temperature: float = 20.0, threshold: float = 0.01
) -> tuple[Tensor, Tensor, Tensor]:
    """Mutual-nearest-neighbour matches of the dual-softmax matrix; returns indices into ``desc0`` / ``desc1``
    and the match confidence."""
    sim = (desc0 @ desc1.t()) * inv_temperature
    prob = sim.softmax(dim=-2) * sim.softmax(dim=-1)
    mutual = (prob == prob.amax(dim=-1, keepdim=True)) & (prob == prob.amax(dim=-2, keepdim=True)) & (prob >= threshold)
    pairs = torch.nonzero(mutual)
    return pairs[:, 0], pairs[:, 1], prob[pairs[:, 0], pairs[:, 1]]
