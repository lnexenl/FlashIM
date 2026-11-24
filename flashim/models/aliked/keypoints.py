"""ALIKED keypoint decoding: soft-NMS, threshold/top-k selection, sub-pixel soft-argmax refinement (DKD),
and mutual-nearest-neighbour descriptor matching."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor

from flashim.utils import device_constant, grid_sample


def simple_nms(scores: Tensor, radius: int) -> Tensor:
    """Two-round soft non-maximum suppression with a ``(2 * radius + 1)`` window."""

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


def _select_indices(nms_scores: Tensor, scores_map: Tensor, top_k: int, scores_th: float, n_limit: int) -> list[Tensor]:
    """Indices into the flattened ``H * W`` score map of the selected keypoints, one tensor per image."""
    b = nms_scores.shape[0]
    flat_nms = nms_scores.reshape(b, -1)
    flat_scores = scores_map.reshape(b, -1)
    if top_k > 0:
        return list(torch.topk(flat_nms, top_k).indices)
    if scores_th > 0:
        masks = flat_nms > scores_th
        # Branchless fallback (matches upstream: falls back to per-image mean when no pixel passes scores_th)
        # without a blocking host sync on `masks.sum() == 0`.
        masks = torch.where(masks.any(), masks, flat_nms > flat_scores.mean(dim=1, keepdim=True))
    else:
        masks = flat_nms > flat_scores.mean(dim=1, keepdim=True)
    indices = []
    for mask, scores in zip(masks, flat_scores):
        idx = mask.nonzero(as_tuple=True)[0]
        if len(idx) > n_limit:
            order = scores[idx].argsort(descending=True)[:n_limit]
            idx = idx[order]
        indices.append(idx)
    return indices


def detect_keypoints(
    score_map: Tensor, radius: int = 2, temperature: float = 0.1, top_k: int = -1, scores_th: float = 0.2, n_limit: int = 5000
) -> tuple[list[Tensor], list[Tensor], list[Tensor]]:
    """NMS maxima of ``score_map`` ``[B, 1, H, W]`` refined by a local soft-argmax (ALIKED's DKD).

    ``top_k > 0`` selects the top-``top_k`` maxima; otherwise every maximum above ``scores_th`` is kept
    (capped at ``n_limit``, highest-scoring first). Returns, per image: keypoints in normalized ``(x, y)``
    ``[-1, 1]`` coordinates, the bilinearly sampled score at each keypoint, and its score dispersity (the
    soft-argmax spread, a localization-uncertainty proxy).
    """
    _, _, h, w = score_map.shape
    nms_scores = simple_nms(score_map.detach(), radius)
    nms_scores[:, :, :radius, :] = 0
    nms_scores[:, :, :, :radius] = 0
    nms_scores[:, :, h - radius :, :] = 0
    nms_scores[:, :, :, w - radius :] = 0
    indices = _select_indices(nms_scores, score_map.detach(), top_k, scores_th, n_limit)

    wh = device_constant([w - 1, h - 1], score_map)
    offs = list(range(-radius, radius + 1))
    # Precomputed patch offsets in row-major (dy, dx) order — avoids both the `[:, [1, 0]]` host sync and
    # materializing a 109 MB dense `F.unfold(score_map, 5, padding=2)` buffer for <= 4096 keypoints.
    grid = device_constant([(dx, dy) for dy in offs for dx in offs], score_map, torch.float32)  # [K^2, 2] (x, y)
    oy = device_constant([dy for dy in offs for _ in offs], score_map, torch.long)
    ox = device_constant([dx for _ in offs for dx in offs], score_map, torch.long)

    keypoints, kptscores, dispersities = [], [], []
    for b_idx, idx in enumerate(indices):
        ix = idx % w
        iy = idx // w
        ys = iy[:, None] + oy[None, :]
        xs = ix[:, None] + ox[None, :]
        inside = (ys >= 0) & (ys < h) & (xs >= 0) & (xs < w)
        patch = score_map[b_idx, 0, ys.clamp(0, h - 1), xs.clamp(0, w - 1)]
        patch = torch.where(inside, patch, torch.zeros_like(patch))  # [N, kernel**2], keeps score_map's grad
        nms_xy = torch.stack((ix, iy), dim=1).float()  # (x, y)

        weight = ((patch - patch.max(dim=1, keepdim=True).values.detach()) / temperature).exp()
        residual = (weight @ grid) / weight.sum(dim=1, keepdim=True)  # soft-argmax, [N, 2]
        dist2 = torch.norm((grid[None] - residual[:, None]) / radius, dim=-1) ** 2
        dispersity = (weight * dist2).sum(dim=1) / weight.sum(dim=1)

        xy = (nms_xy + residual) / wh * 2 - 1
        score = grid_sample(score_map[b_idx : b_idx + 1], xy.view(1, 1, -1, 2), mode="bilinear", align_corners=True)[0, 0, 0]

        keypoints.append(xy)
        kptscores.append(score)
        dispersities.append(dispersity)
    return keypoints, kptscores, dispersities


def mutual_nn_match(desc0: Tensor, desc1: Tensor, threshold: float = 0.75) -> tuple[Tensor, Tensor, Tensor]:
    """Mutual-nearest-neighbour cosine matches above ``threshold`` (``desc0``/``desc1``: ``[N, C]``,
    L2-normalized). Equivalent to ALIKED's ``demo_pair.py`` ``mnn_mather``, without its zero-then-argmax
    quirk (a similarity below ``threshold`` never changes which index is the nearest neighbour here)."""
    if desc0.shape[0] == 0 or desc1.shape[0] == 0:
        empty = torch.zeros(0, dtype=torch.long, device=desc0.device)
        return empty, empty, desc0.new_zeros(0)
    sim = desc0 @ desc1.t()
    nn01, nn10 = sim.argmax(dim=1), sim.argmax(dim=0)
    best = sim.gather(1, nn01[:, None])[:, 0]
    ids0 = torch.arange(sim.shape[0], device=sim.device)
    idx0 = ((ids0 == nn10[nn01]) & (best >= threshold)).nonzero()[:, 0]  # single host sync (match count)
    return idx0, nn01[idx0], best[idx0]
