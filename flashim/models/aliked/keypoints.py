"""ALIKED keypoint decoding: soft-NMS, threshold/top-k selection, sub-pixel soft-argmax refinement (DKD),
and mutual-nearest-neighbour descriptor matching."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor

from flashim.utils import device_constant


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
        if masks.sum() == 0:  # matches upstream: falls back to the per-image mean threshold for every image
            masks = flat_nms > flat_scores.mean(dim=1, keepdim=True)
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
    kernel = 2 * radius + 1
    offsets = torch.linspace(-radius, radius, kernel, device=score_map.device)
    grid = torch.stack(torch.meshgrid(offsets, offsets, indexing="ij")).view(2, -1).t()[:, [1, 0]]  # (kernel**2, 2) (x, y)
    patches = F.unfold(score_map, kernel_size=kernel, padding=radius)  # [B, kernel**2, H*W], keeps score_map's grad

    keypoints, kptscores, dispersities = [], [], []
    for b_idx, idx in enumerate(indices):
        patch = patches[b_idx].t()[idx]  # [N, kernel**2]
        nms_xy = torch.stack((idx % w, idx // w), dim=1).float()  # (x, y)

        weight = ((patch - patch.max(dim=1, keepdim=True).values.detach()) / temperature).exp()
        residual = (weight @ grid) / weight.sum(dim=1, keepdim=True)  # soft-argmax, [N, 2]
        dist2 = torch.norm((grid[None] - residual[:, None]) / radius, dim=-1) ** 2
        dispersity = (weight * dist2).sum(dim=1) / weight.sum(dim=1)

        xy = (nms_xy + residual) / wh * 2 - 1
        score = F.grid_sample(score_map[b_idx : b_idx + 1], xy.view(1, 1, -1, 2), mode="bilinear", align_corners=True)[0, 0, 0]

        keypoints.append(xy)
        kptscores.append(score)
        dispersities.append(dispersity)
    return keypoints, kptscores, dispersities


def mutual_nn_match(desc0: Tensor, desc1: Tensor, threshold: float = 0.75) -> tuple[Tensor, Tensor, Tensor]:
    """Mutual-nearest-neighbour cosine matches above ``threshold`` (``desc0``/``desc1``: ``[N, C]``,
    L2-normalized). Equivalent to ALIKED's ``demo_pair.py`` ``mnn_mather``, without its zero-then-argmax
    quirk (a similarity below ``threshold`` never changes which index is the nearest neighbour here)."""
    sim = desc0 @ desc1.t()
    nn01 = sim.argmax(dim=1)
    nn10 = sim.argmax(dim=0)
    ids0 = torch.arange(sim.shape[0], device=sim.device)
    idx1 = nn01[ids0]
    mutual = ids0 == nn10[idx1]
    valid = mutual & (sim[ids0, idx1] >= threshold)
    idx0 = ids0[valid]
    return idx0, idx1[valid], sim[idx0, idx1[valid]]
