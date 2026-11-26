"""Descriptor distance and ratio-test mutual-nearest-neighbour matching ops (ORB binary Hamming, SIFT L2)."""

from __future__ import annotations

import torch
from torch import Tensor

from flashim.ops._registry import Op

_POPCOUNT = torch.tensor([i.bit_count() for i in range(256)], dtype=torch.uint8)


def _hamming_distances_reference(a: Tensor, b: Tensor) -> Tensor:
    """``a`` ``[N, D]``, ``b`` ``[M, D]`` uint8 packed bits -> Hamming distances ``[N, M]`` (FP32)."""
    if a.shape[0] == 0 or b.shape[0] == 0:
        return a.new_empty(a.shape[0], b.shape[0], dtype=torch.float32)
    table = _POPCOUNT.to(a.device)
    chunk = 1024
    rows = [
        table[(a[i : i + chunk, None, :] ^ b[None, :, :]).long()].sum(-1, dtype=torch.int16)
        for i in range(0, a.shape[0], chunk)
    ]
    return torch.cat(rows).float()


hamming_distances = Op("hamming_distances", _hamming_distances_reference)


def _ratio_mutual_match_reference(dist: Tensor, ratio: float) -> tuple[Tensor, Tensor, Tensor]:
    """``dist`` ``[N, M]`` (non-negative, smaller = closer): mutual nearest neighbours passing ``d1 < ratio * d2``.
    Returns ``(ia, ib, conf)`` where ``conf = 1 - d1 / d2``."""
    if dist.shape[0] == 0 or dist.shape[1] < 2:
        empty = torch.empty(0, dtype=torch.long, device=dist.device)
        return empty, empty, dist.new_empty(0, dtype=torch.float32)
    best, index = dist.topk(2, dim=1, largest=False)
    ids = torch.arange(dist.shape[0], device=dist.device)
    keep = (best[:, 0] < ratio * best[:, 1]) & (dist.argmin(dim=0)[index[:, 0]] == ids)
    ia = keep.nonzero()[:, 0]
    return ia, index[ia, 0], 1.0 - best[ia, 0] / best[ia, 1]


ratio_mutual_match = Op("ratio_mutual_match", _ratio_mutual_match_reference)


def _hamming_ratio_mutual_match_reference(a: Tensor, b: Tensor, ratio: float) -> tuple[Tensor, Tensor, Tensor]:
    """``a`` ``[N, D]``, ``b`` ``[M, D]`` uint8 packed bits -> mutual nearest neighbours passing ``d1 < ratio * d2``
    under Hamming distance. Equivalent to ``ratio_mutual_match(hamming_distances(a, b), ratio)``."""
    if a.shape[0] == 0 or b.shape[0] < 2:
        empty = torch.empty(0, dtype=torch.long, device=a.device)
        return empty, empty, a.new_empty(0, dtype=torch.float32)
    return ratio_mutual_match(hamming_distances(a, b), ratio)


hamming_ratio_mutual_match = Op(
    "hamming_ratio_mutual_match", _hamming_ratio_mutual_match_reference, composite=True
)
