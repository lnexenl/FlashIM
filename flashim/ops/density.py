"""Gaussian kernel density of a point set, as used by RoMaV2's balanced match sampling.

``density[i] = sum_j exp(-|x_i - x_j|^2 / (2 std^2))`` (the ``j = i`` term included).
"""

from __future__ import annotations

import torch
from torch import Tensor

from flashim.ops._registry import Op


def _kde_density_reference(x: Tensor, std: float = 0.1) -> Tensor:
    """RoMaV2's formulation: FP16 points, full ``N x N`` ``cdist`` matrix (O(N^2) memory), FP16 result."""
    x = x.half()
    return (-(torch.cdist(x, x) ** 2) / (2 * std**2)).exp().sum(dim=-1)


kde_density = Op("kde_density", _kde_density_reference)
