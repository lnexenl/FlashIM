"""Gaussian kernel density of a point set, as used by RoMaV2's balanced match sampling.

``density[i] = sum_j exp(-|x_i - x_j|^2 / (2 std^2))`` (the ``j = i`` term included).
"""

from __future__ import annotations

import torch
from torch import Tensor

from flashim.ops._registry import DeviceSpec, Op
from flashim.platforms import Feature


def _kde_density_reference(x: Tensor, std: float = 0.1) -> Tensor:
    """RoMaV2's formulation: FP16 points, full ``N x N`` ``cdist`` matrix (O(N^2) memory), FP16 result."""
    x = x.half()
    return (-(torch.cdist(x, x) ** 2) / (2 * std**2)).exp().sum(dim=-1)


kde_density = Op("kde_density", _kde_density_reference)


def _eligible(x: Tensor, std: float = 0.1) -> bool:
    return x.ndim == 2 and 1 <= x.shape[1] <= 8 and x.shape[0] > 0


@kde_density.register("triton", device=DeviceSpec("cuda"), requires=[Feature.TRITON], when=_eligible)
def _kde_density_triton(x: Tensor, std: float = 0.1) -> Tensor:
    """Tiled FP32 evaluation with no ``N x N`` intermediate; returns FP16 like the reference."""
    from flashim.ops._triton.density import kde_density as kernel

    return kernel(x, std).half()
