"""Coordinate conventions and small tensor helpers for RoMaV2.

Warps are in normalized coordinates ``[-1, 1]`` with the ``align_corners=False``
pixel-center convention, stored last-dim ``(x, y)``.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor

from flashim.utils import grid_sample

_CONSTANTS: dict[tuple[torch.device, tuple[float, ...]], Tensor] = {}


def _constant(device: torch.device, values: tuple[float, ...]) -> Tensor:
    key = (device, values)
    if key not in _CONSTANTS:
        _CONSTANTS[key] = torch.tensor(values, device=device)
    return _CONSTANTS[key]


def imagenet_normalize(img: Tensor) -> Tensor:
    mean = _constant(img.device, (0.485, 0.456, 0.406))
    std = _constant(img.device, (0.229, 0.224, 0.225))
    return (img - mean[None, :, None, None]) / std[None, :, None, None]


def normalized_grid(batch: int, h: int, w: int, device: torch.device) -> Tensor:
    """``[B, H, W, 2]`` pixel-center coordinates in ``[-1, 1]``."""
    grid = torch.meshgrid(
        *[torch.linspace(-1 + 1 / n, 1 - 1 / n, n, device=device) for n in (batch, h, w)],
        indexing="ij",
    )
    return torch.stack((grid[2], grid[1]), dim=-1).reshape(batch, h, w, 2)


def to_pixel(x: Tensor, *, h: int, w: int) -> Tensor:
    return torch.stack(((x[..., 0] + 1) / 2 * w, (x[..., 1] + 1) / 2 * h), dim=-1)


def bhwc_interpolate(x: Tensor, size: tuple[int, int], mode: str = "bilinear", align_corners=None) -> Tensor:
    return F.interpolate(x.permute(0, 3, 1, 2), size=size, mode=mode, align_corners=align_corners).permute(
        0, 2, 3, 1
    )


def bhwc_grid_sample(x: Tensor, grid: Tensor, mode: str = "bilinear", align_corners=None) -> Tensor:
    return grid_sample(x.permute(0, 3, 1, 2), grid, mode=mode, align_corners=align_corners).permute(
        0, 2, 3, 1
    )


def precision_matrix(p: Tensor) -> Tensor:
    """``[B, H, W, 3]`` (p00, p10, p11) -> symmetric ``[B, H, W, 2, 2]``."""
    out = p.new_zeros(*p.shape[:-1], 2, 2)
    out[..., 0, 0] = p[..., 0]
    out[..., 1, 0] = p[..., 1]
    out[..., 0, 1] = p[..., 1]
    out[..., 1, 1] = p[..., 2]
    return out
