"""Fused elementwise activation ops."""

from __future__ import annotations

import torch.nn.functional as F
from torch import Tensor

from flashim.ops._registry import Op


def _relu_affine_reference(x: Tensor, scale: Tensor, shift: Tensor) -> Tensor:
    """Channel-wise ``relu(x) * scale + shift`` on ``[B, C, H, W]`` (eval-mode ``BatchNorm2d(ReLU(x))``)."""
    y = F.relu(x.float()) * scale.float().view(1, -1, 1, 1) + shift.float().view(1, -1, 1, 1)
    return y.to(x.dtype)


relu_affine = Op("relu_affine", _relu_affine_reference)
