"""RoPE tables shared by the DINOv3 descriptor and the matcher ViT (both built from
:class:`flashim.layers.TransformerBlock`, whose parameter names match the RoMaV2 checkpoint)."""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn


class RopeEmbedding(nn.Module):
    """Axial RoPE tables over an ``H x W`` grid (DINOv3 ``normalize_coords="separate"``).

    ``compute_dtype`` is the dtype the tables are built in; ``periods`` is stored in
    the checkpoint (BF16) and promoted as upstream does.
    """

    def __init__(self, head_dim: int, *, base: float, compute_dtype: torch.dtype) -> None:
        super().__init__()
        self.compute_dtype = compute_dtype
        periods = base ** (2 * torch.arange(head_dim // 4, dtype=compute_dtype) / (head_dim // 2))
        self.register_buffer("periods", periods)

    def forward(self, h: int, w: int) -> tuple[Tensor, Tensor]:
        dd = {"device": self.periods.device, "dtype": self.compute_dtype}
        coords_h = torch.arange(0.5, h, **dd) / h
        coords_w = torch.arange(0.5, w, **dd) / w
        coords = torch.stack(torch.meshgrid(coords_h, coords_w, indexing="ij"), dim=-1).flatten(0, 1)
        coords = 2.0 * coords - 1.0
        angles = 2 * math.pi * coords[:, :, None] / self.periods[None, None, :]
        angles = angles.flatten(1, 2).tile(2)
        return torch.sin(angles), torch.cos(angles)
