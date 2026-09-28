"""RDD's deformable transformer: level/position embeddings with cached per-shape geometry around a
:class:`~flashim.layers.deformable.DeformableEncoder`."""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn

from flashim.layers.deformable import DeformableEncoder


def sine_position_embedding(height: int, width: int, num_pos_feats: int, device: torch.device) -> Tensor:
    """Normalized 2-D sine embedding ``[2 * num_pos_feats, H, W]`` (DETR ``PositionEmbeddingSine``, no padding)."""
    eps, scale, temperature = 1e-6, 2 * math.pi, 10000
    y = torch.arange(1, height + 1, dtype=torch.float32, device=device)[:, None].expand(height, width)
    x = torch.arange(1, width + 1, dtype=torch.float32, device=device)[None, :].expand(height, width)
    y = (y - 0.5) / (height + eps) * scale
    x = (x - 0.5) / (width + eps) * scale
    dim_t = torch.arange(num_pos_feats, dtype=torch.float32, device=device)
    dim_t = temperature ** (2 * (dim_t // 2) / num_pos_feats)
    pos_x, pos_y = x[..., None] / dim_t, y[..., None] / dim_t
    pos_x = torch.stack((pos_x[..., 0::2].sin(), pos_x[..., 1::2].cos()), dim=3).flatten(2)
    pos_y = torch.stack((pos_y[..., 0::2].sin(), pos_y[..., 1::2].cos()), dim=3).flatten(2)
    return torch.cat((pos_y, pos_x), dim=2).permute(2, 0, 1)


def reference_points(shapes: tuple[tuple[int, int], ...], device: torch.device) -> Tensor:
    """Normalized ``(x, y)`` pixel centers of every token, ``[S, 2]``."""
    points = []
    for h, w in shapes:
        ys, xs = torch.meshgrid(
            torch.linspace(0.5, h - 0.5, h, dtype=torch.float32, device=device),
            torch.linspace(0.5, w - 0.5, w, dtype=torch.float32, device=device),
            indexing="ij",
        )
        points.append(torch.stack((xs.reshape(-1) / w, ys.reshape(-1) / h), -1))
    return torch.cat(points)


class DeformableTransformer(nn.Module):
    """Flattens the feature levels, adds position + level embeddings and runs the encoder."""

    def __init__(self, d_model: int, num_heads: int, num_layers: int, d_ffn: int, num_levels: int, num_points: int) -> None:
        super().__init__()
        self.d_model = d_model
        self.encoder = DeformableEncoder(
            num_layers, d_model=d_model, d_ffn=d_ffn, n_levels=num_levels, n_heads=num_heads, n_points=num_points
        )
        self.level_embed = nn.Parameter(torch.zeros(num_levels, d_model))
        self._geometry: dict[tuple, tuple[Tensor, Tensor]] = {}

    def geometry(self, shapes: tuple[tuple[int, int], ...], dtype: torch.dtype, device: torch.device):
        """Level+position embedding ``[1, S, C]`` and reference points ``[1, S, 2]`` for ``shapes`` (cached)."""
        key = (shapes, dtype, device)
        cached = self._geometry.get(key)
        if cached is None:
            pos = [
                sine_position_embedding(h, w, self.d_model // 2, device).flatten(1).t() + self.level_embed[level].float()
                for level, (h, w) in enumerate(shapes)
            ]
            cached = (torch.cat(pos)[None].to(dtype), reference_points(shapes, device)[None])
            self._geometry[key] = cached
        return cached

    def forward(self, srcs: list[Tensor]) -> list[Tensor]:
        shapes = tuple((s.shape[-2], s.shape[-1]) for s in srcs)
        batch = srcs[0].shape[0]
        src = torch.cat([s.flatten(2).transpose(1, 2) for s in srcs], dim=1)
        pos, reference = self.geometry(shapes, src.dtype, src.device)
        memory = self.encoder(src, pos.expand(batch, -1, -1), reference.expand(batch, -1, -1), shapes)
        out, start = [], 0
        for h, w in shapes:
            out.append(memory[:, start : start + h * w].transpose(1, 2).reshape(batch, self.d_model, h, w))
            start += h * w
        return out
