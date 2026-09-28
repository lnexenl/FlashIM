"""Multi-scale deformable transformer encoder (Deformable DETR) with cached per-shape geometry."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from flashim import ops
from flashim.quant.methods.linear import StaticMXFP8Linear


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


class MSDeformAttn(nn.Module):
    def __init__(self, d_model: int, n_levels: int, n_heads: int, n_points: int) -> None:
        super().__init__()
        self.d_model, self.n_levels, self.n_heads, self.n_points = d_model, n_levels, n_heads, n_points
        self.sampling_offsets = nn.Linear(d_model, n_heads * n_levels * n_points * 2)
        self.attention_weights = nn.Linear(d_model, n_heads * n_levels * n_points)
        self.value_proj = nn.Linear(d_model, d_model)
        self.output_proj = nn.Linear(d_model, d_model)
        # One GEMM for sampling_offsets + attention_weights (see ``fuse_query_projections``).
        self.query_proj: nn.Linear | None = None

    def fuse_query_projections(self) -> bool:
        """Replace the two BF16/FP32 query projections by one concatenated linear (offsets first)."""
        a, b = self.sampling_offsets, self.attention_weights
        if type(a) is not nn.Linear or type(b) is not nn.Linear:
            return False
        fused = nn.Linear(a.in_features, a.out_features + b.out_features, device=a.weight.device, dtype=a.weight.dtype)
        fused.weight.data = torch.cat([a.weight.data, b.weight.data])
        fused.bias.data = torch.cat([a.bias.data, b.bias.data])
        self.query_proj = fused
        del self.sampling_offsets, self.attention_weights
        return True

    def project_query(self, query: Tensor) -> tuple[Tensor, Tensor]:
        """Sampling offsets ``[N, Lq, M, L, P, 2]`` and attention logits ``[N, Lq, M, L, P]`` (possibly strided views)."""
        n, length, _ = query.shape
        heads, levels, points = self.n_heads, self.n_levels, self.n_points
        if self.query_proj is not None:
            proj = self.query_proj(query)
            split = heads * levels * points * 2
            offsets, logits = proj[..., :split], proj[..., split:]
        else:
            a, b = self.sampling_offsets, self.attention_weights
            if (
                isinstance(a, StaticMXFP8Linear)
                and isinstance(b, StaticMXFP8Linear)
                and torch.equal(a.input_exponent, b.input_exponent)
            ):  # both read the same activation: quantize it once
                quantized = ops.quantize_mxfp8(query, a.input_exponent)
                offsets, logits = a.matmul(quantized, bias=True), b.matmul(quantized, bias=True)
            else:
                offsets, logits = a(query), b(query)
        return (
            offsets.view(n, length, heads, levels, points, 2),
            logits.view(n, length, heads, levels, points),
        )

    def forward(self, query: Tensor, src: Tensor, reference: Tensor, shapes: tuple[tuple[int, int], ...]) -> Tensor:
        n = query.shape[0]
        value = self.value_proj(src).view(n, src.shape[1], self.n_heads, self.d_model // self.n_heads)
        offsets, logits = self.project_query(query)
        out = ops.deformable_attention(value, offsets, logits, reference, shapes, softmax=True)
        return self.output_proj(out)


class EncoderLayer(nn.Module):
    def __init__(self, d_model: int, d_ffn: int, n_levels: int, n_heads: int, n_points: int) -> None:
        super().__init__()
        self.self_attn = MSDeformAttn(d_model, n_levels, n_heads, n_points)
        self.norm1 = nn.LayerNorm(d_model)
        self.linear1 = nn.Linear(d_model, d_ffn)
        self.linear2 = nn.Linear(d_ffn, d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, src: Tensor, pos: Tensor, reference: Tensor, shapes: tuple[tuple[int, int], ...]) -> Tensor:
        src = self.norm1(src + self.self_attn(src + pos, src, reference, shapes))
        return self.norm2(src + self.linear2(F.gelu(self.linear1(src))))


class Encoder(nn.Module):
    def __init__(self, num_layers: int, **layer_kwargs) -> None:
        super().__init__()
        self.layers = nn.ModuleList(EncoderLayer(**layer_kwargs) for _ in range(num_layers))

    def forward(self, src: Tensor, pos: Tensor, reference: Tensor, shapes: tuple[tuple[int, int], ...]) -> Tensor:
        for layer in self.layers:
            src = layer(src, pos, reference, shapes)
        return src


class DeformableTransformer(nn.Module):
    """Flattens the feature levels, adds position + level embeddings and runs the encoder."""

    def __init__(self, d_model: int, num_heads: int, num_layers: int, d_ffn: int, num_levels: int, num_points: int) -> None:
        super().__init__()
        self.d_model = d_model
        self.encoder = Encoder(
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
