"""Multi-scale deformable attention encoder layers (Deformable DETR naming).

A model built from :class:`DeformableEncoder` gets the fused quantized encoder
(:mod:`flashim.quant.passes.deformable_encoder`) once its value / output / FFN linears are static MXFP8.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from flashim import ops
from flashim.quant.methods.linear import StaticMXFP8Linear


class MSDeformAttn(nn.Module):
    """Multi-scale deformable attention (Deformable DETR): per head, ``n_levels * n_points`` bilinear samples at
    offsets predicted from the query, weighted by softmaxed logits (``ops.deformable_attention``)."""

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


class DeformableEncoderLayer(nn.Module):
    """Post-norm Deformable DETR encoder layer: ``norm1(src + self_attn(src + pos))``, then
    ``norm2(src + linear2(GELU(linear1(src))))``."""

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


class DeformableEncoder(nn.Module):
    def __init__(self, num_layers: int, **layer_kwargs) -> None:
        super().__init__()
        self.layers = nn.ModuleList(DeformableEncoderLayer(**layer_kwargs) for _ in range(num_layers))

    def forward(self, src: Tensor, pos: Tensor, reference: Tensor, shapes: tuple[tuple[int, int], ...]) -> Tensor:
        for layer in self.layers:
            src = layer(src, pos, reference, shapes)
        return src
