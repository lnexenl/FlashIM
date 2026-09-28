"""Pre-norm ViT layers (DINOv2/v3, timm naming).

Parameter names are ``norm1 / attn.qkv / attn.proj / ls1 / norm2 / mlp.fc1 / mlp.act / mlp.fc2 / ls2``, so timm-style
checkpoints load strictly. A model built from :class:`TransformerBlock` gets the quantized block fusion
(:mod:`flashim.quant.passes.transformer_block`: LayerNorm -> quantize, GELU -> quantize, prequantized GEMMs).
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from flashim import ops


class LinearKMaskedBias(nn.Linear):
    """Packed QKV projection whose K bias is masked out by a stored ``bias_mask`` buffer (DINOv3). Quantization folds
    the mask into the bias once."""

    def __init__(self, in_features: int, out_features: int, bias: bool = True) -> None:
        super().__init__(in_features, out_features, bias=bias)
        if bias:
            self.register_buffer("bias_mask", torch.full_like(self.bias, math.nan))

    def forward(self, x: Tensor) -> Tensor:
        bias = self.bias * self.bias_mask.to(self.bias.dtype) if self.bias is not None else None
        return F.linear(x, self.weight, bias)


class LayerScale(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(dim))

    def forward(self, x: Tensor) -> Tensor:
        return x * self.gamma


class Mlp(nn.Module):
    """``fc2(GELU(fc1(x)))``."""

    def __init__(self, dim: int, hidden: int) -> None:
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden, dim)

    def forward(self, x: Tensor) -> Tensor:
        return self.fc2(self.act(self.fc1(x)))


class Attention(nn.Module):
    """Multi-head self-attention with a packed QKV projection and optional half-split RoPE on the trailing tokens.

    ``rope_impl`` / ``sdpa_impl`` pin ``ops.rope_qk`` / ``ops.sdpa`` impls (``None`` = dispatch); the defaults
    reproduce upstream eager math.
    """

    def __init__(self, dim: int, num_heads: int, *, mask_k_bias: bool = False, rope_impl: str | None = None) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.qkv = LinearKMaskedBias(dim, 3 * dim) if mask_k_bias else nn.Linear(dim, 3 * dim)
        self.proj = nn.Linear(dim, dim)
        self.rope_impl = rope_impl
        self.sdpa_impl: str | None = "reference"

    def forward(self, x: Tensor, rope: tuple[Tensor, Tensor] | None = None) -> Tensor:
        batch, tokens, channels = x.shape
        qkv = self.qkv(x).reshape(batch, tokens, 3, self.num_heads, channels // self.num_heads)
        q, k, v = (t.transpose(1, 2) for t in qkv.unbind(2))
        if rope is not None:
            q, k = ops.rope_qk.get(self.rope_impl)(q, k, *rope)
        out = ops.sdpa.get(self.sdpa_impl)(q, k, v)
        return self.proj(out.transpose(1, 2).reshape(batch, tokens, channels))


class TransformerBlock(nn.Module):
    """``x + ls1(attn(norm1(x)))``, then ``x + ls2(mlp(norm2(x)))``; ``ls*`` are :class:`LayerScale` or identity."""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        *,
        eps: float = 1e-6,
        layer_scale: bool = False,
        mask_k_bias: bool = False,
        rope_impl: str | None = None,
        mlp_ratio: int = 4,
    ) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, eps=eps)
        self.attn = Attention(dim, num_heads, mask_k_bias=mask_k_bias, rope_impl=rope_impl)
        self.ls1 = LayerScale(dim) if layer_scale else nn.Identity()
        self.norm2 = nn.LayerNorm(dim, eps=eps)
        self.mlp = Mlp(dim, mlp_ratio * dim)
        self.ls2 = LayerScale(dim) if layer_scale else nn.Identity()

    def forward(self, x: Tensor, rope: tuple[Tensor, Tensor] | None = None) -> Tensor:
        x = x + self.ls1(self.attn(self.norm1(x), rope))
        return x + self.ls2(self.mlp(self.norm2(x)))
