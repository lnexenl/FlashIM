"""Pre-norm ViT building blocks shared by the DINOv3 descriptor and the matcher ViT.

Module/parameter names follow the RoMaV2 checkpoint (``norm1/attn.qkv/attn.proj/
ls1/norm2/mlp.fc1/mlp.fc2/ls2``) so ``state_dict`` loads strictly, and so the
quantization passes can recognize blocks structurally.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from flashim import ops


class LinearKMaskedBias(nn.Linear):
    """QKV projection whose K bias is masked out by a stored ``bias_mask`` buffer."""

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
    def __init__(self, dim: int, hidden: int) -> None:
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden, dim)

    def forward(self, x: Tensor) -> Tensor:
        return self.fc2(self.act(self.fc1(x)))


class Attention(nn.Module):
    """Multi-head self-attention with optional RoPE on the trailing tokens.

    ``rope_impl`` pins the ``rope_qk`` implementation (``None`` = dispatch); the
    DINOv3 descriptor pins ``"reference"`` to reproduce upstream eager math.
    """

    def __init__(self, dim: int, num_heads: int, *, mask_k_bias: bool, rope_impl: str | None) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.qkv = LinearKMaskedBias(dim, 3 * dim) if mask_k_bias else nn.Linear(dim, 3 * dim)
        self.proj = nn.Linear(dim, dim)
        self.rope_impl = rope_impl

    def forward(self, x: Tensor, rope: tuple[Tensor, Tensor] | None = None) -> Tensor:
        batch, tokens, channels = x.shape
        qkv = self.qkv(x).reshape(batch, tokens, 3, self.num_heads, channels // self.num_heads)
        q, k, v = (t.transpose(1, 2) for t in qkv.unbind(2))
        if rope is not None:
            q, k = ops.rope_qk.get(self.rope_impl)(q, k, *rope)
        out = F.scaled_dot_product_attention(q, k, v)
        return self.proj(out.transpose(1, 2).reshape(batch, tokens, channels))


class Block(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        *,
        eps: float,
        layer_scale: bool,
        mask_k_bias: bool,
        rope_impl: str | None,
    ) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, eps=eps)
        self.attn = Attention(dim, num_heads, mask_k_bias=mask_k_bias, rope_impl=rope_impl)
        self.ls1 = LayerScale(dim) if layer_scale else nn.Identity()
        self.norm2 = nn.LayerNorm(dim, eps=eps)
        self.mlp = Mlp(dim, 4 * dim)
        self.ls2 = LayerScale(dim) if layer_scale else nn.Identity()

    def forward(self, x: Tensor, rope: tuple[Tensor, Tensor] | None = None) -> Tensor:
        x = x + self.ls1(self.attn(self.norm1(x), rope))
        return x + self.ls2(self.mlp(self.norm2(x)))


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
