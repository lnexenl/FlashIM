# Follows code that RoMaV2 took from Meta (DINOv3 License Agreement, Copyright (c) Meta Platforms, Inc. and affiliates);
# see THIRD_PARTY_NOTICES.md.
"""Inference-only DINOv3 ViT-L/16 feature extractor (RoMaV2 coarse descriptor).

Written for FlashIM; architecture and numerics follow the upstream model as RoMaV2
uses it: all weights in BF16, forward under BF16 autocast, ImageNet normalization,
4 storage tokens + CLS, RoPE (base 100, FP32 tables), K-bias mask, LayerScale.
Outputs the final-norm'ed patch tokens of the requested blocks as ``[B, h, w, D]``.
The weights come from the RoMaV2 checkpoint and are covered by the DINOv3 license.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

from flashim.layers import TransformerBlock
from flashim.models.romav2.geometry import imagenet_normalize
from flashim.models.romav2.vit import RopeEmbedding


class PatchEmbed(nn.Module):
    def __init__(self, patch_size: int, dim: int) -> None:
        super().__init__()
        self.proj = nn.Conv2d(3, dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x: Tensor) -> Tensor:
        return self.proj(x).flatten(2).transpose(1, 2)


class DinoV3(nn.Module):
    def __init__(
        self,
        *,
        dim: int = 1024,
        depth: int = 24,
        num_heads: int = 16,
        patch_size: int = 16,
        n_storage_tokens: int = 4,
        layers: tuple[int, ...] = (11, 17),
    ) -> None:
        super().__init__()
        self.patch_size = patch_size
        self.n_storage_tokens = n_storage_tokens
        self.layers = tuple(layers)
        self.patch_embed = PatchEmbed(patch_size, dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, dim))
        self.storage_tokens = nn.Parameter(torch.zeros(1, n_storage_tokens, dim))
        self.mask_token = nn.Parameter(torch.zeros(1, dim))
        self.rope_embed = RopeEmbedding(dim // num_heads, base=100.0, compute_dtype=torch.float32)
        self.blocks = nn.ModuleList(
            TransformerBlock(dim, num_heads, eps=1e-5, layer_scale=True, mask_k_bias=True, rope_impl="reference")
            for _ in range(depth)
        )
        self.norm = nn.LayerNorm(dim, eps=1e-5)

    def truncate(self) -> int:
        """Drop blocks after the last requested layer (their outputs are never used)."""
        keep = max(self.layers) + 1
        if keep < len(self.blocks):
            self.blocks = nn.ModuleList(list(self.blocks[:keep]))
        return len(self.blocks)

    def forward(self, img: Tensor) -> list[Tensor]:
        with torch.autocast(img.device.type, torch.bfloat16):
            batch, _, height, width = img.shape
            h, w = height // self.patch_size, width // self.patch_size
            x = self.patch_embed(imagenet_normalize(img))
            cls = self.cls_token + 0 * self.mask_token
            x = torch.cat(
                (cls.expand(batch, -1, -1), self.storage_tokens.expand(batch, -1, -1), x), dim=1
            )
            rope = self.rope_embed(h, w)
            outputs = []
            for index, block in enumerate(self.blocks):
                x = block(x, rope)
                if index in self.layers:
                    outputs.append(x)
            prefix = self.n_storage_tokens + 1
            return [self.norm(out)[:, prefix:].reshape(batch, h, w, -1) for out in outputs]
