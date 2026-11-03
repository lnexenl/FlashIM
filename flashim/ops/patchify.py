"""ViT patch embedding: ``patch_embed_tokens`` (the stride-``p`` convolution, class token and positional embedding of a
ViT's input as one fused op)."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor

from flashim.ops._registry import Op


def _patch_embed_tokens_reference(
    image: Tensor, cls_token: Tensor, pos_embed: Tensor, weight: Tensor, bias: Tensor
) -> Tensor:
    """``image`` ``[B, C, H, W]`` (H, W multiples of the patch size ``p = weight.shape[-1]``) -> tokens ``[B, 1 + H W / p^2,
    D]`` in ``weight``'s dtype: ``conv2d(image, weight, bias, stride=p)`` flattened to patch tokens, ``cls_token``
    ``[1, 1, D]`` in front, ``pos_embed`` ``[1, 1 + H W / p^2, D]`` added (each rounded to the token dtype, as the layers
    of a BF16 ViT do)."""
    batch = image.shape[0]
    tokens = F.conv2d(image.to(weight.dtype), weight, bias, stride=weight.shape[-1]).flatten(2).transpose(1, 2)
    tokens = torch.cat([cls_token.to(tokens.dtype).expand(batch, -1, -1), tokens], 1)
    return tokens + pos_embed.to(tokens.dtype)


patch_embed_tokens = Op("patch_embed_tokens", _patch_embed_tokens_reference, weights=2)
