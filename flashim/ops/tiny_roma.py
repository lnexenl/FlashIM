"""Tiny-RoMa-specific fused op: global stride-8 correlation + positional embedding.

``tiny_roma_corr_pos_embed(feat0, feat1, exact_softmax=False)`` computes the coarse normalized target coordinates
``[B, 2, H0, W0]`` (FP32) from ``feat0 [B, C, H0, W0]`` and ``feat1 [B, C, H1, W1]`` without materializing the
``[B, H1, W1, H0, W0]`` correlation volume in global memory.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor

from flashim.ops._registry import Op


def _corr_pos_embed_reference(feat0: Tensor, feat1: Tensor, exact_softmax: bool = False) -> Tensor:
    """Reference torch implementation matching upstream ``TinyRoMa.corr_volume`` + ``TinyRoMa.pos_embed``."""
    b, c, h0, w0 = feat0.shape
    _, _, h1, w1 = feat1.shape
    device = feat0.device
    f0 = feat0.reshape(b, c, h0 * w0)
    f1 = feat1.reshape(b, c, h1 * w1)
    corr_volume = torch.einsum("bci,bcj->bji", f0, f1).reshape(b, h1, w1, h0, w0) / math.sqrt(c)
    grid = torch.stack(
        torch.meshgrid(
            torch.linspace(-1 + 1 / w1, 1 - 1 / w1, w1, device=device),
            torch.linspace(-1 + 1 / h1, 1 - 1 / h1, h1, device=device),
            indexing="xy",
        ),
        dim=-1,
    ).float().reshape(h1 * w1, 2)
    down = 4
    if not exact_softmax:
        grid_lr = torch.stack(
            torch.meshgrid(
                torch.linspace(-1 + down / w1, 1 - down / w1, w1 // down, device=device),
                torch.linspace(-1 + down / h1, 1 - down / h1, h1 // down, device=device),
                indexing="xy",
            ),
            dim=-1,
        ).float().reshape((h1 // down) * (w1 // down), 2)
        best_match = corr_volume.reshape(b, h1 * w1, h0, w0).argmax(dim=1)
        cv_lr = corr_volume[:, ::down, ::down].reshape(b, (h1 // down) * (w1 // down), h0, w0).float()
        p_lowres = torch.cat((cv_lr, best_match[:, None].float()), dim=1).softmax(dim=1)
        pos_embeddings = torch.einsum("bchw,cd->bdhw", p_lowres[:, :-1], grid_lr)
        return pos_embeddings + p_lowres[:, -1:] * grid[best_match].permute(0, 3, 1, 2)
    p = corr_volume.reshape(b, h1 * w1, h0, w0).float().softmax(dim=1)
    return torch.einsum("bchw,cd->bdhw", p, grid)


tiny_roma_corr_pos_embed = Op("tiny_roma_corr_pos_embed", _corr_pos_embed_reference)
