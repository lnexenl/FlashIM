"""Normalization ops: LayerNorm fused with a residual add or left-concatenation."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor

from flashim.ops._registry import Op


def _residual_layernorm_reference(
    x: Tensor, residual: Tensor | None, weight: Tensor, bias: Tensor, eps: float = 1e-5
) -> Tensor:
    """``F.layer_norm(x, (C,), weight, bias, eps)`` (+ ``residual`` if provided), in ``x``'s dtype."""
    normed = F.layer_norm(x, (x.shape[-1],), weight, bias, eps)
    return normed if residual is None else residual + normed


residual_layernorm = Op("residual_layernorm", _residual_layernorm_reference)


def _cat_layernorm_reference(
    left: Tensor, right: Tensor, weight: Tensor, bias: Tensor, eps: float = 1e-5
) -> Tensor:
    """``torch.cat((left, F.layer_norm(right, (C,), weight, bias, eps)), dim=-1)``."""
    return torch.cat((left, F.layer_norm(right, (right.shape[-1],), weight, bias, eps)), dim=-1)


cat_layernorm = Op("cat_layernorm", _cat_layernorm_reference)


def _pos_cat_layernorm_reference(
    x: Tensor,
    x_pre: Tensor,
    pos: Tensor,
    pos_pre: Tensor | None,
    weight: Tensor,
    bias: Tensor,
    eps: float = 1e-5,
    *,
    return_all_pos: bool = False,
) -> tuple[Tensor, Tensor]:
    """Add 1-D window positional encodings to ``x [B, L0, C]`` and ``x_pre [B, L1, C]``, concatenate along ``dim=1``,
    and apply LayerNorm. Returns ``(x_pos, x_all_norm)`` where ``x_pos`` is ``[B, L0, C]`` (or ``[B, L0+L1, C]`` when
    ``return_all_pos=True``) and ``x_all_norm`` is ``[B, L0+L1, C]``."""
    xp = x + pos.to(x.dtype)
    xpp = x_pre if pos_pre is None else x_pre + pos_pre.to(x_pre.dtype)
    x_all = torch.cat((xp, xpp), dim=1)
    x_norm = F.layer_norm(x_all, (x_all.shape[-1],), weight, bias, eps)
    return (x_all if return_all_pos else xp), x_norm


pos_cat_layernorm = Op("pos_cat_layernorm", _pos_cat_layernorm_reference)
