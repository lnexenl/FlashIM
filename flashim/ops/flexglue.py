"""Ops only FlexGlue uses (entmax readout of its correspondence heads)."""

from __future__ import annotations

import torch
from torch import Tensor

from flashim.ops._registry import Op


def _sorted_entmax15_threshold(ys: Tensor) -> Tensor:
    """Exact 1.5-entmax threshold from scores sorted in descending order."""
    rho = torch.arange(1, ys.shape[-1] + 1, device=ys.device, dtype=ys.dtype)
    mean = ys.cumsum(-1) / rho
    mean_sq = ys.square().cumsum(-1) / rho
    tau = mean - ((1 - rho * (mean_sq - mean.square())) / rho).clamp_min(0).sqrt()
    support = (tau <= ys).sum(-1, keepdim=True)
    return tau.gather(-1, support - 1)


def _entmax15_threshold(y: Tensor, k: int = 512) -> Tensor:
    """Tau of entmax15 = (y - tau)_+^2 from the top k (the support is <= k for the search maps FlexGlue sees)."""
    if y.shape[-1] <= k:
        return _sorted_entmax15_threshold(y.sort(-1, descending=True).values)
    yk = y.topk(k, dim=-1).values
    return _sorted_entmax15_threshold(yk)


def _flexglue_entmax_readout_reference(
    logits: Tensor, grid: Tensor, hw: tuple[int, int], radius: int = 1, k: int = 512
) -> tuple[Tensor, Tensor]:
    """1.5-entmax expected readout position ``[B, N, 2]`` and window mass ``[B, N]``.

    The expected position over the ``(2r+1)^2`` cells around the argmax of ``logits`` ``[B, N, h*w]``, weighted by
    the 1.5-entmax sparse mapping ``p = (y - tau)_+^2`` (which sums to 1 over the support); border cells clamped
    outside the map are dropped from both the position sum and the returned mass. ``grid`` is the ``[h*w, 2]`` table
    of normalized cell centers (``align_corners=False``). Mirrors cvg/FlexGlue's ``ops.entmax15_window_readout``.
    """
    if logits.shape[-1] > k:  # maximum and threshold from the same (sorted) top-k
        topv, topi = logits.topk(k, dim=-1)
        vmax, idx = topv[..., :1], topi[..., :1]
        y = (logits - vmax) / 2
        tau = _sorted_entmax15_threshold((topv - vmax) / 2)
    else:
        vmax, idx = logits.max(-1, keepdim=True)
        y = (logits - vmax) / 2
        tau = _entmax15_threshold(y, k)
    h, w = hw
    r = torch.arange(-radius, radius + 1, device=y.device)
    ny = (idx // w)[..., None] + r[:, None]
    nx = (idx % w)[..., None] + r[None, :]
    inside = ((ny >= 0) & (ny < h) & (nx >= 0) & (nx < w)).flatten(-2)
    nidx = (ny.clamp(0, h - 1) * w + nx.clamp(0, w - 1)).flatten(-2)
    p = (y.gather(-1, nidx) - tau).clamp_min(0).square()  # entmax sums to 1
    position = (p[..., None] * grid[nidx]).sum(-2) / p.sum(-1, keepdim=True)
    return position, (p * inside).sum(-1)  # clamped border cells count once


flexglue_entmax_readout = Op("flexglue_entmax_readout", _flexglue_entmax_readout_reference)
