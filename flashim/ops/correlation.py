"""Local correlation between a feature map and a warped window of another; mutual matches of a similarity matrix."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor

from flashim.ops._registry import Op
from flashim.utils import grid_sample


def local_window(radius: int, h: int, w: int, device: torch.device) -> Tensor:
    """``[1, K, 2]`` (x, y) offsets of a (2r+1)^2 window, one feature pixel apart."""
    ys, xs = torch.meshgrid(
        torch.linspace(-2 * radius / h, 2 * radius / h, 2 * radius + 1, device=device),
        torch.linspace(-2 * radius / w, 2 * radius / w, 2 * radius + 1, device=device),
        indexing="ij",
    )
    return torch.stack((xs, ys), dim=-1).reshape(1, (2 * radius + 1) ** 2, 2)


def _local_corr_reference(feature0: Tensor, feature1: Tensor, warp: Tensor, radius: int) -> Tensor:
    """Correlate ``feature0 [B,C,H,W]`` with ``feature1`` sampled around ``warp [B,H,W,2]``.

    Returns ``[B, K, H, W]`` with ``K = (2r+1)^2``; features are scaled by
    ``1/sqrt(C)``; sampling is bilinear, zeros padding, ``align_corners=False``.
    """
    batch, channels, h, w = feature0.shape
    window = local_window(radius, h, w, feature0.device)
    k = window.shape[1]
    corr = torch.empty((batch, k, h, w), device=feature0.device, dtype=feature0.dtype)
    for b in range(batch):
        coords = (warp[b, :, :, None] + window[:, None, None]).reshape(1, h, w * k, 2)
        sampled = grid_sample(
            feature1[b : b + 1], coords, padding_mode="zeros", align_corners=False, mode="bilinear"
        ).reshape(channels, h, w, k)
        corr[b] = (feature0[b, ..., None] / channels**0.5 * sampled).sum(dim=0).permute(2, 0, 1)
    return corr


local_corr = Op("local_corr", _local_corr_reference)


def _dual_softmax_mutual_reference(
    sim: Tensor, row_bias: Tensor | None = None, col_bias: Tensor | None = None
) -> tuple[Tensor, Tensor, Tensor]:
    """Mutual best matches of a dual-softmax assignment over a similarity matrix ``sim [L, S]`` (FP32).

    ``score[i, j] = log_softmax(sim, 1)[i, j] + log_softmax(sim, 0)[i, j] + row_bias[i] + col_bias[j]`` (the log of
    ``softmax(sim, 1) * softmax(sim, 0)``, the biases are e.g. LightGlue's log matchabilities). Returns, per row
    ``i``: ``log_conf [L]`` (FP32, the row maximum of ``score``), ``index [L]`` (int64, its column) and
    ``mutual [L]`` (bool: row ``i`` is also the best row of column ``index[i]``). Exact ties are broken arbitrarily.
    """
    score = F.log_softmax(sim, 1)
    score += F.log_softmax(sim, 0)
    if row_bias is not None:
        score += row_bias[:, None]
    if col_bias is not None:
        score += col_bias[None, :]
    best = score.max(1)
    best_row = score.max(0).indices
    mutual = best_row[best.indices] == torch.arange(sim.shape[0], device=sim.device)
    return best.values, best.indices, mutual


dual_softmax_mutual = Op("dual_softmax_mutual", _dual_softmax_mutual_reference)


def _union_softmax_match_reference(sim: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Row-wise and column-wise one-way softmax maxima and argmax indices over ``sim [L, S]`` (FP32).

    Returns ``(row_prob [L], row_j [L], col_prob [S], col_i [S])`` where ``row_j[i] = argmax_j sim[i, j]``,
    ``row_prob[i] = softmax(sim, 1)[i, row_j[i]]``, and symmetrically for columns."""
    row_max, row_j = sim.max(dim=1)
    row_prob = (row_max - torch.logsumexp(sim, dim=1)).exp()
    col_max, col_i = sim.max(dim=0)
    col_prob = (col_max - torch.logsumexp(sim, dim=0)).exp()
    return row_prob, row_j, col_prob, col_i


union_softmax_match = Op("union_softmax_match", _union_softmax_match_reference)


def _batched_dual_softmax_argmax_reference(
    f0: Tensor, f1: Tensor, scale: float
) -> tuple[Tensor, Tensor, Tensor]:
    """Per-batch dual-softmax global maximum over small ``[M, W, C]`` windows (FP32):
    ``conf = softmax(sim, 1) * softmax(sim, 2)`` where ``sim = (f0 @ f1^T) * scale``.
    Returns ``(best_conf [M], best_i [M], best_j [M])``."""
    if f0.shape[0] == 0:
        empty_i = torch.empty((0,), device=f0.device, dtype=torch.long)
        return torch.empty((0,), device=f0.device, dtype=torch.float32), empty_i, empty_i
    sim = torch.einsum("mlc,msc->mls", f0.float(), f1.float()) * scale
    conf = F.softmax(sim, dim=1) * F.softmax(sim, dim=2)
    flat = conf.reshape(conf.shape[0], -1)
    best_conf, best_flat = flat.max(dim=1)
    w1 = sim.shape[2]
    return best_conf, torch.div(best_flat, w1, rounding_mode="trunc"), best_flat % w1


batched_dual_softmax_argmax = Op("batched_dual_softmax_argmax", _batched_dual_softmax_argmax_reference)
