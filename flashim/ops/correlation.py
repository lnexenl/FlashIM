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


def _mutual_nn_match_reference(desc0: Tensor, desc1: Tensor, threshold: float = 0.0) -> tuple[Tensor, Tensor, Tensor]:
    """Mutual-nearest-neighbour inner-product matches with similarity >= ``threshold``.

    ``desc0 [M, D]``, ``desc1 [N, D]`` -> ``(idx0 [K] int64, idx1 [K] int64, score [K] fp32)``.
    """
    if desc0.shape[0] == 0 or desc1.shape[0] == 0:
        empty = torch.zeros(0, dtype=torch.long, device=desc0.device)
        return empty, empty, desc0.new_zeros(0, dtype=torch.float32)
    sim = desc0.float() @ desc1.float().t()
    nn01, nn10 = sim.argmax(dim=1), sim.argmax(dim=0)
    best = sim.gather(1, nn01[:, None])[:, 0]
    ids0 = torch.arange(sim.shape[0], device=sim.device)
    idx0 = ((ids0 == nn10[nn01]) & (best >= threshold)).nonzero()[:, 0]
    return idx0, nn01[idx0], best[idx0]


mutual_nn_match = Op("mutual_nn_match", _mutual_nn_match_reference)
