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


def _nn_query_blocked(queries: Tensor, db: Tensor, block_size: int, dist: str) -> Tensor:
    """For each row of ``queries [Q, D]``, index of its nearest neighbor in ``db [N, D]`` (first occurrence wins on
    ties, matching ``torch.max`` / ``torch.min``)."""
    q_count, n_count = queries.shape[0], db.shape[0]
    if q_count == 0:
        return torch.empty(0, dtype=torch.int64, device=queries.device)
    if dist not in ("dot", "l2"):
        raise ValueError(f"Unknown dist={dist!r}; expected 'dot' or 'l2'")
    best_idx = torch.empty(q_count, dtype=torch.int64, device=queries.device)
    for qs in range(0, q_count, block_size):
        qe = min(qs + block_size, q_count)
        q_blk = queries[qs:qe]
        if n_count <= block_size:
            if dist == "dot":
                best_idx[qs:qe] = (q_blk @ db.t()).argmax(dim=1)
            else:
                best_idx[qs:qe] = torch.cdist(q_blk, db).argmin(dim=1)
            continue
        best_val = torch.full(
            (qe - qs,), float("-inf") if dist == "dot" else float("inf"),
            device=queries.device, dtype=queries.dtype,
        )
        cur_idx = torch.zeros(qe - qs, dtype=torch.int64, device=queries.device)
        for ns in range(0, n_count, block_size):
            ne = min(ns + block_size, n_count)
            d_blk = db[ns:ne]
            if dist == "dot":
                vals, idx = (q_blk @ d_blk.t()).max(dim=1)
                better = vals > best_val
            else:
                vals, idx = torch.cdist(q_blk, d_blk).min(dim=1)
                better = vals < best_val
            best_val = torch.where(better, vals, best_val)
            cur_idx = torch.where(better, idx + ns, cur_idx)
        best_idx[qs:qe] = cur_idx
    return best_idx


def _fast_reciprocal_nn_reference(
    desc1: Tensor,
    desc2: Tensor,
    subsample: int = 8,
    max_iter: int = 10,
    block_size: int = 8192,
    dist: str = "dot",
) -> tuple[Tensor, Tensor]:
    """Iterative reciprocal nearest-neighbor matching on dense descriptor maps (MASt3R ``fast_reciprocal_NNs``).

    Starts from a stride-``subsample`` grid on ``desc1 [H1, W1, D]`` and alternates nearest-neighbor lookups into
    ``desc2 [H2, W2, D]`` and back into ``desc1`` for up to ``max_iter`` iterations until fixed points (reciprocal
    pairs) converge. Returns unique converged correspondences sorted by image-1 flat index: ``(xy1, xy2)`` as
    int64 ``[M, 2]`` ``(x, y)`` pixel indices.
    """
    h1, w1, dim1 = desc1.shape
    _h2, w2, dim2 = desc2.shape
    if dim1 != dim2:
        raise ValueError(f"Descriptor dim mismatch: {dim1} vs {dim2}")
    device = desc1.device
    start = subsample // 2
    if start >= h1 or start >= w1:
        empty = torch.empty((0, 2), dtype=torch.int64, device=device)
        return empty, empty
    pts1 = desc1.reshape(-1, dim1)
    pts2 = desc2.reshape(-1, dim2)

    ys = torch.arange(start, h1, subsample, device=device, dtype=torch.int64)
    xs = torch.arange(start, w1, subsample, device=device, dtype=torch.int64)
    grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
    xy1 = (grid_x + w1 * grid_y).reshape(-1)
    xy2 = torch.full_like(xy1, -1)
    old_xy1 = xy1.clone()
    old_xy2 = xy2.clone()
    notyet = torch.ones(xy1.shape[0], dtype=torch.bool, device=device)

    for _ in range(max_iter):
        active = notyet.nonzero(as_tuple=True)[0]
        if active.numel() == 0:
            break
        xy2[active] = _nn_query_blocked(pts1[xy1[active]], pts2, block_size, dist)
        notyet &= old_xy2 != xy2

        active = notyet.nonzero(as_tuple=True)[0]
        if active.numel() == 0:
            break
        xy1[active] = _nn_query_blocked(pts2[xy2[active]], pts1, block_size, dist)
        notyet &= old_xy1 != xy1

        old_xy2.copy_(xy2)
        old_xy1.copy_(xy1)

    converged = ~notyet
    c1, c2 = xy1[converged], xy2[converged]
    if c1.numel() == 0:
        empty = torch.empty((0, 2), dtype=torch.int64, device=device)
        return empty, empty
    packed = torch.unique((c1 << 32) | c2, sorted=True)
    idx1 = packed >> 32
    idx2 = packed & 0xFFFFFFFF
    out1 = torch.stack((idx1 % w1, idx1 // w1), dim=-1)
    out2 = torch.stack((idx2 % w2, idx2 // w2), dim=-1)
    return out1, out2


fast_reciprocal_nn = Op("fast_reciprocal_nn", _fast_reciprocal_nn_reference)


def _l2_nn_blocked(queries: Tensor, db: Tensor, block_size: int) -> Tensor:
    """For each row of ``queries [Q, D]``, index of its exact L2 nearest neighbor in ``db [N, D]`` using
    ``donot_use_mm_for_euclid_dist`` to avoid catastrophic cancellation on distant 3D points."""
    q_count, n_count = queries.shape[0], db.shape[0]
    if q_count == 0:
        return torch.empty(0, dtype=torch.int64, device=queries.device)
    best_idx = torch.empty(q_count, dtype=torch.int64, device=queries.device)
    for qs in range(0, q_count, block_size):
        qe = min(qs + block_size, q_count)
        q_blk = queries[qs:qe]
        if n_count <= block_size:
            best_idx[qs:qe] = torch.cdist(
                q_blk, db, compute_mode="donot_use_mm_for_euclid_dist"
            ).argmin(dim=1)
            continue
        best_val = torch.full((qe - qs,), float("inf"), device=queries.device, dtype=queries.dtype)
        cur_idx = torch.zeros(qe - qs, dtype=torch.int64, device=queries.device)
        for ns in range(0, n_count, block_size):
            ne = min(ns + block_size, n_count)
            vals, idx = torch.cdist(
                q_blk, db[ns:ne], compute_mode="donot_use_mm_for_euclid_dist"
            ).min(dim=1)
            better = vals < best_val
            best_val = torch.where(better, vals, best_val)
            cur_idx = torch.where(better, idx + ns, cur_idx)
        best_idx[qs:qe] = cur_idx
    return best_idx


def _reciprocal_nn_reference(pts1: Tensor, pts2: Tensor, block_size: int = 4096) -> tuple[Tensor, Tensor]:
    """Exact Euclidean (L2) reciprocal nearest neighbors between ``pts1 [N1, D]`` and ``pts2 [N2, D]``.

    Equivalent to upstream DUSt3R ``find_reciprocal_matches(P1, P2)``: finds every index pair ``(i, j)`` where
    ``j = argmin_k ||pts1[i] - pts2[k]||_2`` and ``i = argmin_k ||pts2[j] - pts1[k]||_2``. Returns ``(idx1, idx2)``
    as int64 ``[M]`` tensors sorted by ``idx2``.
    """
    if pts1.ndim != 2 or pts2.ndim != 2 or pts1.shape[1] != pts2.shape[1]:
        raise ValueError(f"Expected [N1, D] and [N2, D] point sets, got {tuple(pts1.shape)} and {tuple(pts2.shape)}")
    device = pts1.device
    if pts1.shape[0] == 0 or pts2.shape[0] == 0:
        empty = torch.empty(0, dtype=torch.int64, device=device)
        return empty, empty
    nn1_in_2 = _l2_nn_blocked(pts1, pts2, block_size)
    nn2_in_1 = _l2_nn_blocked(pts2, pts1, block_size)
    idx2 = torch.arange(pts2.shape[0], device=device, dtype=torch.int64)
    reciprocal = nn1_in_2[nn2_in_1] == idx2
    return nn2_in_1[reciprocal], idx2[reciprocal]


reciprocal_nn = Op("reciprocal_nn", _reciprocal_nn_reference)
