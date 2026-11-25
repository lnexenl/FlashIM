"""Assignment ops: differentiable optimal transport (log-space Sinkhorn with dustbin)."""

from __future__ import annotations

import math

import torch
from torch import Tensor

from flashim.ops._registry import Op


def _log_optimal_transport_reference(scores: Tensor, alpha: Tensor, iters: int) -> Tensor:
    """Log-space Sinkhorn optimal transport with a dustbin row and column (SuperGlue, Sarlin et al. 2020).

    ``scores`` ``[B, M, N]``: pairwise similarity matrix; ``alpha``: scalar (or broadcastable) dustbin score;
    ``iters``: number of alternating row/column log-space Sinkhorn iterations. Returns ``[B, M + 1, N + 1]``
    log-assignment scores scaled by ``M + N`` (so each non-dustbin row and column sums to 1 in probability space
    at convergence).
    """
    if scores.ndim != 3:
        raise ValueError(f"scores must have shape [B, M, N], got {tuple(scores.shape)}")
    if iters < 0:
        raise ValueError(f"iters must be non-negative, got {iters}")
    b, m, n = scores.shape
    if m == 0 or n == 0:
        raise ValueError(f"scores must be non-empty along M and N, got {tuple(scores.shape)}")
    alpha_t = alpha.to(device=scores.device, dtype=scores.dtype)
    bins0 = alpha_t.expand(b, m, 1)
    bins1 = alpha_t.expand(b, 1, n)
    corner = alpha_t.expand(b, 1, 1)
    couplings = torch.cat(
        (torch.cat((scores, bins0), dim=-1), torch.cat((bins1, corner), dim=-1)),
        dim=1,
    )
    norm = -math.log(m + n)
    log_mu = scores.new_full((b, m + 1), norm)
    log_mu[:, -1] = math.log(n) + norm
    log_nu = scores.new_full((b, n + 1), norm)
    log_nu[:, -1] = math.log(m) + norm

    u = torch.zeros_like(log_mu)
    v = torch.zeros_like(log_nu)
    for _ in range(iters):
        u = log_mu - torch.logsumexp(couplings + v.unsqueeze(1), dim=2)
        v = log_nu - torch.logsumexp(couplings + u.unsqueeze(2), dim=1)
    return couplings + u.unsqueeze(2) + v.unsqueeze(1) - norm


log_optimal_transport = Op("log_optimal_transport", _log_optimal_transport_reference)


def _sinkhorn_mutual_reference(
    sim: Tensor, bin_score: Tensor, iters: int = 100
) -> tuple[Tensor, Tensor, Tensor]:
    """Mutual best matches of :func:`log_optimal_transport` over ``sim [M, N]`` (FP32) with dustbin ``bin_score``.

    Returns per row ``i``: ``prob [M]`` (FP32, row maximum of ``exp(log_P[:M, :N])``), ``index [M]`` (int64, its
    column) and ``mutual [M]`` (bool).
    """
    m, n = sim.shape
    prob = _log_optimal_transport_reference(sim.float().unsqueeze(0), bin_score.float().reshape(()), iters)[0, :m, :n].exp()
    best0 = prob.max(dim=1)
    best1 = prob.argmax(dim=0)
    mutual = best1[best0.indices] == torch.arange(m, device=sim.device)
    return best0.values, best0.indices, mutual


sinkhorn_mutual = Op("sinkhorn_mutual", _sinkhorn_mutual_reference)
