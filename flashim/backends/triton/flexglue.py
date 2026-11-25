"""Triton impl of :func:`flashim.ops.flexglue_entmax_readout`: zero-sort 1.5-entmax threshold + window readout.

Replaces the torch reference's ``logits.topk(512)`` + prefix sums with a single-pass Newton / exact active-set
quadratic root finder in registers followed by the ``(2r+1)^2`` window readout around the argmax.

For ``y_j = (z_j - z_max) / 2 <= 0``, the 1.5-entmax threshold ``tau`` satisfies ``g(tau) = sum_j max(y_j - tau, 0)^2
= 1`` with ``tau in [-1, 0)``. Because ``g`` is continuous, strictly decreasing and convex on ``[-1, 0]``, Newton's
method starting from ``tau = -1`` converges monotonically from below (never overshooting ``tau*``); once the active
set ``S = {j : y_j > tau}`` satisfies ``min_{j in S} y_j >= tau_quad >= tau``, the closed-form quadratic root
``tau_quad = mean - sqrt((1 - sum_{j in S}(y_j - mean)^2) / |S|)`` is the exact 1.5-entmax threshold.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl
from torch import Tensor

from flashim import ops
from flashim.ops._registry import DeviceSpec
from flashim.platforms import Feature


@triton.jit
def _flexglue_entmax_readout_kernel(
    logits_ptr,
    grid_ptr,
    pos_ptr,
    mass_ptr,
    s_b,
    s_n,
    s_m,
    gs_m,
    gs_c,
    N,
    H,
    W,
    M,
    RADIUS: tl.constexpr,
    BLOCK_M: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    b = row // N
    n = row % N
    base = b * s_b + n * s_n

    cols = tl.arange(0, BLOCK_M)
    col_ok = cols < M
    z = tl.load(logits_ptr + base + cols.to(tl.int64) * s_m, mask=col_ok, other=-float("inf")).to(tl.float32)

    vmax = tl.max(z, axis=0)
    # Deterministic smallest-index argmax tie-breaking, matching torch.max / torch.topk.
    idx = tl.min(tl.where(col_ok & (z == vmax), cols, M), axis=0)

    y = tl.where(col_ok, (z - vmax) * 0.5, -2.0)

    # Monotonic safeguarded Newton from tau = -1.0 (g(-1) >= 1) with exact active-set quadratic termination.
    tau = -1.0
    done = False
    for _ in range(14):
        if not done:
            active = y > tau
            c = tl.sum(active.to(tl.float32), axis=0)
            y_act = tl.where(active, y, 0.0)
            s1 = tl.sum(y_act, axis=0)
            s2 = tl.sum(y_act * y_act, axis=0)
            # Check exact quadratic root on the current active set: exact iff every active y_j >= tau_q >= tau.
            mean = s1 / tl.maximum(c, 1.0)
            var_sum = tl.maximum(s2 - s1 * mean, 0.0)
            disc = 1.0 - var_sum
            min_act = tl.min(tl.where(active, y, 1.0), axis=0)
            tau_q = mean - tl.sqrt(tl.maximum(disc, 0.0) / tl.maximum(c, 1.0))
            exact = (disc >= 0.0) & (min_act >= tau_q) & (tau_q >= tau)
            d = tl.where(active, y - tau, 0.0)
            s_d = tl.sum(d, axis=0)
            g = tl.sum(d * d, axis=0)
            step = (g - 1.0) / tl.maximum(2.0 * s_d, 1e-12)
            tau_newton = tau + tl.maximum(step, 0.0)
            tau = tl.where(exact, tau_q, tau_newton)
            done = exact | (tl.abs(g - 1.0) <= 1e-7)

    # Final closed-form refinement on the converged active set (matches _sorted_entmax15_threshold).
    if not done:
        active = y > tau
        c = tl.maximum(tl.sum(active.to(tl.float32), axis=0), 1.0)
        y_act = tl.where(active, y, 0.0)
        s1 = tl.sum(y_act, axis=0)
        s2 = tl.sum(y_act * y_act, axis=0)
        mean = s1 / c
        disc = tl.maximum(1.0 - tl.maximum(s2 - s1 * mean, 0.0), 0.0)
        tau = mean - tl.sqrt(disc / c)

    cy = idx // W
    cx = idx % W
    pos_x = 0.0
    pos_y = 0.0
    p_sum = 0.0
    mass = 0.0
    for dy in tl.static_range(-RADIUS, RADIUS + 1):
        ny = cy + dy
        y_in = (ny >= 0) & (ny < H)
        cny = tl.minimum(tl.maximum(ny, 0), H - 1)
        for dx in tl.static_range(-RADIUS, RADIUS + 1):
            nx = cx + dx
            inside = y_in & (nx >= 0) & (nx < W)
            cnx = tl.minimum(tl.maximum(nx, 0), W - 1)
            nidx = (cny * W + cnx).to(tl.int64)
            z_nb = tl.load(logits_ptr + base + nidx * s_m).to(tl.float32)
            diff = tl.maximum((z_nb - vmax) * 0.5 - tau, 0.0)
            p = diff * diff
            gx = tl.load(grid_ptr + nidx * gs_m).to(tl.float32)
            gy = tl.load(grid_ptr + nidx * gs_m + gs_c).to(tl.float32)
            pos_x += p * gx
            pos_y += p * gy
            p_sum += p
            mass += tl.where(inside, p, 0.0)

    inv_sum = 1.0 / p_sum
    tl.store(pos_ptr + row * 2, pos_x * inv_sum)
    tl.store(pos_ptr + row * 2 + 1, pos_y * inv_sum)
    tl.store(mass_ptr + row, mass)


def flexglue_entmax_readout(
    logits: Tensor, grid: Tensor, hw: tuple[int, int], radius: int = 1, k: int = 512
) -> tuple[Tensor, Tensor]:
    b, n, m = logits.shape
    h, w = hw
    pos = torch.empty((b, n, 2), device=logits.device, dtype=torch.float32)
    mass = torch.empty((b, n), device=logits.device, dtype=torch.float32)
    if b * n == 0:
        return pos.to(logits.dtype), mass.to(logits.dtype)
    block_m = max(16, triton.next_power_of_2(m))
    num_warps = 8 if block_m >= 2048 else (4 if block_m >= 256 else 2)
    _flexglue_entmax_readout_kernel[(b * n,)](
        logits,
        grid,
        pos,
        mass,
        logits.stride(0),
        logits.stride(1),
        logits.stride(2),
        grid.stride(0),
        grid.stride(1),
        n,
        h,
        w,
        m,
        RADIUS=radius,
        BLOCK_M=block_m,
        num_warps=num_warps,
    )
    return pos.to(logits.dtype), mass.to(logits.dtype)


def _eligible(logits: Tensor, grid: Tensor, hw: tuple[int, int], radius: int = 1, k: int = 512) -> bool:
    h, w = hw
    m = h * w
    return (
        not torch.is_grad_enabled()
        and logits.ndim == 3
        and logits.shape[-1] == m
        and 1 <= m <= 8192
        and logits.dtype in (torch.float32, torch.bfloat16, torch.float16)
        and grid.ndim == 2
        and grid.shape == (m, 2)
        and grid.device == logits.device
        and radius >= 0
        and radius <= 4
    )


ops.flexglue_entmax_readout.register(
    "triton", device=DeviceSpec("cuda"), requires=[Feature.TRITON], when=_eligible
)(flexglue_entmax_readout)
