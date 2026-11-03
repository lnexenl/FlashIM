"""LayerNorm statistics with the reduction tree of ATen's ``vectorized_layer_norm_kernel`` (torch 2.11, CUDA, BF16 / FP16 /
FP32 rows with ``N % 4 == 0``), so a fused kernel's normalized output equals ``nn.LayerNorm``'s bit for bit.

ATen runs 128 threads per row (a ``(32, 4)`` block): thread ``t`` reads the 4-vectors ``t, t + 128, ...``, folds each
element into a per-thread Welford state (``mean += delta * (1 / n)`` with the reciprocal correctly rounded, the
multiply-adds contracted into FMAs as nvcc emits them), then the 128 states are combined by a shuffle-down tree within
each warp and a halving tree over the four warps. ``tl.reduce`` with the Welford combine below follows the same tree
(butterfly 16 .. 1 in a warp, then over the warps; measured identical to ATen on every ViT / matcher LayerNorm input
tried, 0 mismatches). The combine is symmetric because every level merges equal counts, which holds for ``N`` a multiple
of 512. Measured and discarded: emulating the tree level by level with reshapes (also exact, 4-7x slower than torch's
LayerNorm), 256 threads (1e-5 of the outputs differ).

Re-expression of ATen's arithmetic (PyTorch, BSD-3-Clause; see THIRD_PARTY_NOTICES.md), no source copied.

Kernels using these must launch with ``enable_fp_fusion=False`` (the FMAs are explicit).
"""

from __future__ import annotations

import triton
import triton.language as tl
from triton.language.extra import libdevice

THREADS = 128  # ATen's threads per row
EXACT_MULTIPLE = 4 * THREADS  # N must be a multiple of this for the equal-count combines


@triton.jit
def _welford_combine(m1, s1, c1, m2, s2, c2):
    """ATen ``cuWelfordCombine(dataB = 1, dataA = 2)``: the operand roles matter for the FMA's rounding (the first
    state is the thread's own, the second the shuffled one)."""
    delta = m1 - m2
    count = c2 + c1
    coef = tl.math.div_rn(tl.full(count.shape, 1.0, tl.float32), tl.where(count > 0, count, 1.0))
    na = c2 * coef
    nb = c1 * coef
    mean = tl.fma(na, m2, nb * m1)
    sigma2 = tl.fma(delta * delta * c2, nb, s2 + s1)
    ok = count > 0
    return tl.where(ok, mean, 0.0), tl.where(ok, sigma2, 0.0), count


@triton.jit
def _welford_online(m, s, c, value, active):
    """ATen ``cuWelfordOnlineSum``."""
    delta = value - m
    count = c + 1.0
    mean = tl.fma(delta, tl.math.div_rn(tl.full(count.shape, 1.0, tl.float32), count), m)
    sigma2 = tl.fma(delta, value - mean, s)
    return tl.where(active, mean, m), tl.where(active, sigma2, s), tl.where(active, count, c)


@triton.jit
def _split4(v):
    """``[T, 4]`` -> its four columns ``[T]``."""
    a, b = tl.split(tl.reshape(v, (v.shape[0], 2, 2)))
    a0, a1 = tl.split(a)
    b0, b1 = tl.split(b)
    return a0, b0, a1, b1


@triton.jit
def torch_layernorm_stats(x_ptr, row_base, N: tl.constexpr, T: tl.constexpr, ITERS: tl.constexpr):
    """``(mean, biased variance)`` of the row at ``x_ptr + row_base`` as ATen computes them (``T`` = ``THREADS``)."""
    t = tl.arange(0, T)
    k = tl.arange(0, 4)
    m = tl.zeros([T], tl.float32)
    s = tl.zeros([T], tl.float32)
    c = tl.zeros([T], tl.float32)
    for j in tl.static_range(ITERS):
        vec = t + j * T
        active = vec < N // 4
        v = tl.load(x_ptr + row_base + vec[:, None] * 4 + k[None, :], mask=active[:, None], other=0.0).to(tl.float32)
        x0, x1, x2, x3 = _split4(v)
        m, s, c = _welford_online(m, s, c, x0, active)
        m, s, c = _welford_online(m, s, c, x1, active)
        m, s, c = _welford_online(m, s, c, x2, active)
        m, s, c = _welford_online(m, s, c, x3, active)
    mean, sigma2, _ = tl.reduce((m, s, c), 0, _welford_combine)
    return mean, tl.math.div_rn(sigma2, N * 1.0)


@triton.jit
def torch_layernorm_apply(x, weight, bias, mean, var, eps):
    """ATen's affine step on FP32 values: ``gamma * (rstd * (x - mean)) + beta`` (contracted), rstd = ``rsqrtf``."""
    rstd = libdevice.rsqrt(var + eps)
    return tl.fma(weight, rstd * (x - mean), bias)
