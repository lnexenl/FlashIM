"""Correctness tests (real call-site shapes + edge/odd shapes) for Triton ops used by XoFTR:
``sdpa`` (short_window, symmetric & asymmetric), ``bidirectional_cross_attention``,
``elu_linear_attention``, ``gather_windows``, ``union_softmax_match``,
``batched_dual_softmax_argmax``, ``residual_layernorm``, ``cat_layernorm``,
``pos_cat_layernorm``, and ``cat_gelu_mlp``."""

from __future__ import annotations

import pytest
import torch

from flashim import ops

pytestmark = [pytest.mark.cuda]


@pytest.mark.parametrize(
    ("b", "h", "l", "d"),
    [
        (4922, 7, 10, 28),  # XoFTR self_attn_m real shape
        (4922, 8, 34, 16),  # XoFTR self_attn_f real shape
        (64, 7, 9, 28),  # smaller batch, odd L, non-pow2 D
        (128, 4, 17, 12),  # non-pow2 L and D
        (64, 4, 64, 32),  # max eligible L and D
    ],
)
def test_short_window_sdpa(b, h, l, d):
    torch.manual_seed(0)
    # Non-contiguous strided view matching WindowAttention's permute output.
    qkv = torch.randn(b, l, 3, h, d, device="cuda", dtype=torch.bfloat16)
    q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(0)
    with torch.inference_mode():
        assert ops.sdpa.explain(q, k, v)["triton_short_window"] == "selected"
        got = ops.sdpa(q, k, v)
        ref = ops.sdpa.get("reference")(q, k, v)
    assert got.shape == ref.shape and got.dtype == ref.dtype
    assert (got.float() - ref.float()).abs().max().item() < 2e-2


@pytest.mark.parametrize(
    ("m", "l0", "l1", "h", "d"),
    [
        (2461, 9, 9, 7, 28),  # XoFTR cross_attn_m real shape
        (2461, 25, 25, 8, 16),  # XoFTR cross_attn_f real shape
        (17, 9, 13, 5, 20),  # rectangular odd L0 != L1, non-pow2 H/D
        (0, 25, 25, 8, 16),  # empty matches corner case
    ],
)
def test_bidirectional_cross_attention(m, l0, l1, h, d):
    torch.manual_seed(1)
    c = h * d
    qk0 = torch.randn(m, l0, c, device="cuda", dtype=torch.bfloat16)
    qk1 = torch.randn(m, l1, c, device="cuda", dtype=torch.bfloat16)
    v0 = torch.randn(m, l0, c, device="cuda", dtype=torch.bfloat16)
    v1 = torch.randn(m, l1, c, device="cuda", dtype=torch.bfloat16)
    with torch.inference_mode():
        assert ops.bidirectional_cross_attention.explain(qk0, qk1, v0, v1, h)["triton"] == "selected"
        o0, o1 = ops.bidirectional_cross_attention(qk0, qk1, v0, v1, h)
        r0, r1 = ops.bidirectional_cross_attention.get("reference")(qk0, qk1, v0, v1, h)
    assert o0.shape == r0.shape and o1.shape == r1.shape
    if m > 0:
        assert (o0.float() - r0.float()).abs().max().item() < 2e-2
        assert (o1.float() - r1.float()).abs().max().item() < 2e-2


@pytest.mark.parametrize(
    ("b", "l", "s", "h", "d", "dtype", "gemm_dtype"),
    [
        (1, 4800, 4800, 8, 32, torch.bfloat16, torch.bfloat16),  # XoFTR coarse real shape (bf16)
        (2, 4800, 4800, 8, 32, torch.bfloat16, torch.bfloat16),  # batched self-attn real shape
        (1, 4560, 4800, 8, 32, torch.bfloat16, torch.bfloat16),  # unequal L != S (sacre_coeur)
        (1, 317, 211, 8, 32, torch.float32, None),  # odd sizes, FP32 reference
        (1, 129, 137, 4, 24, torch.bfloat16, torch.bfloat16),  # non-pow2 D
    ],
)
def test_elu_linear_attention(b, l, s, h, d, dtype, gemm_dtype):
    torch.manual_seed(2)
    q = torch.randn(b, l, h, d, device="cuda", dtype=dtype) * 0.5
    k = torch.randn(b, s, h, d, device="cuda", dtype=dtype) * 0.5
    v = torch.randn(b, s, h, d, device="cuda", dtype=dtype) * 0.5
    with torch.inference_mode():
        assert ops.elu_linear_attention.explain(q, k, v, 1e-6, gemm_dtype)["triton"] == "selected"
        got = ops.elu_linear_attention(q, k, v, 1e-6, gemm_dtype)
        ref = ops.elu_linear_attention.get("reference")(q, k, v, 1e-6, gemm_dtype)
    assert got.shape == ref.shape and got.dtype == ref.dtype
    tol = 2e-2 if dtype == torch.bfloat16 else 1e-4
    assert (got.float() - ref.float()).abs().max().item() < tol


@pytest.mark.parametrize(
    ("c", "hc", "wc", "win", "stride", "m", "cl"),
    [
        (196, 60, 80, 3, 2, 2461, False),  # XoFTR medium NCHW
        (196, 60, 80, 3, 2, 2461, True),  # XoFTR medium NHWC
        (128, 60, 80, 5, 4, 2461, False),  # XoFTR fine NCHW
        (128, 57, 80, 5, 4, 950, True),  # sacre_coeur NHWC
        (128, 12, 15, 5, 4, 180, False),  # includes every border cell (zero-padding check)
        (128, 12, 15, 5, 4, 0, False),  # empty ids
    ],
)
def test_gather_windows(c, hc, wc, win, stride, m, cl):
    torch.manual_seed(3)
    feat = torch.randn(1, c, hc * stride, wc * stride, device="cuda", dtype=torch.bfloat16)
    if cl:
        feat = feat.to(memory_format=torch.channels_last)
    ids = torch.arange(m, device="cuda", dtype=torch.long) if m <= hc * wc else torch.randint(0, hc * wc, (m,), device="cuda")
    with torch.inference_mode():
        assert ops.gather_windows.explain(feat, ids, wc, win, stride)["triton"] == "selected"
        got = ops.gather_windows(feat, ids, wc, win, stride)
        ref = ops.gather_windows.get("reference")(feat, ids, wc, win, stride)
    assert got.shape == ref.shape and got.dtype == ref.dtype
    assert torch.equal(got, ref)  # bit-identical!


@pytest.mark.parametrize(("l", "s"), [(4800, 4800), (4560, 4800), (137, 211), (17, 19)])
def test_union_softmax_match(l, s):
    torch.manual_seed(4)
    sim = torch.randn(l, s, device="cuda", dtype=torch.float32) * 3.0
    with torch.inference_mode():
        assert ops.union_softmax_match.explain(sim)["triton"] == "selected"
        rp, rj, cp, ci = ops.union_softmax_match(sim)
        rrp, rrj, rcp, rci = ops.union_softmax_match.get("reference")(sim)
    assert torch.equal(rj, rrj)
    assert torch.equal(ci, rci)
    assert (rp - rrp).abs().max().item() < 1e-6
    assert (cp - rcp).abs().max().item() < 1e-6


@pytest.mark.parametrize(("rows", "c", "has_res"), [(167348, 128, True), (49220, 196, False), (9600, 256, True), (17, 131, True)])
def test_residual_layernorm(rows, c, has_res):
    torch.manual_seed(5)
    x = torch.randn(rows, c, device="cuda", dtype=torch.bfloat16)
    res = torch.randn(rows, c, device="cuda", dtype=torch.bfloat16) if has_res else None
    w = torch.randn(c, device="cuda", dtype=torch.bfloat16)
    b = torch.randn(c, device="cuda", dtype=torch.bfloat16)
    with torch.inference_mode():
        assert ops.residual_layernorm.explain(x, res, w, b, 1e-5)["triton"] == "selected"
        got = ops.residual_layernorm(x, res, w, b, 1e-5)
        ref = ops.residual_layernorm.get("reference")(x, res, w, b, 1e-5)
    assert got.shape == ref.shape and got.dtype == ref.dtype
    torch.testing.assert_close(got.float(), ref.float(), atol=2e-2, rtol=1e-2)


@pytest.mark.parametrize(("rows", "c"), [(9600, 256), (4560, 256), (33, 196)])
def test_cat_layernorm(rows, c):
    torch.manual_seed(6)
    left = torch.randn(rows, c, device="cuda", dtype=torch.bfloat16)
    right = torch.randn(rows, c, device="cuda", dtype=torch.bfloat16)
    w = torch.randn(c, device="cuda", dtype=torch.bfloat16)
    b = torch.randn(c, device="cuda", dtype=torch.bfloat16)
    with torch.inference_mode():
        assert ops.cat_layernorm.explain(left, right, w, b, 1e-5)["triton"] == "selected"
        got = ops.cat_layernorm(left, right, w, b, 1e-5)
        ref = ops.cat_layernorm.get("reference")(left, right, w, b, 1e-5)
    assert got.shape == ref.shape and got.dtype == ref.dtype
    assert torch.equal(got[:, :c], left)
    assert (got[:, c:].float() - ref[:, c:].float()).abs().max().item() < 3e-2


@pytest.mark.parametrize(("m", "w", "c"), [(2461, 25, 128), (19, 9, 64), (0, 25, 128)])
def test_batched_dual_softmax_argmax(m, w, c):
    torch.manual_seed(7)
    f0 = torch.randn(m, w, c, device="cuda", dtype=torch.bfloat16)
    f1 = torch.randn(m, w, c, device="cuda", dtype=torch.bfloat16)
    scale = 1.0 / (c * 0.1)
    with torch.inference_mode():
        assert ops.batched_dual_softmax_argmax.explain(f0, f1, scale)["triton"] == "selected"
        conf, bi, bj = ops.batched_dual_softmax_argmax(f0, f1, scale)
        rconf, rbi, rbj = ops.batched_dual_softmax_argmax.get("reference")(f0, f1, scale)
    if m > 0:
        assert torch.equal(bi, rbi) and torch.equal(bj, rbj)
        assert (conf - rconf).abs().max().item() < 2e-3


@pytest.mark.parametrize(("b", "h", "lq", "lkv", "d"), [(128, 8, 25, 34, 16), (128, 7, 9, 10, 28)])
def test_short_window_sdpa_asymmetric(b, h, lq, lkv, d):
    torch.manual_seed(8)
    q = torch.randn(b, h, lq, d, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(b, h, lkv, d, device="cuda", dtype=torch.bfloat16)
    v = torch.randn(b, h, lkv, d, device="cuda", dtype=torch.bfloat16)
    with torch.inference_mode():
        assert ops.sdpa.explain(q, k, v)["triton_short_window"] == "selected"
        got = ops.sdpa(q, k, v)
        ref = ops.sdpa.get("reference")(q, k, v)
    assert got.shape == ref.shape and got.dtype == ref.dtype
    assert (got.float() - ref.float()).abs().max().item() < 2e-2


@pytest.mark.parametrize(
    ("rows", "c", "h", "with_norm"),
    [(2500, 128, 256, True), (2500, 196, 392, True), (37, 128, 256, False), (0, 128, 256, True)],
)
def test_cat_gelu_mlp(rows, c, h, with_norm):
    torch.manual_seed(9)
    a = torch.randn(rows, c, device="cuda", dtype=torch.bfloat16) * 0.5
    b = torch.randn(rows, c, device="cuda", dtype=torch.bfloat16) * 0.5
    w1 = torch.randn(h, 2 * c, device="cuda", dtype=torch.bfloat16) * (1.0 / (2 * c) ** 0.5)
    b1 = torch.randn(h, device="cuda", dtype=torch.bfloat16) * 0.1
    w2 = torch.randn(c, h, device="cuda", dtype=torch.bfloat16) * (1.0 / h**0.5)
    b2 = torch.randn(c, device="cuda", dtype=torch.bfloat16) * 0.1
    nw = torch.randn(c, device="cuda", dtype=torch.bfloat16) if with_norm else None
    nb = torch.randn(c, device="cuda", dtype=torch.bfloat16) if with_norm else None
    with torch.inference_mode():
        assert ops.cat_gelu_mlp.explain(a, b, w1, b1, w2, b2, nw, nb, 1e-5)["triton"] == "selected"
        got = ops.cat_gelu_mlp(a, b, w1, b1, w2, b2, nw, nb, 1e-5)
        ref = ops.cat_gelu_mlp.get("reference")(a, b, w1, b1, w2, b2, nw, nb, 1e-5)
    assert got.shape == ref.shape and got.dtype == ref.dtype
    if rows > 0:
        torch.testing.assert_close(got.float(), ref.float(), atol=5e-2, rtol=3e-2)


@pytest.mark.parametrize(
    ("b", "l0", "l1", "c", "has_pre_pos", "return_all"),
    [(128, 25, 9, 128, True, False), (128, 9, 1, 196, False, False), (19, 25, 9, 128, True, True), (0, 25, 9, 128, True, False)],
)
def test_pos_cat_layernorm(b, l0, l1, c, has_pre_pos, return_all):
    torch.manual_seed(10)
    x = torch.randn(b, l0, c, device="cuda", dtype=torch.bfloat16)
    x_pre = torch.randn(b, l1, c, device="cuda", dtype=torch.bfloat16)
    pos = torch.randn(1, l0, c, device="cuda", dtype=torch.bfloat16)
    pos_pre = torch.randn(1, l1, c, device="cuda", dtype=torch.bfloat16) if has_pre_pos else None
    w = torch.randn(c, device="cuda", dtype=torch.bfloat16)
    bias = torch.randn(c, device="cuda", dtype=torch.bfloat16)
    with torch.inference_mode():
        assert ops.pos_cat_layernorm.explain(x, x_pre, pos, pos_pre, w, bias, 1e-5, return_all_pos=return_all)["triton"] == "selected"
        gp, gn = ops.pos_cat_layernorm(x, x_pre, pos, pos_pre, w, bias, 1e-5, return_all_pos=return_all)
        rp, rn = ops.pos_cat_layernorm.get("reference")(x, x_pre, pos, pos_pre, w, bias, 1e-5, return_all_pos=return_all)
    assert gp.shape == rp.shape and gn.shape == rn.shape
    if b > 0:
        torch.testing.assert_close(gp.float(), rp.float(), atol=1e-2, rtol=1e-2)
        torch.testing.assert_close(gn.float(), rn.float(), atol=2e-2, rtol=1e-2)


