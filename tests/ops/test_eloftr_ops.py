"""Correctness tests for EfficientLoFTR fused ops vs their torch references."""

from __future__ import annotations

import pytest
import torch

from flashim import ops
from flashim.quant.formats.int8 import safe_scale
from flashim.quant.formats.mx import scale_exponent_from_amax


@pytest.mark.cuda
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("layout", ["nchw", "nhwc"])
@pytest.mark.parametrize("window_size,pad", [(8, 0), (10, 1)])
@pytest.mark.parametrize("hc,wc,m", [(4, 5, 0), (4, 5, 7), (80, 104, 571)])
def test_eloftr_upsample_gather_windows_matches_reference(dtype, layout, window_size, pad, hc, wc, m):
    torch.manual_seed(0)
    feat = torch.randn(1, 64, hc * 4, wc * 4, device="cuda", dtype=dtype)
    if layout == "nhwc":
        feat = feat.to(memory_format=torch.channels_last)
    ids = torch.randperm(hc * wc, device="cuda")[:m]

    ref = ops.eloftr_upsample_gather_windows.get("reference")(feat, ids, hc, wc, window_size, pad)
    out = ops.eloftr_upsample_gather_windows(feat, ids, hc, wc, window_size, pad)
    assert ops.eloftr_upsample_gather_windows.explain(feat, ids, hc, wc, window_size, pad)["triton"] == "selected"
    assert out.shape == ref.shape and out.dtype == ref.dtype
    if m > 0:
        tol = 1e-5 if dtype == torch.float32 else 2e-2
        assert (out.float() - ref.float()).abs().max().item() <= tol


@pytest.mark.cuda
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("h,w,c", [(12, 16, 64), (17, 23, 96), (80, 104, 512)])
def test_eloftr_upsample_add_leaky_relu_dense(dtype, h, w, c):
    torch.manual_seed(1)
    x_proj = torch.randn(1, h, w, c, device="cuda", dtype=dtype)
    m_proj = torch.randn(1, max(1, h // 4), max(1, w // 4), c, device="cuda", dtype=dtype)

    ref = ops.eloftr_upsample_add_leaky_relu.get("reference")(x_proj, m_proj, 0.01)
    out = ops.eloftr_upsample_add_leaky_relu(x_proj, m_proj, 0.01)
    assert ops.eloftr_upsample_add_leaky_relu.explain(x_proj, m_proj, 0.01)["triton"] == "selected"
    assert out.shape == ref.shape and out.dtype == ref.dtype
    tol = 1e-5 if dtype == torch.float32 else 2e-2
    assert (out.float() - ref.float()).abs().max().item() <= tol


@pytest.mark.cuda
@pytest.mark.parametrize("mode", ["int8", "mxfp8"])
@pytest.mark.parametrize("h,w,c", [(16, 20, 128), (80, 104, 512)])
def test_eloftr_upsample_add_leaky_relu_quantized(mode, h, w, c):
    torch.manual_seed(2)
    x_proj = torch.randn(1, h, w, c, device="cuda", dtype=torch.bfloat16)
    m_proj = torch.randn(1, h // 4, w // 4, c, device="cuda", dtype=torch.bfloat16)

    if mode == "int8":
        scale = safe_scale(torch.tensor(6.0 / 127.0, device="cuda"))
        ref = ops.eloftr_upsample_add_leaky_relu.get("reference")(x_proj, m_proj, 0.01, int8_scale=scale)
        out = ops.eloftr_upsample_add_leaky_relu(x_proj, m_proj, 0.01, int8_scale=scale)
        assert (
            ops.eloftr_upsample_add_leaky_relu.explain(x_proj, m_proj, 0.01, int8_scale=scale)["triton"]
            == "selected"
        )
        diff = (out.values.int() - ref.values.int()).abs()
        assert diff.max().item() <= 1
        assert (diff > 0).float().mean().item() < 1e-3
    else:
        exp = scale_exponent_from_amax(torch.full((c // 32,), 6.0, device="cuda")).to(torch.int8)
        ref = ops.eloftr_upsample_add_leaky_relu.get("reference")(x_proj, m_proj, 0.01, mxfp8_exponent=exp)
        out = ops.eloftr_upsample_add_leaky_relu(x_proj, m_proj, 0.01, mxfp8_exponent=exp)
        assert (
            ops.eloftr_upsample_add_leaky_relu.explain(x_proj, m_proj, 0.01, mxfp8_exponent=exp)["triton"]
            == "selected"
        )
        mismatch = (out.values.view(torch.uint8) != ref.values.view(torch.uint8)).float().mean().item()
        assert mismatch < 1e-3


@pytest.mark.cuda
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("rows,c", [(1, 64), (17, 192), (8320, 256)])
def test_eloftr_post_norm_add_matches_reference(dtype, rows, c):
    torch.manual_seed(3)
    m = torch.randn(1, rows, c, device="cuda", dtype=dtype)
    x = torch.randn(1, rows, c, device="cuda", dtype=dtype)
    w = torch.randn(c, device="cuda", dtype=dtype)
    b = torch.randn(c, device="cuda", dtype=dtype)

    ref = ops.eloftr_post_norm_add.get("reference")(m, x, w, b, 1e-5)
    out = ops.eloftr_post_norm_add(m, x, w, b, 1e-5)
    assert ops.eloftr_post_norm_add.explain(m, x, w, b, 1e-5)["triton"] == "selected"
    tol = 1e-5 if dtype == torch.float32 else 2e-2
    assert (out.float() - ref.float()).abs().max().item() <= tol


@pytest.mark.cuda
@pytest.mark.parametrize("kind", ["int8", "mxfp8"])
def test_conv2d_residual_scale_matches_reference(kind):
    from flashim.quant.methods.conv2d import StaticInt8Conv2d, StaticMXFP8Conv2d

    torch.manual_seed(4)
    conv = torch.nn.Conv2d(64, 64, 3, padding=1, bias=True, device="cuda", dtype=torch.bfloat16)
    x = torch.randn(2, 64, 24, 32, device="cuda", dtype=torch.bfloat16)
    id_scale = torch.randn(64, device="cuda", dtype=torch.float32) * 5.0

    if kind == "int8":
        s = safe_scale(x.float().abs().max() / 127.0)
        qconv = StaticInt8Conv2d.from_conv(conv, s)
    else:
        exp = scale_exponent_from_amax(
            x.permute(0, 2, 3, 1).float().abs().reshape(-1, 2, 32).amax(dim=(0, 2))
        ).to(torch.int8)
        qconv = StaticMXFP8Conv2d.from_conv(conv, exp)

    fused = qconv(x, relu=True, residual=x, residual_scale=id_scale)
    ref = torch.relu(qconv(x, relu=False).float() + x.float() * id_scale[None, :, None, None]).to(x.dtype)
    torch.testing.assert_close(fused, ref, rtol=1e-2, atol=2e-2)


@pytest.mark.cuda
@pytest.mark.parametrize("mode", ["dense", "int8", "int8_rows", "mxfp8"])
@pytest.mark.parametrize("b,h,w,c", [(1, 80, 104, 256), (2, 80, 104, 256)])
def test_eloftr_agg_and_pool_norm_matches_reference(mode, b, h, w, c):
    torch.manual_seed(5)
    x = torch.randn(b, h, w, c, device="cuda", dtype=torch.bfloat16)
    agg_w = torch.randn(c, 1, 4, 4, device="cuda", dtype=torch.bfloat16)
    ln_w = torch.randn(c, device="cuda", dtype=torch.bfloat16)
    ln_b = torch.randn(c, device="cuda", dtype=torch.bfloat16)

    kw = {}
    if mode == "int8":
        kw["int8_scale"] = safe_scale(torch.tensor(4.0 / 127.0, device="cuda"))
    elif mode == "int8_rows":
        kw["int8_rows"] = True
    elif mode == "mxfp8":
        kw["mxfp8_exponent"] = scale_exponent_from_amax(torch.full((c // 32,), 4.0, device="cuda")).to(torch.int8)

    for op, arg in ((ops.eloftr_agg_norm, agg_w), (ops.eloftr_pool_norm, 4)):
        ref = op.get("reference")(x, arg, ln_w, ln_b, 1e-5, **kw)
        out = op(x, arg, ln_w, ln_b, 1e-5, **kw)
        assert op.explain(x, arg, ln_w, ln_b, 1e-5, **kw)["triton"] == "selected"
        if mode == "dense":
            assert (out.float() - ref.float()).abs().max().item() <= 3e-2
        elif mode == "int8":
            diff = (out.values.int() - ref.values.int()).abs()
            assert diff.max().item() <= 1
            assert (diff > 0).float().mean().item() < 2e-3
        elif mode == "int8_rows":
            ov, os_ = out
            rv, rs = ref
            assert (os_ - rs).abs().max().item() <= 2e-3
            diff = (ov.int() - rv.int()).abs()
            assert diff.max().item() <= 2
            assert (diff > 0).float().mean().item() < 5e-3
        else:
            mismatch = (out.values.view(torch.uint8) != ref.values.view(torch.uint8)).float().mean().item()
            assert mismatch < 2e-3


@pytest.mark.cuda
@pytest.mark.parametrize("mode", ["int8", "mxfp8"])
@pytest.mark.parametrize("b,h,w,c", [(2, 160, 208, 256), (2, 320, 416, 128)])
def test_eloftr_leaky_relu_quantize_matches_reference(mode, b, h, w, c):
    torch.manual_seed(6)
    x = torch.randn(b, h, w, c, device="cuda", dtype=torch.bfloat16)
    if mode == "int8":
        s = safe_scale(torch.tensor(6.0 / 127.0, device="cuda"))
        ref = ops.eloftr_leaky_relu_quantize.get("reference")(x, 0.01, int8_scale=s)
        out = ops.eloftr_leaky_relu_quantize(x, 0.01, int8_scale=s)
        assert ops.eloftr_leaky_relu_quantize.explain(x, 0.01, int8_scale=s)["triton"] == "selected"
        assert (out.values.int() - ref.values.int()).abs().max().item() <= 1
    else:
        exp = scale_exponent_from_amax(torch.full((c // 32,), 6.0, device="cuda")).to(torch.int8)
        ref = ops.eloftr_leaky_relu_quantize.get("reference")(x, 0.01, mxfp8_exponent=exp)
        out = ops.eloftr_leaky_relu_quantize(x, 0.01, mxfp8_exponent=exp)
        assert ops.eloftr_leaky_relu_quantize.explain(x, 0.01, mxfp8_exponent=exp)["triton"] == "selected"
        assert (out.values.view(torch.uint8) != ref.values.view(torch.uint8)).float().mean().item() < 1e-3

