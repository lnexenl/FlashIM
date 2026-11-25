"""Tests for fused LayerNorm + add / cat (+ quantize) and aligned upsample_add ops."""

from __future__ import annotations

import pytest
import torch

from flashim import ops


def _cuda_available() -> bool:
    return torch.cuda.is_available()


def test_layernorm_ops_cpu_reference():
    torch.manual_seed(0)
    x = torch.randn(2, 17, 64)
    v = torch.randn(2, 17, 64)
    w = torch.randn(64)
    b = torch.randn(64)
    scale = torch.tensor(0.05)
    exp = torch.zeros(128 // 32, dtype=torch.int8)

    out_add = ops.layernorm_add(x, v, w, b, 1e-5)
    assert out_add.shape == x.shape
    out_cat = ops.layernorm_cat(x, v, w, b, 1e-5)
    assert out_cat.shape == (2, 17, 128)
    out_i8 = ops.layernorm_cat_quantize_int8(x, v, w, b, 1e-5, scale)
    assert out_i8.values.shape == (2, 17, 128) and out_i8.values.dtype == torch.int8
    out_dyn, s_dyn = ops.layernorm_cat_quantize_int8_rows(x, v, w, b, 1e-5)
    assert out_dyn.shape == (2, 17, 128) and s_dyn.shape == (2, 17, 1)
    out_fp8 = ops.layernorm_cat_quantize_mxfp8(x, v, w, b, 1e-5, exp)
    assert out_fp8.values.shape == (2, 17, 128)


@pytest.mark.cuda
@pytest.mark.parametrize(("rows", "c"), [(1, 64), (17, 128), (8190, 256), (513, 192)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_layernorm_add_and_cat_triton(rows, c, dtype):
    torch.manual_seed(7)
    x = torch.randn(1, rows, c, device="cuda", dtype=dtype)
    v = torch.randn(1, rows, c, device="cuda", dtype=dtype)
    w = torch.randn(c, device="cuda", dtype=dtype)
    b = torch.randn(c, device="cuda", dtype=dtype)
    scale = torch.tensor(0.05, device="cuda", dtype=torch.float32)

    assert ops.layernorm_add.explain(x, v, w, b, 1e-5)["triton"] == "selected"
    tol = 2e-2 if dtype == torch.bfloat16 else 1e-5
    torch.testing.assert_close(
        ops.layernorm_add.get("triton")(x, v, w, b, 1e-5),
        ops.layernorm_add.get("reference")(x, v, w, b, 1e-5),
        atol=tol,
        rtol=tol,
    )
    torch.testing.assert_close(
        ops.layernorm_cat.get("triton")(x, v, w, b, 1e-5),
        ops.layernorm_cat.get("reference")(x, v, w, b, 1e-5),
        atol=tol,
        rtol=tol,
    )

    f_i8 = ops.layernorm_cat_quantize_int8.get("triton")(x, v, w, b, 1e-5, scale)
    r_i8 = ops.layernorm_cat_quantize_int8.get("reference")(x, v, w, b, 1e-5, scale)
    assert (f_i8.values.int() - r_i8.values.int()).abs().max().item() <= 1

    f_dyn, fs_dyn = ops.layernorm_cat_quantize_int8_rows.get("triton")(x, v, w, b, 1e-5)
    r_dyn, rs_dyn = ops.layernorm_cat_quantize_int8_rows.get("reference")(x, v, w, b, 1e-5)
    assert (f_dyn.int() - r_dyn.int()).abs().max().item() <= 1
    torch.testing.assert_close(fs_dyn, rs_dyn, atol=1e-3, rtol=1e-3)

    cs = torch.rand(2 * c, device="cuda", dtype=torch.float32) * 1.5 + 0.5
    f_cs, fs_cs = ops.layernorm_cat_quantize_int8_rows.get("triton")(x, v, w, b, 1e-5, channel_scale=cs)
    r_cs, rs_cs = ops.layernorm_cat_quantize_int8_rows.get("reference")(x, v, w, b, 1e-5, channel_scale=cs)
    assert (f_cs.int() - r_cs.int()).abs().max().item() <= 1
    torch.testing.assert_close(fs_cs, rs_cs, atol=1e-3, rtol=1e-3)

    if c % 32 == 0 and torch.cuda.get_device_capability()[0] >= 9 or (
        c % 32 == 0 and torch.cuda.get_device_capability() == (8, 9)
    ):
        exp = torch.zeros(2 * c // 32, device="cuda", dtype=torch.int8)
        f_fp8 = ops.layernorm_cat_quantize_mxfp8.get("triton")(x, v, w, b, 1e-5, exp)
        r_fp8 = ops.layernorm_cat_quantize_mxfp8.get("reference")(x, v, w, b, 1e-5, exp)
        torch.testing.assert_close(f_fp8.values.float(), r_fp8.values.float(), atol=0.5, rtol=0.15)


@pytest.mark.cuda
@pytest.mark.parametrize(("c", "h", "w"), [(256, 180, 210), (128, 360, 420), (196, 65, 97)])
def test_upsample_add_align_corners_triton(c, h, w):
    torch.manual_seed(3)
    coarse = torch.randn(1, c, h // 2, w // 2, device="cuda", dtype=torch.bfloat16).to(memory_format=torch.channels_last)
    delta = torch.randn(1, c, h, w, device="cuda", dtype=torch.bfloat16).to(memory_format=torch.channels_last)
    assert ops.upsample_add.explain(coarse, delta, align_corners=True, out_dtype=torch.bfloat16)["triton"] == "selected"
    fast = ops.upsample_add.get("triton")(coarse, delta, align_corners=True, out_dtype=torch.bfloat16)
    ref = ops.upsample_add.get("reference")(coarse, delta, align_corners=True, out_dtype=torch.bfloat16)
    torch.testing.assert_close(fast, ref, atol=2e-2, rtol=2e-2)
