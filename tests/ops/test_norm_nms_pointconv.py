"""Correctness tests for ``instance_norm_prelu``, ``rgb_mean_instance_norm``, ``strict_nms_mask``,
``upsample_concat(..., coarse_first=True)`` and the Triton backend of ``conv2d_at_points``."""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from flashim import ops

pytestmark = pytest.mark.cuda
DEV = "cuda"


@pytest.mark.parametrize("shape", [(1, 16, 448, 600), (1, 80, 896, 1200), (2, 37, 31, 53), (1, 128, 56, 75)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("in_nhwc,out_cl", [(False, False), (True, True), (True, False), (False, True)])
def test_instance_norm_prelu(shape, dtype, in_nhwc, out_cl):
    torch.manual_seed(0)
    fmt = torch.channels_last if in_nhwc else torch.contiguous_format
    x = (torch.randn(shape, device=DEV) * 3.0 + 1.5).to(dtype).contiguous(memory_format=fmt)
    w = (torch.randn(shape[1], device=DEV) * 0.25).to(dtype)
    assert ops.instance_norm_prelu.explain(x, w, out_channels_last=out_cl)["triton"] == "selected"
    fast = ops.instance_norm_prelu.get("triton")(x, w, 1e-5, out_channels_last=out_cl)
    ref = ops.instance_norm_prelu.get("reference")(x, w, 1e-5, out_channels_last=out_cl)
    assert fast.shape == ref.shape and fast.dtype == ref.dtype
    assert fast.is_contiguous(memory_format=torch.channels_last if out_cl else torch.contiguous_format)
    tol = 1e-5 if dtype == torch.float32 else 2e-2
    torch.testing.assert_close(fast.float(), ref.float(), atol=tol, rtol=tol)
    if dtype == torch.bfloat16:
        assert (fast != ref).float().mean().item() < 1e-3


@pytest.mark.parametrize("shape", [(1, 3, 896, 1200), (2, 3, 64, 96), (1, 3, 480, 640)])
@pytest.mark.parametrize("out_dtype", [torch.float32, torch.bfloat16])
def test_rgb_mean_instance_norm_unfold(shape, out_dtype):
    torch.manual_seed(1)
    img = torch.rand(shape, device=DEV, dtype=torch.float32)
    assert ops.rgb_mean_instance_norm.explain(img, 1e-5, dtype=out_dtype, unfold=8)["triton"] == "selected"
    g_fast, u_fast = ops.rgb_mean_instance_norm.get("triton")(img, 1e-5, dtype=out_dtype, unfold=8)
    g_ref, u_ref = ops.rgb_mean_instance_norm.get("reference")(img, 1e-5, dtype=out_dtype, unfold=8)
    assert g_fast.shape == g_ref.shape and u_fast.shape == u_ref.shape
    tol = 1e-5 if out_dtype == torch.float32 else 2e-2
    torch.testing.assert_close(g_fast.float(), g_ref.float(), atol=tol, rtol=tol)
    torch.testing.assert_close(u_fast.float(), u_ref.float(), atol=tol, rtol=tol)


@pytest.mark.parametrize("hw", [(7, 9), (37, 45), (896, 1200)])
@pytest.mark.parametrize("window", [3, 5])
@pytest.mark.parametrize("threshold", [None, 0.0, 0.5])
def test_strict_nms_mask_bit_identical(hw, window, threshold):
    torch.manual_seed(2)
    hm = torch.randn(1, 1, *hw, device=DEV, dtype=torch.float32)
    # Plant ties to verify strict `>` semantics.
    hm[:, :, 3:6, 3:6] = 10.0
    assert ops.strict_nms_mask.explain(hm, window, threshold)["triton"] == "selected"
    fast = ops.strict_nms_mask.get("triton")(hm, window, threshold)
    ref = ops.strict_nms_mask.get("reference")(hm, window, threshold)
    assert torch.equal(fast, ref)


@pytest.mark.parametrize("size,coarse_size", [((37, 45), (19, 23)), ((896, 1200), (448, 600))])
@pytest.mark.parametrize("layout", ["nchw", "nhwc"])
def test_upsample_concat_coarse_first(size, coarse_size, layout):
    torch.manual_seed(3)
    fmt = torch.channels_last if layout == "nhwc" else torch.contiguous_format
    features = torch.randn(1, 16, *size, device=DEV, dtype=torch.bfloat16).contiguous(memory_format=fmt)
    coarse = torch.randn(1, 64, *coarse_size, device=DEV, dtype=torch.bfloat16).contiguous(memory_format=fmt)
    fast = ops.upsample_concat.get("triton")(features, coarse, coarse_first=True)
    ref = ops.upsample_concat.get("reference")(features, coarse, coarse_first=True)
    assert fast.shape == ref.shape and fast.dtype == ref.dtype
    assert torch.equal(fast[:, 64:], ref[:, 64:])
    assert ((fast.float() - ref.float()).abs() <= ref.float().abs() * 2**-7 + 1e-6).all()


@pytest.mark.parametrize("cin,cout,k,npts", [(80, 128, 5, 4096), (24, 16, 5, 300), (33, 19, 3, 77), (16, 32, 1, 0)])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("nhwc", [False, True])
def test_conv2d_at_points_triton(cin, cout, k, npts, dtype, nhwc):
    torch.manual_seed(4)
    fmt = torch.channels_last if nhwc else torch.contiguous_format
    h, w = 64, 80
    x = torch.randn(1, cin, h, w, device=DEV).to(dtype).contiguous(memory_format=fmt)
    weight = torch.randn(cout, cin, k, k, device=DEV, dtype=torch.float32) * 0.1
    bias = torch.randn(cout, device=DEV, dtype=torch.float32)
    points = torch.cat([torch.randint(0, w, (npts, 1)), torch.randint(0, h, (npts, 1))], 1).to(DEV)
    if npts >= 4:
        points[:4] = torch.tensor([[0, 0], [w - 1, h - 1], [0, h - 1], [w - 1, 0]], device=DEV)
    assert ops.conv2d_at_points.explain(x, weight, bias, points)["triton"] == "selected"
    fast = ops.conv2d_at_points.get("triton")(x, weight, bias, points)
    ref = ops.conv2d_at_points.get("reference")(x, weight, bias, points)
    assert fast.shape == ref.shape and fast.dtype == ref.dtype
    if npts > 0:
        dense = F.conv2d(x.double(), weight.double(), bias.double(), padding=k // 2)[0][:, points[:, 1], points[:, 0]].t()
        torch.testing.assert_close(fast, dense.float(), atol=1e-4, rtol=1e-4)


@pytest.mark.parametrize("in_nhwc,out_cl", [(False, False), (True, True), (True, False), (False, True)])
def test_instance_norm_prelu_torch_compile(in_nhwc, out_cl):
    """Regression: Inductor passes Python float ``eps`` as fp64; Triton kernels must cast it to fp32."""
    torch.manual_seed(5)
    fmt = torch.channels_last if in_nhwc else torch.contiguous_format
    x = (torch.randn(1, 16, 32, 32, device=DEV) * 3.0 + 1.5).to(torch.bfloat16).contiguous(memory_format=fmt)
    w = (torch.randn(16, device=DEV) * 0.25).to(torch.bfloat16)

    def fn(inp, weight):
        return ops.instance_norm_prelu(inp, weight, 1e-5, out_channels_last=out_cl)

    torch._dynamo.reset()
    with torch.inference_mode():
        eager = fn(x, w)
        compiled = torch.compile(fn)(x, w)
    assert compiled.shape == eager.shape and compiled.dtype == eager.dtype
    assert torch.equal(compiled, eager)


def test_rgb_mean_instance_norm_torch_compile():
    """Regression: ``_rgb_in_apply_kernel`` also passes Python float ``eps`` to ``tl.math.sqrt_rn``."""
    torch.manual_seed(6)
    img = torch.rand(1, 3, 64, 64, device=DEV, dtype=torch.float32)

    def fn(inp):
        return ops.rgb_mean_instance_norm(inp, 1e-5, dtype=torch.bfloat16, unfold=8)

    torch._dynamo.reset()
    with torch.inference_mode():
        g_eager, u_eager = fn(img)
        g_comp, u_comp = torch.compile(fn)(img)
    assert torch.equal(g_comp, g_eager)
    assert torch.equal(u_comp, u_eager)


def test_disk_upblock_path_torch_compile():
    """Regression: DISK-like ``upsample_concat`` -> ``instance_norm_prelu`` -> 5x5 conv compiles in BF16."""
    torch.manual_seed(7)
    coarse = torch.randn(1, 16, 16, 16, device=DEV, dtype=torch.bfloat16)
    skip = torch.randn(1, 16, 32, 32, device=DEV, dtype=torch.bfloat16)
    prelu_w = (torch.randn(32, device=DEV) * 0.25).to(torch.bfloat16)
    conv = torch.nn.Conv2d(32, 16, 5, padding=2, device=DEV, dtype=torch.bfloat16).eval()

    def block(bot, horiz):
        cat = ops.upsample_concat(horiz, bot, coarse_first=True)
        normed = ops.instance_norm_prelu(cat, prelu_w, 1e-5, out_channels_last=False)
        return conv(normed)

    torch._dynamo.reset()
    with torch.inference_mode():
        eager = block(coarse, skip)
        compiled = torch.compile(block)(coarse, skip)
    torch.testing.assert_close(compiled, eager, atol=2e-2, rtol=2e-2)
