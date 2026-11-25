"""Correctness tests for ``ops.conv3x3_small_cin_relu`` (BF16 / INT8 / MXFP8) and ``ops.int8_conv2d_maxpool_only``."""

from __future__ import annotations

import pytest
import torch

from flashim import ops
from flashim.platforms import Feature, platform_for
from flashim.quant.formats import mx
from flashim.quant.methods.conv2d import StaticInt8Conv2d

pytestmark = pytest.mark.cuda
DEV = "cuda"

_needs_e4m3 = pytest.mark.skipif(
    not torch.cuda.is_available() or not platform_for(DEV).supports(Feature.FP8_E4M3), reason="needs E4M3 (sm_89+)"
)


@pytest.mark.parametrize("cin,cout", [(1, 64), (3, 16), (3, 64)])
@pytest.mark.parametrize("hw", [(37, 45), (224, 300), (896, 1200)])
def test_conv3x3_small_cin_relu_bf16(cin, cout, hw):
    torch.manual_seed(0)
    x = torch.randn(1, cin, *hw, device=DEV, dtype=torch.bfloat16)
    w = (torch.randn(cout, cin, 3, 3, device=DEV) * 0.2).to(torch.bfloat16)
    b = (torch.randn(cout, device=DEV) * 0.1).to(torch.bfloat16)
    assert ops.conv3x3_small_cin_relu.explain(x, w, b)["triton"] == "selected"
    fast = ops.conv3x3_small_cin_relu.get("triton")(x, w, b)
    ref = ops.conv3x3_small_cin_relu.get("reference")(x, w, b)
    assert fast.shape == ref.shape and fast.dtype == torch.bfloat16
    assert fast.is_contiguous(memory_format=torch.channels_last)
    # FP32 summation order differs by at most 1 BF16 ulp on a tiny fraction of outputs.
    assert ((fast.float() - ref.float()).abs() <= ref.float().abs() * 2**-6 + 1e-3).all()


def test_conv3x3_cin1_exact_480x640_bf16():
    """Cin=1 9-tap row-major FP32 FMA matches cuDNN SIMT bit-for-bit (or <1e-5 midpoint ties)."""
    torch.manual_seed(0)
    x = torch.rand(1, 1, 480, 640, device=DEV, dtype=torch.bfloat16)
    w = (torch.randn(64, 1, 3, 3, device=DEV) * 0.2).to(torch.bfloat16)
    b = (torch.randn(64, device=DEV) * 0.1).to(torch.bfloat16)
    fast = ops.conv3x3_small_cin_relu.get("triton")(x, w, b)
    ref = ops.conv3x3_small_cin_relu.get("reference")(x, w, b)
    mismatch = (fast != ref).float().mean().item()
    assert mismatch < 1e-5, f"Cin=1 mismatch rate {mismatch:.2e} too high"


@pytest.mark.parametrize("cin,cout", [(1, 64), (3, 64)])
@pytest.mark.parametrize("hw", [(37, 45), (224, 300)])
def test_conv3x3_small_cin_relu_int8(cin, cout, hw):
    torch.manual_seed(1)
    x = torch.randn(1, cin, *hw, device=DEV, dtype=torch.bfloat16)
    w = (torch.randn(cout, cin, 3, 3, device=DEV) * 0.2).to(torch.bfloat16)
    b = (torch.randn(cout, device=DEV) * 0.1).to(torch.bfloat16)
    scale = torch.tensor(0.02, device=DEV)
    fast = ops.conv3x3_small_cin_relu.get("triton")(x, w, b, out_scale=scale)
    ref = ops.conv3x3_small_cin_relu.get("reference")(x, w, b, out_scale=scale)
    assert fast.values.shape == ref.values.shape and fast.scale is scale
    assert (fast.values.int() - ref.values.int()).abs().max() <= 1
    assert (fast.values != ref.values).float().mean().item() < 1e-3


@_needs_e4m3
@pytest.mark.parametrize("cin,cout", [(1, 64), (3, 64)])
@pytest.mark.parametrize("hw", [(37, 45), (224, 300)])
def test_conv3x3_small_cin_relu_mxfp8(cin, cout, hw):
    torch.manual_seed(2)
    x = torch.randn(1, cin, *hw, device=DEV, dtype=torch.bfloat16)
    w = (torch.randn(cout, cin, 3, 3, device=DEV) * 0.2).to(torch.bfloat16)
    b = (torch.randn(cout, device=DEV) * 0.1).to(torch.bfloat16)
    exp = mx.calibrate_block32_exponent(torch.randn(8, cout) * 2).to(DEV)
    fast = ops.conv3x3_small_cin_relu.get("triton")(x, w, b, out_exponent=exp)
    ref = ops.conv3x3_small_cin_relu.get("reference")(x, w, b, out_exponent=exp)
    assert fast.values.shape == ref.values.shape
    mismatch = (fast.values.view(torch.uint8) != ref.values.view(torch.uint8)).float().mean().item()
    assert mismatch < 1e-3


@pytest.mark.parametrize("cin,cout,hw", [(64, 64, (64, 80)), (64, 128, (224, 300)), (128, 128, (112, 150))])
def test_int8_conv2d_maxpool_only_bit_identical(cin, cout, hw):
    torch.manual_seed(3)
    conv = torch.nn.Conv2d(cin, cout, 3, padding=1, device=DEV)
    in_scale = torch.tensor(0.03, device=DEV)
    out_scale = torch.tensor(0.05, device=DEV)
    qconv = StaticInt8Conv2d.from_conv(conv, in_scale)
    x = torch.randint(-127, 127, (1, *hw, cin), device=DEV, dtype=torch.int8)
    args = (x, qconv.weight_int8, qconv.input_scale, qconv.weight_scale, qconv.bias, *qconv._geometry_args())
    assert ops.int8_conv2d_maxpool_only.explain(*args, relu=True, out_scale=out_scale)["triton"] == "selected"
    fast = ops.int8_conv2d_maxpool_only.get("triton")(*args, relu=True, out_scale=out_scale)
    ref = ops.int8_conv2d_maxpool_only.get("reference")(*args, relu=True, out_scale=out_scale)
    assert torch.equal(fast, ref)
