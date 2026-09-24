"""Every optimized impl against its torch reference (and RoMaV2's kernels where available)."""

import pytest
import torch
import torch.nn.functional as F

from flashim import ops
from flashim.quant.formats import mx
from flashim.quant.tensor import QuantizedWeight, quantize_static

pytestmark = pytest.mark.cuda
DEV = "cuda"


def _both(op, impl, *args, **kwargs):
    return op.get(impl)(*args, **kwargs), op.get("reference")(*args, **kwargs)


def _mismatch(a: torch.Tensor, b: torch.Tensor) -> float:
    return (a.view(torch.uint8) != b.view(torch.uint8)).float().mean().item()


@pytest.mark.parametrize("channels", [768, 1024, 1000, 48])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("residual", [False, True])
def test_layernorm_quantize(channels, dtype, residual):
    torch.manual_seed(0)
    x = torch.randn(333, channels, device=DEV, dtype=dtype) * 3
    r = torch.randn_like(x) if residual else None
    w, b = torch.randn(channels, device=DEV, dtype=dtype), torch.randn(channels, device=DEV, dtype=dtype)
    exponent = mx.calibrate_block32_exponent(torch.randn(8, channels) * 2).to(DEV)
    fast, ref = _both(ops.layernorm_quantize, "triton", x, r, w, b, 1e-5, exponent)
    assert fast.values.shape == ref.values.shape and fast.logical_shape == ref.logical_shape
    # FP32 statistics differ only by summation order: allow rare one-code flips.
    assert _mismatch(fast.values, ref.values) < 1e-4
    assert (fast.dequantize() - ref.dequantize()).abs().max() <= ref.dequantize().abs().max() * 0.07


@pytest.mark.parametrize("approximate", ["none", "tanh"])
def test_gelu_quantize(approximate):
    x = torch.randn(257, 3072, device=DEV, dtype=torch.bfloat16) * 4
    exponent = mx.calibrate_block32_exponent(F.gelu(x.float())).to(DEV)
    fast, ref = _both(ops.gelu_quantize, "triton", x, exponent, approximate)
    assert _mismatch(fast.values, ref.values) < 1e-4


@pytest.mark.parametrize("channels", [32, 128, 512])
@pytest.mark.parametrize("hw", [(7, 9), (50, 50)])
def test_depthwise_relu_mxfp8(channels, hw):
    torch.manual_seed(1)
    x = torch.randn(2, channels, *hw, device=DEV).to(memory_format=torch.channels_last)
    w = (torch.randn(25, channels, device=DEV) * 0.3).to(torch.bfloat16)
    b = torch.randn(channels, device=DEV)
    (q1, s1), (q2, s2) = _both(ops.depthwise5x5_relu_mxfp8, "triton", x, w, b)
    assert torch.equal(s1.view(torch.uint8), s2.view(torch.uint8))
    assert _mismatch(q1, q2) < 1e-3
    # All-zero rows (ReLU of negative) must encode the -127 exponent, not NaN/Inf.
    z1, _ = ops.depthwise5x5_relu_mxfp8(x, w, torch.full_like(b, -1e4))
    assert torch.equal(z1.float(), torch.zeros_like(z1.float()))


@pytest.mark.sm120
@pytest.mark.parametrize("rows", [1, 300, 2500])
def test_mxfp8_scaled_mm_matches_emulation(rows):
    w = QuantizedWeight.from_float(torch.randn(768, 1024, device=DEV))
    x = torch.randn(rows, 1024, device=DEV)
    exponent = mx.calibrate_block32_exponent(x)
    q = quantize_static(x, exponent)
    args = (q.matrix(), w.values.t(), mx.static_input_scales(exponent, rows), mx.swizzle_scales(w.scale_exponent))
    fast, ref = _both(ops.mxfp8_scaled_mm, "cublas", *args, None)
    rel = (fast.float() - ref.float()).norm() / ref.float().norm()
    assert rel < 1e-4


def test_fp8_linear_triton_matches_reference():
    w = QuantizedWeight.from_float(torch.randn(96, 160, device=DEV))  # not a multiple of 128
    x = torch.randn(77, 160, device=DEV)
    q = quantize_static(x, mx.calibrate_block32_exponent(x))
    bias = torch.randn(96, device=DEV)
    fast, ref = _both(ops.fp8_linear, "triton", q, w, bias)
    assert ((fast.float() - ref.float()).norm() / ref.float().norm()) < 1e-2


@pytest.mark.parametrize("table_dtype", [torch.bfloat16, torch.float32])
def test_rope_qk(table_dtype):
    q = torch.randn(2, 12, 505, 64, device=DEV, dtype=torch.bfloat16)
    k = torch.randn_like(q)
    sin = torch.randn(500, 64, device=DEV, dtype=table_dtype)
    cos = torch.randn_like(sin)
    with torch.inference_mode():
        (fq, fk), (rq, rk) = _both(ops.rope_qk, "triton", q, k, sin, cos)
    # Prefix tokens untouched; the fused kernel may differ by one BF16 ulp (FMA contraction).
    assert torch.equal(fq[:, :, :5], q[:, :, :5]) and torch.equal(fk[:, :, :5], k[:, :, :5])
    tol = 2 * torch.finfo(torch.bfloat16).eps * rq.float().abs().max()
    assert (fq.float() - rq.float()).abs().max() <= tol
    assert (fk.float() - rk.float()).abs().max() <= tol


@pytest.mark.parametrize("radius", [1, 3])
def test_local_corr(radius):
    f0 = torch.randn(2, 48, 30, 40, device=DEV)
    f1 = torch.randn_like(f0)
    warp = torch.rand(2, 30, 40, 2, device=DEV) * 2.2 - 1.1  # includes out-of-frame samples
    fast, ref = _both(ops.local_corr, "fused_local_corr", f0, f1, warp, radius)
    assert fast.shape == (2, (2 * radius + 1) ** 2, 30, 40)
    assert torch.allclose(fast, ref, atol=1e-4, rtol=1e-4)


@pytest.mark.parametrize("channels,hw", [(32, (37, 53)), (32, (256, 256)), (64, (40, 48))])
def test_depthwise_relu_pointwise_bf16(channels, hw):
    torch.manual_seed(2)
    x = torch.randn(2, channels, *hw, device=DEV).to(memory_format=torch.channels_last)
    dw = (torch.randn(25, channels, device=DEV) * 0.2).to(torch.bfloat16)
    dwb = torch.randn(channels, device=DEV).to(torch.bfloat16)
    pw = (torch.randn(channels, channels, device=DEV) * 0.2).to(torch.bfloat16).contiguous()
    pwb = torch.randn(channels, device=DEV).to(torch.bfloat16)
    fast, ref = _both(ops.depthwise5x5_relu_pointwise, "triton", x, dw, dwb, pw, pwb)
    assert fast.shape == ref.shape and fast.is_contiguous(memory_format=torch.channels_last)
    # Only FP32 reduction order differs: at most one BF16 ulp.
    assert ((fast.float() - ref.float()).abs() <= ref.float().abs() * 2**-7 + 1e-6).all()
