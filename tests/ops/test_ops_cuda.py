"""Every optimized impl against its torch reference (and RoMaV2's kernels where available)."""

import pytest
import torch
import torch.nn.functional as F

from flashim import ops
from flashim.platforms import Feature, platform_for
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


_needs_tilelang = pytest.mark.skipif(
    not torch.cuda.is_available() or not platform_for(DEV).supports(Feature.TILELANG, Feature.TMA), reason="needs TileLang + TMA (sm_90+)"
)


@_needs_tilelang
@pytest.mark.parametrize(
    "batch,channels,hw", [(1, 64, (7, 17)), (2, 128, (37, 45)), (3, 512, (50, 300)), (8, 512, (256, 256))]
)
def test_depthwise_relu_mxfp8_tilelang_matches_triton(batch, channels, hw):
    """Bit-identical to the Triton kernel (same FP32 order and scale math), and stable
    across repeated launches (guards the shared-memory stage reuse race)."""
    torch.manual_seed(2)
    x = torch.randn(batch, channels, *hw, device=DEV).to(torch.bfloat16).to(memory_format=torch.channels_last)
    w = (torch.randn(25, channels, device=DEV) * 0.3).to(torch.bfloat16)
    b = torch.randn(channels, device=DEV)
    q_ref, s_ref = ops.depthwise5x5_relu_mxfp8.get("triton")(x, w, b)
    fast = ops.depthwise5x5_relu_mxfp8.get("tilelang")
    for _ in range(20):
        q, s = fast(x, w, b)
        assert torch.equal(q.view(torch.uint8), q_ref.view(torch.uint8))
        assert torch.equal(s.view(torch.uint8), s_ref.view(torch.uint8))
    z, _ = fast(x, w, torch.full_like(b, -1e4))
    assert torch.equal(z.float(), torch.zeros_like(z.float()))


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


def test_mxfp8_linear_triton_matches_reference():
    w = QuantizedWeight.from_float(torch.randn(96, 160, device=DEV))  # not a multiple of 128
    x = torch.randn(77, 160, device=DEV)
    q = quantize_static(x, mx.calibrate_block32_exponent(x))
    bias = torch.randn(96, device=DEV)
    ops.reset_call_counts()
    fast, ref = _both(ops.mxfp8_linear, "triton", q, bias, w)
    # Dataclass arguments must resolve to their CUDA device (a pinned impl used to fall back silently).
    assert ops.call_counts()["mxfp8_linear"] == {"triton": 1, "reference": 1}
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


@pytest.mark.parametrize("proj,emb,radius,hw", [(192, 79, 3, (37, 45)), (48, 23, 1, (64, 80)), (12, 8, -1, (50, 70))])
def test_romav2_refiner_input(proj, emb, radius, hw):
    torch.manual_seed(3)
    f_a = torch.randn(2, *hw, proj, device=DEV)
    f_b = torch.randn(2, *hw, proj, device=DEV)
    warp = (torch.rand(2, *hw, 2, device=DEV) * 2.4 - 1.2).contiguous()  # includes out-of-frame
    weight, bias = torch.randn(emb, 2, 1, 1, device=DEV), torch.randn(emb, device=DEV)
    scale = torch.tensor([1.5625, 2.0], device=DEV)
    fast, ref = _both(ops.romav2_refiner_input, "triton", f_a, f_b, warp, weight, bias, scale, radius)
    assert fast.shape == ref.shape and fast.dtype == torch.bfloat16
    # f_a and f_ba are exact; emb/corr differ by FP32 order (and cuDNN TF32 in the reference): <= 1 BF16 ulp
    assert torch.equal(fast[:, : 2 * proj], ref[:, : 2 * proj])
    tail_fast, tail_ref = fast[:, 2 * proj :].float(), ref[:, 2 * proj :].float()
    assert ((tail_fast - tail_ref).abs() <= tail_ref.abs().amax() * 2**-7).all()


@pytest.mark.parametrize("channels,hw,prev_c", [(512, (40, 50), 4), (128, (64, 64), 4), (32, (70, 90), 1)])
def test_romav2_refiner_heads(channels, hw, prev_c):
    torch.manual_seed(4)
    z = (torch.randn(2, channels, *hw, device=DEV) * 0.5).to(torch.bfloat16).contiguous(memory_format=torch.channels_last)
    weight, bias = torch.randn(6, channels, device=DEV) * 0.05, torch.randn(6, device=DEV)
    prev_warp = torch.rand(2, *hw, 2, device=DEV) * 2 - 1
    prev_conf = torch.randn(2, *hw, prev_c, device=DEV)
    with torch.backends.cudnn.flags(allow_tf32=False):  # compare against a true-FP32 reference
        (w1, c1), (w0, c0) = _both(ops.romav2_refiner_heads, "triton", z, weight, bias, prev_warp, prev_conf, 4.0)
    assert w1.shape == w0.shape == (2, *hw, 2) and c1.shape == c0.shape == (2, *hw, 4)
    assert torch.allclose(w1, w0, atol=1e-6, rtol=1e-5)
    assert torch.allclose(c1, c0, atol=1e-4, rtol=1e-4)


@pytest.mark.parametrize("batch", [1, 2])
@pytest.mark.parametrize("proj,emb,radius,hw", [(192, 79, 3, (37, 45)), (12, 8, -1, (50, 70))])
def test_romav2_refiner_input_stacked_directions(batch, proj, emb, radius, hw):
    """pair=B: a stack [A-side; B-side] matches image i with image (i + B) % 2B, both directions at once."""
    torch.manual_seed(5)
    stack = torch.randn(2 * batch, *hw, proj, device=DEV)
    warp = (torch.rand(2 * batch, *hw, 2, device=DEV) * 2.4 - 1.2).contiguous()
    weight, bias = torch.randn(emb, 2, 1, 1, device=DEV), torch.randn(emb, device=DEV)
    scale = torch.tensor([1.5625, 2.0], device=DEV)
    fast = ops.romav2_refiner_input.get("triton")(stack, stack, warp, weight, bias, scale, radius, batch)
    ref = ops.romav2_refiner_input.get("reference")(stack, stack, warp, weight, bias, scale, radius, batch)
    # the stacked call must equal two separate one-direction calls
    a, b = stack[:batch], stack[batch:]
    ab = ops.romav2_refiner_input.get("triton")(a, b, warp[:batch], weight, bias, scale, radius)
    ba = ops.romav2_refiner_input.get("triton")(b, a, warp[batch:], weight, bias, scale, radius)
    assert torch.equal(fast[:batch], ab) and torch.equal(fast[batch:], ba)
    assert torch.equal(fast[:, : 2 * proj], ref[:, : 2 * proj])
    tail_fast, tail_ref = fast[:, 2 * proj :].float(), ref[:, 2 * proj :].float()
    assert ((tail_fast - tail_ref).abs() <= tail_ref.abs().amax() * 2**-7).all()


@pytest.mark.parametrize("k,p,rows", [(64, 12, 5000), (32, 8, 777), (128, 16, 4096)])
def test_linear_fp32_small_n(k, p, rows):
    x = torch.randn(2, rows // 2 + 1, k, device=DEV).to(torch.bfloat16)
    weight, bias = torch.randn(p, k, device=DEV) * 0.1, torch.randn(p, device=DEV)
    with torch.backends.cudnn.flags(allow_tf32=False):
        fast, ref = _both(ops.linear_fp32, "triton", x, weight, bias)
    assert fast.shape == ref.shape == (2, rows // 2 + 1, p) and fast.dtype == torch.float32
    assert torch.allclose(fast, ref, atol=1e-5, rtol=1e-5)


@pytest.mark.sm120
@pytest.mark.parametrize("batch,hw", [(1, (37, 45)), (2, (64, 64)), (2, (100, 33))])
def test_depthwise5x5_relu_pointwise_mxfp8_fused(batch, hw):
    """The fused kernel equals depthwise5x5_relu_mxfp8 -> mxfp8_scaled_mm (up to BF16 rounding of a
    different FP32 summation order)."""
    torch.manual_seed(6)
    channels = 128
    x = torch.randn(batch, channels, *hw, device=DEV).to(torch.bfloat16).to(memory_format=torch.channels_last)
    dw = (torch.randn(25, channels, device=DEV) * 0.3).to(torch.bfloat16)
    dwb = torch.randn(channels, device=DEV).to(torch.bfloat16)
    weight, exponent = mx.quantize_dynamic_2d(torch.randn(channels, channels, device=DEV) * 0.1)
    args = (
        x, dw, dwb, weight, mx.swizzle_scales(exponent), (exponent.to(torch.int16) + 127).to(torch.uint8),
        torch.randn(channels, device=DEV).to(torch.bfloat16),
    )
    fast, ref = _both(ops.depthwise5x5_relu_pointwise_mxfp8, "triton_fused", *args)
    assert fast.shape == ref.shape and fast.dtype == torch.bfloat16
    assert ((fast.float() - ref.float()).abs() <= ref.float().abs() * 2**-7 + 1e-6).all()


@pytest.mark.parametrize("n", [1, 777, 20000])
def test_kde_density(n):
    torch.manual_seed(7)
    centers = torch.rand(20, 4, device=DEV) * 2 - 1
    x = (centers[torch.randint(0, 20, (n,), device=DEV)] + torch.randn(n, 4, device=DEV) * 0.08).clamp(-1, 1)
    fast = ops.kde_density.get("triton")(x)
    exact = torch.exp(-torch.cdist(x.double(), x.double()) ** 2 / 0.02).sum(-1)
    assert fast.dtype == torch.float16 and fast.shape == (n,)
    # FP32 accumulation: only the FP16 output rounding remains (the reference FP16 matrix is ~1e-1 off)
    assert ((fast.double() - exact).abs() / exact).max() < 1e-3


@pytest.mark.parametrize("rows,k,n", [(3000, 256, 192), (5000, 128, 48), (77, 64, 40)])
def test_linear_fp32_split_close_to_fp32(rows, k, n):
    torch.manual_seed(0)
    x = torch.randn(rows, k, device=DEV).bfloat16()
    w, b = torch.randn(n, k, device=DEV) * 0.1, torch.randn(n, device=DEV)
    got = ops.linear_fp32.get("triton_split")(x, w, b)
    want = ops.linear_fp32.get("reference")(x, w, b)
    assert got.dtype == torch.float32 and ((got - want).norm() / want.norm()) < 1e-5


@pytest.mark.parametrize("impl,tol", [("sage", 0.02), ("sage-fp8", 0.06)])
def test_sdpa_sage_close_to_sdpa(impl, tol):
    pytest.importorskip("sageattention")
    torch.manual_seed(0)
    qkv = torch.randn(2, 700, 3, 12, 64, device=DEV, dtype=torch.bfloat16)
    q, k, v = qkv.permute(2, 0, 3, 1, 4)  # strided views, as the transformer block passes them
    want = ops.sdpa.get("reference")(q, k, v).float()
    got = ops.sdpa.get(impl)(q, k, v).float()
    assert got.shape == want.shape and ((got - want).norm() / want.norm()) < tol


def _deformable_inputs(batch=2, heads=8, dim=32, levels=((24, 32), (12, 16), (6, 8), (3, 4)), points=4, dtype=torch.bfloat16):
    torch.manual_seed(0)
    tokens = sum(h * w for h, w in levels)
    value = torch.randn(batch, tokens, heads, dim, device=DEV).to(dtype)
    width = heads * len(levels) * points
    proj = (torch.randn(batch, tokens, width * 3, device=DEV) * 3).to(dtype)  # fused [offsets | logits] projection
    offsets = proj[..., : width * 2].view(batch, tokens, heads, len(levels), points, 2)
    logits = proj[..., width * 2 :].view(batch, tokens, heads, len(levels), points)
    reference = torch.rand(batch, tokens, 2, device=DEV) * 1.2 - 0.1  # some samples fall outside the map
    return value, offsets, logits, reference, levels


def _detr_core(value, shapes, locations, weights):
    """Deformable DETR's ``ms_deform_attn_core_pytorch`` (the RDD reference path), for cross-checking."""
    n, _, m, d = value.shape
    _, lq, _, levels, points, _ = locations.shape
    value_list = value.split([h * w for h, w in shapes], dim=1)
    grids = 2 * locations - 1
    sampled = []
    for level, (h, w) in enumerate(shapes):
        v = value_list[level].flatten(2).transpose(1, 2).reshape(n * m, d, h, w)
        g = grids[:, :, :, level].transpose(1, 2).flatten(0, 1)
        sampled.append(F.grid_sample(v, g, mode="bilinear", padding_mode="zeros", align_corners=False))
    weights = weights.transpose(1, 2).reshape(n * m, 1, lq, levels * points)
    out = (torch.stack(sampled, dim=-2).flatten(-2) * weights).sum(-1).view(n, m * d, lq)
    return out.transpose(1, 2)


def test_deformable_attention_reference_matches_detr_core():
    value, offsets, logits, reference, shapes = _deformable_inputs(dtype=torch.float32)
    weights = torch.softmax(logits.flatten(-2), -1).view_as(logits)
    normalizer = torch.tensor([[w, h] for h, w in shapes], device=DEV, dtype=torch.float32)
    locations = reference[:, :, None, None, None, :] + offsets / normalizer[None, None, None, :, None, :]
    expected = _detr_core(value, shapes, locations, weights)
    out = ops.deformable_attention.get("reference")(value, offsets, logits, reference, shapes, softmax=True)
    torch.testing.assert_close(out, expected, atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize("softmax", [False, True])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_deformable_attention_triton(softmax, dtype):
    value, offsets, logits, reference, shapes = _deformable_inputs(dtype=dtype)
    weights = logits if softmax else torch.softmax(logits.float().flatten(-2), -1).view_as(logits)
    ops.reset_call_counts()
    fast, ref = _both(ops.deformable_attention, "triton", value, offsets, weights, reference, shapes, softmax)
    assert ops.call_counts()["deformable_attention"] == {"triton": 1, "reference": 1}
    tol = 1e-2 if dtype == torch.bfloat16 else 1e-5  # output rounding only (FP32 accumulation in both)
    torch.testing.assert_close(fast.float(), ref.float(), atol=tol, rtol=tol)


def test_deformable_attention_compiles_without_fallback():
    value, offsets, logits, reference, shapes = _deformable_inputs()
    compiled = torch.compile(lambda *a: ops.deformable_attention(*a, shapes, softmax=True), fullgraph=True)
    with torch.inference_mode():
        out = compiled(value, offsets, logits, reference)
        ref = ops.deformable_attention.get("reference")(value, offsets, logits, reference, shapes, softmax=True)
    torch.testing.assert_close(out.float(), ref.float(), atol=1e-2, rtol=1e-2)


def test_quantize_mxfp8():
    x = torch.randn(300, 96, device=DEV, dtype=torch.bfloat16) * 5
    exponent = mx.calibrate_block32_exponent(x.float())
    fast, ref = _both(ops.quantize_mxfp8, "triton", x, exponent)
    assert torch.equal(fast.values.view(torch.uint8), ref.values.view(torch.uint8))


def test_layernorm_quantize_input_bias_and_dense_output():
    torch.manual_seed(0)
    x = torch.randn(333, 256, device=DEV, dtype=torch.bfloat16) * 3
    r, bias = torch.randn_like(x), torch.randn(256, device=DEV, dtype=torch.bfloat16)
    w, b = torch.randn(256, device=DEV), torch.randn(256, device=DEV)
    exponent = mx.calibrate_block32_exponent(torch.randn(8, 256) * 2).to(DEV)
    (fast, fast_dense), (ref, ref_dense) = _both(
        ops.layernorm_quantize, "triton", x, r, w, b, 1e-5, exponent, input_bias=bias, return_dense=True
    )
    assert _mismatch(fast.values, ref.values) < 1e-4
    torch.testing.assert_close(fast_dense, ref_dense, atol=0.02, rtol=0.01)


def test_gelu_quantize_input_bias():
    x = torch.randn(257, 1024, device=DEV, dtype=torch.bfloat16) * 4
    bias = torch.randn(1024, device=DEV, dtype=torch.bfloat16)
    exponent = mx.calibrate_block32_exponent(F.gelu(x.float() + bias.float())).to(DEV)
    fast, ref = _both(ops.gelu_quantize, "triton", x, exponent, "none", input_bias=bias)
    assert _mismatch(fast.values, ref.values) < 1e-4


@pytest.mark.sm120
@pytest.mark.parametrize("rows", [1, 200, 4099])
def test_mxfp8_gelu_mlp_matches_two_linears(rows):
    torch.manual_seed(0)
    w1 = QuantizedWeight.from_float(torch.randn(1024, 256, device=DEV) * 0.05)
    w2 = QuantizedWeight.from_float(torch.randn(256, 1024, device=DEV) * 0.05)
    b1 = torch.randn(1024, device=DEV) * 0.1
    x = torch.randn(rows, 256, device=DEV, dtype=torch.bfloat16)
    q = quantize_static(x, mx.calibrate_block32_exponent(x.float()))
    hidden_exponent = mx.calibrate_block32_exponent(F.gelu(torch.randn(64, 1024) * 2)).to(DEV)
    fast, ref = _both(ops.mxfp8_gelu_mlp, "triton", q, w1, b1, hidden_exponent, w2)
    # Same rounding points; FP32 accumulation order differs only inside each block-scaled MMA.
    assert ((fast.float() - ref.float()).norm() / ref.float().norm()) < 1e-3
