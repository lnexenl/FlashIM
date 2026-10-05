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


@pytest.mark.parametrize("n,k", [(64, 3200), (196, 2016), (196, 2304), (128, 2016)])  # real DISK / LoFTR shapes
@pytest.mark.parametrize("bias", [False, True])
def test_mxfp8_linear_tilelang_matches_reference(n, k, bias):
    """The batched-chunk K-loop kernel (``backends/tilelang/_gemm_kernels.py``) against the dequantize-then-matmul
    reference, at the large-K shapes it targets (cuBLAS rejects them; below K=2000 this backend declines and
    Triton handles them, see ``backends/tilelang/gemm.py:MIN_K``)."""
    w = QuantizedWeight.from_float(torch.randn(n, k, device=DEV) * 0.3)
    x = torch.randn(4000, k, device=DEV) * 0.3
    q = quantize_static(x, mx.calibrate_block32_exponent(x))
    # bias is padded to the physical (32-padded) out_features before it reaches the op, same as StaticMXFP8Linear
    # pads it once at construction (__init__); an unpadded bias is a caller bug, not something the op handles.
    b = F.pad(torch.randn(n, device=DEV), (0, w.out_features - n)) if bias else None
    ops.reset_call_counts()
    fast, ref = _both(ops.mxfp8_linear, "tilelang", q, b, w)
    assert ops.call_counts()["mxfp8_linear"] == {"tilelang": 1, "reference": 1}
    assert ((fast.float() - ref.float()).norm() / ref.float().norm()) < 1e-2


def test_mxfp8_linear_dispatch_prefers_tilelang_above_k_threshold():
    """Large-K, cuBLAS-ineligible shapes reach the new kernel through normal dispatch; smaller ones stay on Triton."""
    big = QuantizedWeight.from_float(torch.randn(64, 3200, device=DEV))
    small = QuantizedWeight.from_float(torch.randn(64, 576, device=DEV))
    x_big, x_small = torch.randn(500, 3200, device=DEV), torch.randn(500, 576, device=DEV)
    q_big = quantize_static(x_big, mx.calibrate_block32_exponent(x_big))
    q_small = quantize_static(x_small, mx.calibrate_block32_exponent(x_small))
    ops.reset_call_counts()
    ops.mxfp8_linear(q_big, None, big)
    ops.mxfp8_linear(q_small, None, small)
    assert ops.call_counts()["mxfp8_linear"] == {"tilelang": 1, "triton": 1}


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


@pytest.mark.parametrize(
    "channels,hw", [(16, (33, 70)), (32, (37, 53)), (32, (256, 256)), (64, (40, 48)), (64, (130, 97)), (128, (45, 67))]
)
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


@pytest.mark.parametrize("channels", [16, 48, 96, 257])
@pytest.mark.parametrize("fmt", ["int8", "mxfp8"])
def test_quantize_channels_first_view(channels, fmt):
    """The channels-last view of NCHW memory (quantized convolutions' input) is read in place: bytes equal to
    quantizing the packed copy, and to the reference."""
    torch.manual_seed(0)
    x = torch.randn(2, channels, 17, 19, device=DEV, dtype=torch.bfloat16) * 5
    view = x.permute(0, 2, 3, 1)
    assert not view.is_contiguous()
    if fmt == "int8":
        op, scale = ops.quantize_int8, torch.tensor(0.04, device=DEV)
    else:
        op, scale = ops.quantize_mxfp8, mx.calibrate_block32_exponent(view.reshape(-1, channels).float())
    fast, ref = _both(op, "triton", view, scale)
    packed = op.get("triton")(view.contiguous(), scale)
    assert fast.values.is_contiguous() and fast.values.shape == packed.values.shape == ref.values.shape
    assert torch.equal(fast.values.view(torch.uint8), packed.values.view(torch.uint8))
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


@pytest.mark.parametrize(
    "cin,cout,kh,kw,sh,sw,ph,pw,dh,dw",
    [
        (64, 64, 3, 3, 1, 1, 1, 1, 1, 1),
        (48, 40, 3, 3, 1, 1, 1, 1, 1, 1),  # Cin not a multiple of 32
        (16, 24, 3, 3, 1, 1, 1, 1, 1, 1),  # Cin < 32 (tl.dot's K minimum)
        (64, 64, 1, 1, 1, 1, 0, 0, 1, 1),  # 1x1
        (32, 32, 3, 3, 2, 2, 1, 1, 1, 1),  # stride 2
        (32, 32, 3, 3, 1, 1, 2, 2, 2, 2),  # dilation 2
        (8, 16, 7, 7, 2, 2, 3, 3, 1, 1),  # 7x7 stride 2 (loftr-like stem), Cin < 32
        (196, 128, 3, 3, 1, 1, 1, 1, 1, 1),  # Cin > 128, not a multiple of the 64/128 chunk (LoFTR)
        (512, 64, 3, 3, 1, 1, 1, 1, 1, 1),  # four 128-channel chunks per tap (VGG conv4)
        (96, 64, 5, 5, 1, 1, 2, 2, 1, 1),  # 5x5, Cin a multiple of neither 64 nor 128 (DISK)
    ],
)
@pytest.mark.parametrize("has_bias", [True, False])
def test_int8_conv2d(cin, cout, kh, kw, sh, sw, ph, pw, dh, dw, has_bias):
    torch.manual_seed(0)
    x = torch.randint(-127, 127, (2, 17, 19, cin), dtype=torch.int8, device=DEV)
    weight = torch.randint(-127, 127, (cout, kh * kw * cin), dtype=torch.int8, device=DEV)
    a_scale = torch.tensor(0.017, device=DEV)
    w_scale = torch.rand(cout, device=DEV) * 0.01 + 0.001
    bias = torch.randn(cout, device=DEV, dtype=torch.bfloat16) if has_bias else None
    fast, ref = _both(
        ops.int8_conv2d, "triton", x, weight, a_scale, w_scale, bias, kh, kw, sh, sw, ph, pw, dh, dw
    )
    assert fast.shape == ref.shape
    assert torch.equal(fast, ref)  # exact INT32 accumulation and a non-contracted epilogue, whatever the config
    # ReLU epilogue: the same bits as a separate ReLU on the plain result (it commutes with the BF16 rounding)
    fused = ops.int8_conv2d.get("triton")(x, weight, a_scale, w_scale, bias, kh, kw, sh, sw, ph, pw, dh, dw, relu=True)
    assert torch.equal(fused, F.relu(fast))


@pytest.mark.sm120
@pytest.mark.parametrize(
    "cin,cout,kh,kw,sh,sw,ph,pw,dh,dw",
    [
        (64, 64, 3, 3, 1, 1, 1, 1, 1, 1),
        (224, 128, 3, 3, 1, 1, 1, 1, 1, 1),  # LoFTR's 196 channels padded to 224: 32- and 64-channel chunks only
        (512, 64, 3, 3, 1, 1, 1, 1, 1, 1),  # 128-channel chunks (VGG conv4)
        (96, 64, 5, 5, 1, 1, 2, 2, 1, 1),  # 5x5 (DISK)
        (64, 196, 3, 3, 2, 2, 1, 1, 1, 1),  # stride 2, Cout not a multiple of the tile
        (32, 32, 3, 3, 1, 1, 2, 2, 2, 2),  # dilation 2
        (128, 128, 1, 1, 1, 1, 0, 0, 1, 1),  # 1x1
    ],
)
@pytest.mark.parametrize("has_bias", [True, False])
def test_mxfp8_conv2d_scaled(cin, cout, kh, kw, sh, sw, ph, pw, dh, dw, has_bias):
    """Block-scaled MMA implicit GEMM vs the reference (im2col + MXFP8 linear): same E4M3 operands and scales, FP32
    accumulation in a different order."""
    torch.manual_seed(0)
    x = (torch.randn(2, 17, 19, cin, device=DEV) * 3).to(torch.float8_e4m3fn)
    channel_exponent = torch.randint(-6, 3, (cin // 32,), dtype=torch.int8, device=DEV)
    weight = (torch.randn(cout, kh * kw * cin, device=DEV) * 2).to(torch.float8_e4m3fn)
    weight_exponent = torch.randint(-9, -3, (cout, kh * kw * cin // 32), dtype=torch.int8, device=DEV)
    bias = torch.randn(cout, device=DEV, dtype=torch.bfloat16) if has_bias else None
    args = (x, channel_exponent, weight, weight_exponent, bias, kh, kw, sh, sw, ph, pw, dh, dw)
    fast, ref = _both(ops.mxfp8_conv2d, "triton_scaled", *args)
    assert fast.shape == ref.shape
    assert ((fast.float() - ref.float()).norm() / ref.float().norm()) < 1e-3
    fused = ops.mxfp8_conv2d.get("triton_scaled")(*args, relu=True)
    assert torch.equal(fused, F.relu(fast))


@pytest.mark.parametrize("impl", ["triton", "cublas"])
def test_int8_linear_relu(impl):
    torch.manual_seed(0)
    a = torch.randint(-127, 127, (300, 256), dtype=torch.int8, device=DEV)
    weight = torch.randint(-127, 127, (96, 256), dtype=torch.int8, device=DEV)
    a_scale, w_scale = torch.tensor(0.02, device=DEV), torch.rand(96, device=DEV) * 0.01
    bias = torch.randn(96, device=DEV, dtype=torch.bfloat16)
    plain = ops.int8_linear.get(impl)(a, weight, a_scale, w_scale, bias)
    fused, ref = _both(ops.int8_linear, impl, a, weight, a_scale, w_scale, bias, relu=True)
    assert torch.equal(fused, F.relu(plain))
    torch.testing.assert_close(fused, ref)


@pytest.mark.parametrize("fmt", ["int8", "mxfp8"])
def test_layernorm_quantize_return_sum(fmt):
    """``return_sum``: the residual stream ``value + residual`` with exactly a BF16 add's rounding."""
    torch.manual_seed(0)
    x = torch.randn(333, 1024, device=DEV, dtype=torch.bfloat16) * 3
    r = torch.randn_like(x) * 10
    w, b = torch.randn(1024, device=DEV, dtype=torch.bfloat16), torch.randn(1024, device=DEV, dtype=torch.bfloat16)
    if fmt == "int8":
        op, scale = ops.layernorm_quantize_int8, torch.tensor(0.05, device=DEV)
    else:
        op, scale = ops.layernorm_quantize, mx.calibrate_block32_exponent(torch.randn(8, 1024) * 2).to(DEV)
    (fast, fast_sum), (ref, ref_sum) = _both(op, "triton", x, r, w, b, 1e-6, scale, return_sum=True)
    plain = op.get("triton")(x, r, w, b, 1e-6, scale)
    assert torch.equal(fast_sum, x + r) and torch.equal(ref_sum, x + r)
    assert torch.equal(fast.values.view(torch.uint8), plain.values.view(torch.uint8))
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


@pytest.mark.parametrize("k,dtype", [(5, torch.bfloat16), (3, torch.float32), (1, torch.bfloat16)])
def test_conv2d_at_points_matches_dense(k, dtype):
    """The convolution read at given pixels (borders included) equals the dense FP32 convolution indexed there."""
    torch.manual_seed(0)
    x = torch.randn(1, 24, 37, 45, device=DEV).to(dtype)
    weight, bias = torch.randn(16, 24, k, k, device=DEV), torch.randn(16, device=DEV)
    points = torch.cat([torch.randint(0, 45, (300, 1)), torch.randint(0, 37, (300, 1))], 1).to(DEV)
    points[:4] = torch.tensor([[0, 0], [44, 36], [0, 36], [44, 0]], device=DEV)  # corners: padding taps
    torch.backends.cudnn.allow_tf32 = False
    try:
        dense = F.conv2d(x.float(), weight, bias, padding=k // 2)[0][:, points[:, 1], points[:, 0]].t()
    finally:
        torch.backends.cudnn.allow_tf32 = True
    out = ops.conv2d_at_points(x, weight, bias, points)
    torch.testing.assert_close(out, dense, atol=1e-4, rtol=1e-4)


@pytest.mark.parametrize("k,hw,dtype", [(5, (37, 45), torch.bfloat16), (3, (130, 70), torch.float32), (1, (64, 90), torch.bfloat16)])
def test_conv2d_single_output(k, hw, dtype):
    torch.manual_seed(0)
    x = torch.randn(2, 12, *hw, device=DEV).to(dtype)
    weight, bias = torch.randn(1, 12, k, k, device=DEV) * 0.1, torch.randn(1, device=DEV)
    exact = F.conv2d(x.double(), weight.double(), bias.double(), padding=k // 2)
    for view in (x, x.contiguous(memory_format=torch.channels_last)):
        out = ops.conv2d_single_output.get("triton")(view, weight, bias)
        assert out.dtype == torch.float32 and out.shape == exact.shape
        assert ((out.double() - exact).abs().max() / exact.abs().max()) < 1e-5  # full FP32, not TF32


@pytest.mark.parametrize("fmt", ["int8", "mxfp8"])
@pytest.mark.parametrize("layout", ["nhwc", "nchw"])
def test_conv2d_residual_relu_epilogue(fmt, layout):
    """``relu(conv + residual)`` with the residual in the epilogue: one BF16 rounding instead of two, so within one
    BF16 ulp of the reference (which rounds the convolution first)."""
    torch.manual_seed(0)
    cin, cout = 64, 96
    residual = torch.randn(2, cout, 17, 19, device=DEV, dtype=torch.bfloat16)
    residual = residual.contiguous(memory_format=torch.channels_last) if layout == "nhwc" else residual
    if fmt == "int8":
        x = torch.randint(-127, 127, (2, 17, 19, cin), dtype=torch.int8, device=DEV)
        weight = torch.randint(-127, 127, (cout, 9 * cin), dtype=torch.int8, device=DEV)
        args = (x, weight, torch.tensor(0.002, device=DEV), torch.rand(cout, device=DEV) * 0.001,
                torch.randn(cout, device=DEV, dtype=torch.bfloat16), 3, 3, 1, 1, 1, 1, 1, 1)
        op, impl = ops.int8_conv2d, "triton"
    else:
        if not platform_for(DEV).supports(Feature.MXFP8_SCALED_MM):
            pytest.skip("needs block-scaled MXFP8")
        x = (torch.randn(2, 17, 19, cin, device=DEV) * 3).to(torch.float8_e4m3fn)
        weight = (torch.randn(cout, 9 * cin, device=DEV) * 2).to(torch.float8_e4m3fn)
        args = (x, torch.randint(-6, 0, (cin // 32,), dtype=torch.int8, device=DEV), weight,
                torch.randint(-9, -5, (cout, 9 * cin // 32), dtype=torch.int8, device=DEV),
                torch.randn(cout, device=DEV, dtype=torch.bfloat16), 3, 3, 1, 1, 1, 1, 1, 1)
        op, impl = ops.mxfp8_conv2d, "triton_scaled"
    fused, ref = _both(op, impl, *args, relu=True, residual=residual.permute(0, 2, 3, 1))
    assert fused.shape == ref.shape and (fused >= 0).all()
    # the reference rounds the convolution to BF16 before adding: its error scales with the operands, not the sum
    plain = op.get(impl)(*args)
    bound = (plain.float().abs() + residual.permute(0, 2, 3, 1).float().abs()) * 2**-7 + 1e-6
    assert ((fused.float() - ref.float()).abs() <= bound).all()


@pytest.mark.parametrize("batch,tokens,heads,dim", [(2, 4096, 4, 64), (1, 3614, 4, 64), (2, 777, 1, 96)])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("table_dtype", [torch.float32, torch.bfloat16])  # LightGlue's bf16 presets cast the tables
def test_rope_qk_interleaved(batch, tokens, heads, dim, dtype, table_dtype):
    """LightGlue's RoPE on q / k views of the packed QKV output: bit-identical to the FP32 torch reference."""
    torch.manual_seed(0)
    qkv = torch.randn(batch, tokens, 3, heads, dim, device=DEV).to(dtype)
    q, k, _ = qkv.permute(2, 0, 3, 1, 4)
    angles = torch.randn(batch, tokens, dim // 2, device=DEV) * 3
    tables = torch.stack((angles.cos(), angles.sin()), 0).unsqueeze(-3).repeat_interleave(2, dim=-1).to(table_dtype)
    with torch.inference_mode():
        assert ops.rope_qk_interleaved.select(q, k, tables[0], tables[1])[0] == "triton"
        (fq, fk), (rq, rk) = _both(ops.rope_qk_interleaved, "triton", q, k, tables[0], tables[1])
    assert torch.equal(fq, rq) and torch.equal(fk, rk)


@pytest.mark.parametrize("fmt", ["int8", "mxfp8"])
@pytest.mark.parametrize("approximate", ["none", "tanh"])
def test_layernorm_gelu_quantize(fmt, approximate):
    """LayerNorm -> GELU -> quantize in one kernel: the rounding points of nn.LayerNorm -> nn.GELU in BF16."""
    torch.manual_seed(0)
    x = torch.randn(4096, 512, device=DEV, dtype=torch.bfloat16) * 3
    w, b = torch.randn(512, device=DEV, dtype=torch.bfloat16), torch.randn(512, device=DEV, dtype=torch.bfloat16)
    if fmt == "int8":
        op, scale = ops.layernorm_quantize_int8, torch.tensor(0.03, device=DEV)
    else:
        op, scale = ops.layernorm_quantize, mx.calibrate_block32_exponent(torch.randn(8, 512) * 2).to(DEV)
    fast, ref = _both(op, "triton", x, None, w, b, 1e-5, scale, activation=approximate)
    unfused = F.gelu(F.layer_norm(x, (512,), w, b, 1e-5), approximate=approximate)
    expected = (ops.quantize_int8 if fmt == "int8" else ops.quantize_mxfp8)(unfused, scale)
    assert _mismatch(fast.values, ref.values) < 1e-4  # FP32 statistics differ only by summation order
    assert _mismatch(fast.values, expected.values) < 1e-3


@pytest.mark.parametrize("size,coarse_size", [((37, 45), (19, 23)), ((448, 600), (224, 300)), ((64, 80), (8, 10))])
@pytest.mark.parametrize("layout", ["nchw", "nhwc"])
def test_upsample_concat_and_add(size, coarse_size, layout):
    """Bilinear (align_corners=False) upsampling fused with a concat / an add: the reference's values (FP32 rounding
    of the interpolation weights aside)."""
    torch.manual_seed(0)
    fmt = torch.channels_last if layout == "nhwc" else torch.contiguous_format
    features = torch.randn(2, 24, *size, device=DEV, dtype=torch.bfloat16).contiguous(memory_format=fmt)
    coarse = torch.randn(2, 40, *coarse_size, device=DEV).contiguous(memory_format=fmt)
    fast, ref = _both(ops.upsample_concat, "triton", features, coarse)
    assert fast.shape == ref.shape and fast.dtype == ref.dtype and fast.is_contiguous(memory_format=torch.channels_last)
    assert torch.equal(fast[:, :24], ref[:, :24])
    assert ((fast.float() - ref.float()).abs() <= ref.float().abs() * 2**-7 + 1e-6).all()  # within one BF16 ulp
    delta = torch.randn(2, 40, *size, device=DEV, dtype=torch.bfloat16).contiguous(memory_format=fmt)
    fast, ref = _both(ops.upsample_add, "triton", coarse, delta)
    assert fast.dtype == torch.float32
    torch.testing.assert_close(fast, ref, atol=1e-5, rtol=1e-5)
