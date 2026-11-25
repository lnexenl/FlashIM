"""CUDA op tests for FlexGlue kernels (entmax readout, 3D KDE, QKV/QK-norm attention glue)."""

from __future__ import annotations

import pytest
import torch

from flashim import ops
from flashim.models.flexglue.model import cell_centers

pytestmark = pytest.mark.cuda


@pytest.mark.parametrize(
    "b,n,hw,radius,scale",
    [
        (1, 5, (3, 4), 1, 1.0),
        (2, 17, (7, 9), 1, 5.0),
        (1, 33, (16, 16), 2, 10.0),
        (1, 64, (32, 32), 1, 10.0),
        (1, 128, (64, 64), 1, 10.0),
        (1, 2048, (64, 64), 1, 10.0),
        (1, 4096, (64, 64), 1, 10.0),
        (1, 16, (50, 75), 1, 8.0),  # non-power-of-2 > 512
    ],
)
@torch.inference_mode()
def test_flexglue_entmax_readout(b, n, hw, radius, scale):
    torch.manual_seed(0)
    h, w = hw
    logits = torch.randn(b, n, h * w, device="cuda", dtype=torch.float32) * scale
    grid = cell_centers(h, w, scale=(w, h), device=logits.device)

    ref_pos, ref_mass = ops.flexglue_entmax_readout.get("reference")(logits, grid, hw, radius=radius)
    tri_pos, tri_mass = ops.flexglue_entmax_readout.get("triton")(logits, grid, hw, radius=radius)

    assert ops.flexglue_entmax_readout.explain(logits, grid, hw, radius=radius)["triton"] == "selected"
    assert torch.allclose(tri_pos, ref_pos, atol=1e-5, rtol=1e-5)
    assert torch.allclose(tri_mass, ref_mass, atol=1e-5, rtol=1e-5)


@torch.inference_mode()
def test_flexglue_entmax_readout_edge_cases():
    h, w = 64, 64
    grid = cell_centers(h, w, scale=(w, h), device="cuda")

    # 1. Border corners (argmax at (0,0), (0,w-1), (h-1,0), (h-1,w-1))
    logits = torch.full((1, 4, h * w), -10.0, device="cuda")
    for i, idx in enumerate((0, w - 1, (h - 1) * w, h * w - 1)):
        logits[0, i, idx] = 10.0
    ref_pos, ref_mass = ops.flexglue_entmax_readout.get("reference")(logits, grid, (h, w))
    tri_pos, tri_mass = ops.flexglue_entmax_readout.get("triton")(logits, grid, (h, w))
    assert torch.allclose(tri_pos, ref_pos, atol=1e-6)
    assert torch.allclose(tri_mass, ref_mass, atol=1e-6)

    # 2. Sharp peak (support = 1) and flat cluster (support = 100 equal logits)
    logits = torch.full((1, 2, h * w), -20.0, device="cuda")
    logits[0, 0, 1000] = 5.0
    logits[0, 1, 500:600] = 5.0
    ref_pos, ref_mass = ops.flexglue_entmax_readout.get("reference")(logits, grid, (h, w))
    tri_pos, tri_mass = ops.flexglue_entmax_readout.get("triton")(logits, grid, (h, w))
    assert torch.allclose(tri_pos, ref_pos, atol=1e-6)
    assert torch.allclose(tri_mass, ref_mass, atol=1e-6)


@torch.inference_mode()
def test_kde_density_batched():
    torch.manual_seed(0)
    x = torch.rand(2, 512, 4, device="cuda", dtype=torch.float32) * 2 - 1
    assert ops.kde_density.explain(x, std=0.1)["triton"] == "selected"
    batched = ops.kde_density.get("triton")(x, std=0.1)
    assert batched.shape == (2, 512)
    for i in range(2):
        single = ops.kde_density.get("triton")(x[i], std=0.1)
        assert torch.equal(batched[i], single)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("b,n,h,d,cos_b", [(2, 4096, 8, 64, 1), (1, 17, 4, 32, 1), (2, 77, 8, 48, 2)])
@torch.inference_mode()
def test_qkv_norm_rope_interleaved(dtype, b, n, h, d, cos_b):
    torch.manual_seed(0)
    qkv = torch.randn(b, n, 3 * h * d, device="cuda", dtype=dtype)
    qw, qb = torch.randn(d, device="cuda", dtype=dtype), torch.randn(d, device="cuda", dtype=dtype)
    kw, kb = torch.randn(d, device="cuda", dtype=dtype), torch.randn(d, device="cuda", dtype=dtype)
    theta = torch.randn(cos_b, n, d // 2, device="cuda", dtype=torch.float32)
    cos = theta.cos().repeat_interleave(2, dim=-1)
    sin = theta.sin().repeat_interleave(2, dim=-1)

    assert ops.qkv_norm_rope_interleaved.explain(qkv, qw, qb, kw, kb, 1e-5, cos, sin, h)["triton"] == "selected"
    rq, rk, rv = ops.qkv_norm_rope_interleaved.get("reference")(qkv, qw, qb, kw, kb, 1e-5, cos, sin, h)
    tq, tk, tv = ops.qkv_norm_rope_interleaved.get("triton")(qkv, qw, qb, kw, kb, 1e-5, cos, sin, h)

    assert torch.equal(tv, rv)
    tol = 2e-2 if dtype == torch.bfloat16 else 1e-5
    assert torch.allclose(tq.float(), rq.float(), atol=tol, rtol=tol)
    assert torch.allclose(tk.float(), rk.float(), atol=tol, rtol=tol)
    if dtype == torch.bfloat16:
        # At most a tiny fraction of 1-ulp BF16 rounding differences from LayerNorm reduction order.
        diff_q = (tq != rq).float().mean().item()
        diff_k = (tk != rk).float().mean().item()
        assert diff_q < 0.01 and diff_k < 0.01


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("b,n,h,d", [(2, 4096, 8, 64), (4, 33, 4, 48)])
@torch.inference_mode()
def test_qk_norm_swap_heads(dtype, b, n, h, d):
    torch.manual_seed(0)
    qk = torch.randn(b, n, h * d, device="cuda", dtype=dtype)
    v = torch.randn(b, n, h * d, device="cuda", dtype=dtype)
    w, bias = torch.randn(d, device="cuda", dtype=dtype), torch.randn(d, device="cuda", dtype=dtype)

    assert ops.qk_norm_swap_heads.explain(qk, v, w, bias, 1e-5, h)["triton"] == "selected"
    rq, rk, rv = ops.qk_norm_swap_heads.get("reference")(qk, v, w, bias, 1e-5, h)
    tq, tk, tv = ops.qk_norm_swap_heads.get("triton")(qk, v, w, bias, 1e-5, h)

    assert torch.equal(tv, rv)
    tol = 2e-2 if dtype == torch.bfloat16 else 1e-5
    assert torch.allclose(tq.float(), rq.float(), atol=tol, rtol=tol)
    assert torch.allclose(tk.float(), rk.float(), atol=tol, rtol=tol)
    assert torch.equal(tk[: b // 2], tq[b // 2 :]) and torch.equal(tk[b // 2 :], tq[: b // 2])


@pytest.mark.parametrize("b,h,n,d", [(2, 8, 4096, 64), (1, 4, 77, 32)])
@torch.inference_mode()
def test_quantize_mxfp8_heads(b, h, n, d):
    torch.manual_seed(0)
    x = torch.randn(b, h, n, d, device="cuda", dtype=torch.bfloat16)
    exp = torch.randint(-4, 4, (h * d // 32,), device="cuda", dtype=torch.int8)
    assert ops.quantize_mxfp8_heads.explain(x, exp)["triton"] == "selected"
    ref = ops.quantize_mxfp8_heads.get("reference")(x, exp)
    tri = ops.quantize_mxfp8_heads.get("triton")(x, exp)
    assert torch.equal(tri.values.view(torch.uint8), ref.values.view(torch.uint8))


@pytest.mark.parametrize("rows,c", [(8192, 512), (77, 128)])
@torch.inference_mode()
def test_layernorm_cat_quantize_mxfp8_and_int8(rows, c):
    torch.manual_seed(0)
    x = torch.randn(2, rows // 2, c, device="cuda", dtype=torch.bfloat16)
    msg = torch.randn_like(x)
    w, b = torch.randn(c, device="cuda", dtype=torch.bfloat16), torch.randn(c, device="cuda", dtype=torch.bfloat16)
    msg_bias = torch.randn(c, device="cuda", dtype=torch.float32) * 0.1
    exp = torch.randint(-3, 3, (2 * c // 32,), device="cuda", dtype=torch.int8)

    assert ops.layernorm_cat_quantize_mxfp8.explain(x, msg, w, b, 1e-5, exp, input_bias=msg_bias)["triton"] == "selected"
    ref_fp8 = ops.layernorm_cat_quantize_mxfp8.get("reference")(x, msg, w, b, 1e-5, exp, input_bias=msg_bias)
    tri_fp8 = ops.layernorm_cat_quantize_mxfp8.get("triton")(x, msg, w, b, 1e-5, exp, input_bias=msg_bias)
    # Left half (x) is bit-exact; right half (LayerNorm) allows tiny 1-ulp rounding diff.
    assert torch.equal(tri_fp8.values[..., :c].view(torch.uint8), ref_fp8.values[..., :c].view(torch.uint8))
    diff_fp8 = (tri_fp8.values[..., c:].view(torch.uint8) != ref_fp8.values[..., c:].view(torch.uint8)).float().mean().item()
    assert diff_fp8 < 0.005

    scale = torch.tensor(0.05, device="cuda", dtype=torch.float32)
    assert ops.layernorm_cat_quantize_int8.explain(x, msg, w, b, 1e-5, scale)["triton"] == "selected"
    ref_i8 = ops.layernorm_cat_quantize_int8.get("reference")(x, msg, w, b, 1e-5, scale)
    tri_i8 = ops.layernorm_cat_quantize_int8.get("triton")(x, msg, w, b, 1e-5, scale)
    assert torch.equal(tri_i8.values[..., :c], ref_i8.values[..., :c])
    assert (tri_i8.values[..., c:].int() - ref_i8.values[..., c:].int()).abs().max().item() <= 1


@pytest.mark.parametrize("channels_last", [False, True])
@pytest.mark.parametrize("c,hw,n", [(512, (64, 64), 2048), (512, (64, 64), 4096), (2, (64, 64), 4096), (37, (17, 23), 99)])
@torch.inference_mode()
def test_bilinear_sample_points_border(channels_last, c, hw, n):
    torch.manual_seed(0)
    feat = torch.randn(1, c, *hw, device="cuda", dtype=torch.float32)
    if channels_last:
        feat = feat.to(memory_format=torch.channels_last)
    # Include out-of-range coordinates to exercise border clamping.
    pts = torch.rand(n, 2, device="cuda", dtype=torch.float32) * 2.6 - 1.3
    kw = {"align_corners": False, "padding_mode": "border", "out_dtype": feat.dtype}
    assert ops.bilinear_sample_points.explain((feat,), pts, **kw)["triton"] == "selected"
    ref = ops.bilinear_sample_points.get("reference")((feat,), pts, **kw)
    tri = ops.bilinear_sample_points.get("triton")((feat,), pts, **kw)
    assert torch.allclose(tri, ref, atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize("m,n,d", [(2048, 2048, 4), (77, 133, 4), (0, 10, 4), (1, 1, 2)])
@torch.inference_mode()
def test_reciprocal_nn_tiled(m, n, d):
    torch.manual_seed(0)
    x = torch.randn(m, d, device="cuda", dtype=torch.float32) * 100.0
    y = torch.randn(n, d, device="cuda", dtype=torch.float32) * 100.0
    assert ops.reciprocal_nn.explain(x, y)["triton_tiled"] == "selected"
    r1, r2 = ops.reciprocal_nn.get("reference")(x, y)
    t1, t2 = ops.reciprocal_nn.get("triton_tiled")(x, y)
    assert torch.equal(t1, r1) and torch.equal(t2, r2)
