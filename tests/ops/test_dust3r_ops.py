"""Tests for ``ops.reciprocal_nn`` (exact L2 reciprocal nearest neighbors used by DUSt3R)."""

from __future__ import annotations

import pytest
import torch

from flashim import ops


def test_reciprocal_nn_reference_exact():
    torch.manual_seed(0)
    p1 = torch.randn(150, 3) * 10.0 + 50.0  # large offset tests numerical stability (no ||a||^2+||b||^2-2ab loss)
    perm = torch.randperm(150)
    p2 = p1[perm[:120]] + 1e-3 * torch.randn(120, 3)
    # Add 30 distant outliers to p2
    p2 = torch.cat((p2, torch.randn(30, 3) * 10.0 - 50.0), dim=0)

    idx1, idx2 = ops.reciprocal_nn(p1, p2, block_size=32)
    idx1_full, idx2_full = ops.reciprocal_nn(p1, p2, block_size=4096)
    assert torch.equal(idx1, idx1_full) and torch.equal(idx2, idx2_full)
    # All 120 noisy copies must be reciprocal matches to their originals
    matched = set(zip(idx1.tolist(), idx2.tolist()))
    for j in range(120):
        assert (int(perm[j]), j) in matched
    # Verify sorted by idx2
    assert (idx2[1:] > idx2[:-1]).all()


def test_reciprocal_nn_empty():
    empty = torch.empty(0, 3)
    pts = torch.randn(10, 3)
    for a, b in ((empty, pts), (pts, empty), (empty, empty)):
        i1, i2 = ops.reciprocal_nn(a, b)
        assert i1.shape == (0,) and i2.shape == (0,)
        assert i1.dtype == torch.int64 and i2.dtype == torch.int64


@pytest.mark.cuda
@pytest.mark.parametrize("n1,n2", [(257, 311), (1024, 1024), (4096, 3500)])
@pytest.mark.parametrize("impl", ["cuda", "triton"])
def test_reciprocal_nn_cuda_matches_reference(n1: int, n2: int, impl: str):
    torch.manual_seed(42)
    p1 = torch.randn(n1, 3, device="cuda") * 5.0 + torch.tensor([0.0, 0.0, 20.0], device="cuda")
    m = min(n1, n2) // 2
    p2 = torch.cat(
        (
            p1[:m] + 1e-2 * torch.randn(m, 3, device="cuda"),
            torch.randn(n2 - m, 3, device="cuda") * 5.0 + torch.tensor([0.0, 0.0, 20.0], device="cuda"),
        ),
        dim=0,
    )
    with torch.inference_mode():
        i1, i2 = ops.reciprocal_nn.get(impl)(p1, p2)
        r1, r2 = ops.reciprocal_nn.get("reference")(p1, p2)
    assert torch.equal(i1, r1) and torch.equal(i2, r2)


@pytest.mark.cuda
@pytest.mark.parametrize("shape", [(1, 256, 12, 16), (1, 256, 96, 128), (1, 128, 192, 256), (2, 37, 13, 19)])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_upsample_bilinear2x_triton_matches_reference(shape, dtype):
    torch.manual_seed(0)
    x = torch.randn(shape, device="cuda", dtype=dtype).to(memory_format=torch.channels_last)
    assert ops.upsample_bilinear2x.select(x)[0] == "triton"
    with torch.inference_mode():
        out = ops.upsample_bilinear2x.get("triton")(x)
        ref = ops.upsample_bilinear2x.get("reference")(x)
    assert out.is_contiguous(memory_format=torch.channels_last)
    if dtype == torch.float32:
        # Exact integer-quotient coordinates match FP64 reference to <1e-6 (PyTorch's FP32 scale*ox drifts up to 7e-5).
        ref64 = torch.nn.functional.interpolate(x.double(), scale_factor=2, mode="bilinear", align_corners=True).float()
        torch.testing.assert_close(out, ref64, atol=1e-6, rtol=1e-6)
        torch.testing.assert_close(out, ref, atol=1e-4, rtol=1e-4)
    else:
        tol = 1e-3 if dtype == torch.float16 else 1e-2
        torch.testing.assert_close(out, ref, atol=tol, rtol=tol)


@pytest.mark.cuda
@pytest.mark.parametrize("shape", [(1, 4, 384, 512), (1, 4, 352, 512), (2, 4, 37, 53)])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_dust3r_pts3d_conf_triton_matches_reference(shape, dtype):
    torch.manual_seed(0)
    raw = (torch.randn(shape, device="cuda", dtype=dtype) * 0.5).to(memory_format=torch.channels_last)
    assert ops.dust3r_pts3d_conf.select(raw, 1.0)[0] == "triton"
    with torch.inference_mode():
        p_t, c_t = ops.dust3r_pts3d_conf.get("triton")(raw, 1.0)
        p_r, c_r = ops.dust3r_pts3d_conf.get("reference")(raw, 1.0)
    torch.testing.assert_close(p_t, p_r, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(c_t, c_r, atol=1e-5, rtol=1e-5)


@pytest.mark.cuda
@pytest.mark.parametrize("gelu,with_res", [(False, False), (True, False), (False, True)])
def test_int8_linear_fp16_matches_reference(gelu: bool, with_res: bool):
    torch.manual_seed(0)
    a = torch.randint(-127, 128, (256, 512), device="cuda", dtype=torch.int8)
    w = torch.randint(-127, 128, (768, 512), device="cuda", dtype=torch.int8)
    a_s = torch.rand(256, 1, device="cuda", dtype=torch.float32) * 0.02 + 1e-4
    w_s = torch.rand(768, device="cuda", dtype=torch.float32) * 0.02 + 1e-4
    bias = torch.randn(768, device="cuda", dtype=torch.bfloat16) * 0.1
    res = torch.randn(256, 768, device="cuda", dtype=torch.float16) if with_res else None
    with torch.inference_mode():
        out = ops.int8_linear.get("triton")(a, w, a_s, w_s, bias, gelu=gelu, residual=res, out_dtype=torch.float16)
        ref = ops.int8_linear.get("reference")(a, w, a_s, w_s, bias, gelu=gelu, residual=res, out_dtype=torch.float16)
    assert out.dtype == torch.float16
    assert torch.equal(out, ref)


@pytest.mark.cuda
@pytest.mark.parametrize("dim", [768, 1024])
def test_layernorm_quantize_int8_rows_fp16_matches_reference(dim: int):
    torch.manual_seed(0)
    x = torch.randn(128, dim, device="cuda", dtype=torch.float16)
    w = torch.randn(dim, device="cuda", dtype=torch.float16)
    b = torch.randn(dim, device="cuda", dtype=torch.float16)
    with torch.inference_mode():
        q_t, s_t = ops.layernorm_quantize_int8_rows.get("triton")(x, w, b, 1e-6)
        q_r, s_r = ops.layernorm_quantize_int8_rows.get("reference")(x, w, b, 1e-6)
    assert torch.equal(q_t, q_r)
    assert torch.equal(s_t, s_r)
