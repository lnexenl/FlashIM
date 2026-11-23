"""Tests for ops added for MASt3R: ``ops.rope_qk_2d`` and ``ops.fast_reciprocal_nn``."""

from __future__ import annotations

import pytest
import torch

from flashim import ops


def _croco_rope2d_ref(tokens: torch.Tensor, positions: torch.Tensor, base: float = 100.0) -> torch.Tensor:
    """Direct transcription of CroCo's ``RoPE2D`` formula for checking ``ops.rope_qk_2d``."""
    d_half = tokens.shape[-1] // 2
    inv_freq = 1.0 / (base ** (torch.arange(0, d_half, 2, device=tokens.device, dtype=torch.float32) / d_half))
    t = torch.arange(int(positions.max()) + 1, device=tokens.device, dtype=torch.float32)
    freqs = torch.einsum("i,j->ij", t, inv_freq).to(tokens.dtype)
    freqs = torch.cat((freqs, freqs), dim=-1)
    cos_t, sin_t = freqs.cos(), freqs.sin()

    def rot_half(z: torch.Tensor) -> torch.Tensor:
        z1, z2 = z.chunk(2, dim=-1)
        return torch.cat((-z2, z1), dim=-1)

    def rope1d(z: torch.Tensor, pos1d: torch.Tensor) -> torch.Tensor:
        c = cos_t[pos1d][:, None, :, :]
        s = sin_t[pos1d][:, None, :, :]
        return (z * c) + (rot_half(z) * s)

    y, x = tokens.chunk(2, dim=-1)
    return torch.cat((rope1d(y, positions[..., 0]), rope1d(x, positions[..., 1])), dim=-1)


@pytest.mark.parametrize("h,w,head_dim", [(4, 6, 64), (7, 5, 32), (1, 1, 16)])
def test_rope_qk_2d_matches_croco_formula(h: int, w: int, head_dim: int):
    from flashim.models.mast3r.croco import RoPE2D

    torch.manual_seed(0)
    q = torch.randn(2, 4, h * w, head_dim)
    k = torch.randn(2, 4, h * w, head_dim)
    gy, gx = torch.meshgrid(torch.arange(h), torch.arange(w), indexing="ij")
    pos = torch.stack((gy.reshape(-1), gx.reshape(-1)), dim=-1)[None].expand(2, -1, -1)

    rope = RoPE2D(freq=100.0)
    sin, cos = rope.for_grid(head_dim, h, w, q.device, q.dtype)
    rq, rk = ops.rope_qk_2d(q, k, sin, cos)
    torch.testing.assert_close(rq, _croco_rope2d_ref(q, pos), atol=0.0, rtol=0.0)
    torch.testing.assert_close(rk, _croco_rope2d_ref(k, pos), atol=0.0, rtol=0.0)


def test_rope_qk_2d_cross_attention_different_lengths():
    from flashim.models.mast3r.croco import RoPE2D

    torch.manual_seed(1)
    q = torch.randn(1, 3, 12, 32)  # 3x4
    k = torch.randn(1, 3, 15, 32)  # 5x3
    rope = RoPE2D(freq=100.0)
    sin_q, cos_q = rope.for_grid(32, 3, 4, q.device, q.dtype)
    sin_k, cos_k = rope.for_grid(32, 5, 3, k.device, k.dtype)
    rq, rk = ops.rope_qk_2d(q, k, sin_q, cos_q, sin_k, cos_k)
    assert rq.shape == q.shape and rk.shape == k.shape
    gy_q, gx_q = torch.meshgrid(torch.arange(3), torch.arange(4), indexing="ij")
    pos_q = torch.stack((gy_q.reshape(-1), gx_q.reshape(-1)), dim=-1)[None]
    gy_k, gx_k = torch.meshgrid(torch.arange(5), torch.arange(3), indexing="ij")
    pos_k = torch.stack((gy_k.reshape(-1), gx_k.reshape(-1)), dim=-1)[None]
    torch.testing.assert_close(rq, _croco_rope2d_ref(q, pos_q), atol=0.0, rtol=0.0)
    torch.testing.assert_close(rk, _croco_rope2d_ref(k, pos_k), atol=0.0, rtol=0.0)


def test_fast_reciprocal_nn_exact_shift():
    # Shifted unique normalized descriptors -> every interior sampled pixel converges to exact shift.
    torch.manual_seed(42)
    h, w, d = 24, 32, 24
    base = torch.nn.functional.normalize(torch.randn(h, w, d), dim=-1)
    shift_x, shift_y = 3, 2
    shifted = torch.roll(base, shifts=(shift_y, shift_x), dims=(0, 1))

    xy1, xy2 = ops.fast_reciprocal_nn(base, shifted, subsample=4, block_size=64, dist="dot")
    assert xy1.shape == xy2.shape and xy1.shape[0] > 0
    assert (xy2[:, 0] == (xy1[:, 0] + shift_x) % w).all()
    assert (xy2[:, 1] == (xy1[:, 1] + shift_y) % h).all()

    # Blocking invariance: block_size=32 matches unblocked block_size=4096 bit-for-bit.
    xy1_b, xy2_b = ops.fast_reciprocal_nn(base, shifted, subsample=4, block_size=32, dist="dot")
    assert torch.equal(xy1, xy1_b) and torch.equal(xy2, xy2_b)

    # L2 distance gives the same matches on unit-norm descriptors.
    xy1_l2, xy2_l2 = ops.fast_reciprocal_nn(base, shifted, subsample=4, block_size=64, dist="l2")
    assert torch.equal(xy1, xy1_l2) and torch.equal(xy2, xy2_l2)


def test_fast_reciprocal_nn_tiny_empty():
    desc = torch.randn(2, 2, 8)
    xy1, xy2 = ops.fast_reciprocal_nn(desc, desc, subsample=8)
    assert xy1.shape == (0, 2) and xy2.shape == (0, 2)


@pytest.mark.cuda
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("shape_q,shape_k", [((2, 16, 768, 64), (2, 16, 768, 64)), ((1, 12, 608, 64), (1, 12, 736, 64)), ((1, 4, 35, 32), (1, 4, 19, 32))])
def test_rope_qk_2d_triton_matches_reference(dtype, shape_q, shape_k):
    torch.manual_seed(0)
    q = torch.randn(shape_q, device="cuda", dtype=dtype)
    k = torch.randn(shape_k, device="cuda", dtype=dtype)
    sin_q = torch.randn(shape_q[2], shape_q[3], device="cuda", dtype=dtype)
    cos_q = torch.randn(shape_q[2], shape_q[3], device="cuda", dtype=dtype)
    with torch.inference_mode():
        if shape_q == shape_k:
            assert ops.rope_qk_2d.select(q, k, sin_q, cos_q)[0] == "triton"
            tq, tk = ops.rope_qk_2d.get("triton")(q, k, sin_q, cos_q)
            rq, rk = ops.rope_qk_2d.get("reference")(q, k, sin_q, cos_q)
        else:
            sin_k = torch.randn(shape_k[2], shape_k[3], device="cuda", dtype=dtype)
            cos_k = torch.randn(shape_k[2], shape_k[3], device="cuda", dtype=dtype)
            assert ops.rope_qk_2d.select(q, k, sin_q, cos_q, sin_k, cos_k)[0] == "triton"
            tq, tk = ops.rope_qk_2d.get("triton")(q, k, sin_q, cos_q, sin_k, cos_k)
            rq, rk = ops.rope_qk_2d.get("reference")(q, k, sin_q, cos_q, sin_k, cos_k)
    # Bit-identical in FP32 and BF16 (same operation order and rounding)
    torch.testing.assert_close(tq, rq, atol=0.0, rtol=0.0)
    torch.testing.assert_close(tk, rk, atol=0.0, rtol=0.0)


@pytest.mark.cuda
@pytest.mark.parametrize("rows,cols", [(768, 1024), (768, 4096), (768, 7168), (37, 1792)])
def test_quantize_int8_rows_channel_scale_triton(rows, cols):
    torch.manual_seed(0)
    x = torch.randn(rows, cols, device="cuda", dtype=torch.bfloat16)
    cs = torch.rand(cols, device="cuda", dtype=torch.float32) * 2.0 + 0.1
    assert ops.quantize_int8_rows.select(x, channel_scale=cs)[0] == "triton"
    with torch.inference_mode():
        tq, ts = ops.quantize_int8_rows.get("triton")(x, channel_scale=cs)
        rq, rs = ops.quantize_int8_rows.get("reference")(x, channel_scale=cs)
    assert torch.equal(tq, rq)
    assert torch.equal(ts, rs)


@pytest.mark.cuda
@pytest.mark.parametrize("rows,cols,block", [(1536, 4096, 64), (768, 3072, 64), (768, 7168, 64), (37, 128, 64)])
@pytest.mark.parametrize("with_cs", [False, True])
def test_rotate_quantize_int8_rows_triton(rows, cols, block, with_cs):
    from flashim.quant.rotation import hadamard

    torch.manual_seed(0)
    x = torch.randn(rows, cols, device="cuda", dtype=torch.bfloat16)
    r = hadamard(block, device="cuda", dtype=torch.bfloat16).contiguous()
    cs = (torch.rand(cols, device="cuda", dtype=torch.float32) * 2.0 + 0.1) if with_cs else None
    assert ops.rotate_quantize_int8_rows.select(x, r, channel_scale=cs)[0] == "triton"
    with torch.inference_mode():
        tq, ts = ops.rotate_quantize_int8_rows.get("triton")(x, r, channel_scale=cs)
        rq, rs = ops.rotate_quantize_int8_rows.get("reference")(x, r, channel_scale=cs)
    # BF16 tensor-core dot (with exact +-0.125 H_64) vs FP32 reference:
    # - without channel_scale, BF16 x * exact H_64 accumulates in FP32 -> bit-exact (<= 1 bin on rare ties);
    # - with channel_scale, the kernel rounds (x * cs) to BF16 before tl.dot -> within +-1 bin (~5% of bins).
    torch.testing.assert_close(ts, rs, atol=1e-4, rtol=5e-3)
    diff = (tq.int() - rq.int()).abs()
    assert diff.max().item() <= 2
    assert (diff > 0).float().mean().item() < (0.08 if with_cs else 1e-4)
    if with_cs:
        bq, bs = ops.rotate_quantize_int8_rows.get("reference")((x.float() * cs).to(torch.bfloat16), r)
        torch.testing.assert_close(ts, bs, atol=1e-6, rtol=1e-6)
        assert (tq.int() - bq.int()).abs().max().item() <= 1


@pytest.mark.cuda
@pytest.mark.parametrize("shape", [(2, 16, 768, 64), (1, 12, 768, 64), (1, 12, 608, 64), (2, 8, 37, 64)])
def test_quantize_int8_rows_heads_triton(shape):
    torch.manual_seed(0)
    x = torch.randn(shape, device="cuda", dtype=torch.bfloat16)
    with torch.inference_mode():
        assert ops.quantize_int8_rows_heads.select(x)[0] == "triton"
        tq, ts = ops.quantize_int8_rows_heads.get("triton")(x)
        rq, rs = ops.quantize_int8_rows_heads.get("reference")(x)
    assert torch.equal(tq, rq)
    assert torch.equal(ts, rs)


@pytest.mark.cuda
@pytest.mark.parametrize("shape1,shape2,sub", [
    ((64, 80, 24), (64, 80, 24), 8),
    ((35, 51, 24), (43, 37, 24), 4),
    ((384, 512, 24), (352, 512, 24), 8),
])
def test_fast_reciprocal_nn_triton_matches_reference(shape1, shape2, sub):
    torch.manual_seed(7)
    # Create structured descriptors with genuine reciprocal matches + noise so multiple fixed-point iters run.
    base = torch.nn.functional.normalize(torch.randn(shape1, device="cuda", dtype=torch.float32), dim=-1)
    h2, w2, d = shape2
    desc2 = torch.nn.functional.normalize(torch.randn(shape2, device="cuda", dtype=torch.float32), dim=-1)
    mh, mw = min(shape1[0], h2) - 4, min(shape1[1], w2) - 4
    desc2[2 : 2 + mh, 2 : 2 + mw] = torch.nn.functional.normalize(
        base[:mh, :mw] + 0.05 * torch.randn(mh, mw, d, device="cuda"), dim=-1
    )
    assert ops.fast_reciprocal_nn.select(base, desc2, subsample=sub, dist="dot")[0] == "triton"
    with torch.inference_mode():
        t1, t2 = ops.fast_reciprocal_nn.get("triton")(base, desc2, subsample=sub, dist="dot")
        r1, r2 = ops.fast_reciprocal_nn.get("reference")(base, desc2, subsample=sub, dist="dot")
    assert torch.equal(t1, r1) and torch.equal(t2, r2)


@pytest.mark.cuda
def test_fast_reciprocal_nn_triton_fp16():
    torch.manual_seed(11)
    base = torch.nn.functional.normalize(torch.randn(64, 80, 24, device="cuda"), dim=-1).half()
    desc2 = torch.nn.functional.normalize(torch.randn(64, 80, 24, device="cuda"), dim=-1).half()
    desc2[2:60, 2:76] = torch.nn.functional.normalize(
        base[:58, :74].float() + 0.05 * torch.randn(58, 74, 24, device="cuda"), dim=-1
    ).half()
    assert ops.fast_reciprocal_nn.select(base, desc2, subsample=8, dist="dot")[0] == "triton"
    with torch.inference_mode():
        t1, t2 = ops.fast_reciprocal_nn.get("triton")(base, desc2, subsample=8, dist="dot")
        r1, r2 = ops.fast_reciprocal_nn.get("reference")(base, desc2, subsample=8, dist="dot")
    assert t1.shape[0] > 20
    # FP16 cuBLAS vs FP16 Triton with FP32 accumulation may differ only on exact ties; check >= 99% overlap.
    tset = set(zip(t1[:, 0].tolist(), t1[:, 1].tolist(), t2[:, 0].tolist(), t2[:, 1].tolist()))
    rset = set(zip(r1[:, 0].tolist(), r1[:, 1].tolist(), r2[:, 0].tolist(), r2[:, 1].tolist()))
    assert len(tset & rset) / max(len(rset), 1) >= 0.99
