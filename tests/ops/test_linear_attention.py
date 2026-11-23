"""Tests for ``ops.linear_attention`` (ELU+1 linear attention: Triton vs torch reference)."""

from __future__ import annotations

import pytest
import torch

from flashim import ops


def test_linear_attention_reference_cpu():
    torch.manual_seed(0)
    q = torch.randn(2, 17, 4, 16)
    k = torch.randn(2, 23, 4, 16)
    v = torch.randn(2, 23, 4, 16)
    out = ops.linear_attention(q, k, v)
    assert out.shape == (2, 17, 4, 16) and out.dtype == torch.float32
    assert torch.isfinite(out).all()


@pytest.mark.cuda
@pytest.mark.parametrize(
    ("n", "seq_l", "seq_s", "h", "d", "v_dim"),
    [
        (1, 8190, 8190, 8, 32, 32),  # LoFTR coarse self (840px)
        (1, 8190, 8085, 8, 32, 32),  # LoFTR coarse cross (non-square)
        (2, 4800, 4800, 8, 32, 32),  # batched coarse
        (512, 25, 25, 8, 16, 16),    # LoFTR fine (5x5 windows)
        (1, 1, 1, 8, 32, 32),        # tiny
        (3, 7, 13, 4, 24, 20),       # odd non-power-of-2 dims
        (1, 130, 129, 8, 32, 32),    # non-multiple of BLOCK_S/L=128
        (1, 1025, 1023, 4, 64, 64),  # max supported D=64
    ],
)
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_linear_attention_triton_matches_reference(n, seq_l, seq_s, h, d, v_dim, dtype):
    torch.manual_seed(42)
    q = torch.randn(n, seq_l, h, d, device="cuda", dtype=dtype) * 0.5
    k = torch.randn(n, seq_s, h, d, device="cuda", dtype=dtype) * 0.5
    v = torch.randn(n, seq_s, h, v_dim, device="cuda", dtype=dtype) * 0.5

    assert ops.linear_attention.explain(q, k, v)["triton"] == "selected"
    fast = ops.linear_attention.get("triton")(q, k, v)
    ref = ops.linear_attention.get("reference")(q, k, v)
    assert fast.shape == ref.shape and fast.dtype == ref.dtype

    if dtype == torch.float32:
        torch.testing.assert_close(fast, ref, atol=1e-4, rtol=1e-4)
    else:
        # BF16 dot matches LoFTR's bf16_linear_attention pass (tensor-core BF16 operands, FP32 accum).
        torch.testing.assert_close(fast.float(), ref.float(), atol=3e-2, rtol=2e-2)
        assert (fast.float() - ref.float()).abs().mean().item() < 3e-3


@pytest.mark.cuda
def test_linear_attention_strided_qkv_view():
    """Supports strided QKV slices without a contiguous copy."""
    torch.manual_seed(1)
    qkv = torch.randn(1, 4097, 3, 8, 32, device="cuda", dtype=torch.bfloat16) * 0.5
    q, k, v = qkv[:, :, 0], qkv[:, :, 1], qkv[:, :, 2]
    assert not q.is_contiguous() and q.stride(-1) == 1
    assert ops.linear_attention.explain(q, k, v)["triton"] == "selected"
    fast = ops.linear_attention.get("triton")(q, k, v)
    ref = ops.linear_attention.get("reference")(q, k, v)
    torch.testing.assert_close(fast.float(), ref.float(), atol=3e-2, rtol=2e-2)
