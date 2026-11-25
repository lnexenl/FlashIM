"""Correctness tests for ALIKED fused ops: ``aliked_score_head`` (CUDA C++), ``aliked_sample_patches``,
and ``aliked_bilinear_sample`` (Triton)."""

from __future__ import annotations

import pytest
import torch

from flashim import ops

pytestmark = pytest.mark.cuda
DEV = "cuda"


@pytest.mark.parametrize("hw", [(7, 9), (37, 45), (128, 160), (896, 1216)])
@pytest.mark.parametrize("apply_sigmoid", [True, False])
def test_aliked_score_head_cuda(hw, apply_sigmoid, monkeypatch):
    monkeypatch.setattr(torch.backends.cudnn, "allow_tf32", False)  # kernel accumulates in full FP32
    torch.manual_seed(0)
    head = torch.randn(1, 8, *hw, device=DEV, dtype=torch.float32) * 0.5
    w1 = torch.randn(4, 8, 3, 3, device=DEV, dtype=torch.float32) * 0.2
    w2 = torch.randn(4, 4, 3, 3, device=DEV, dtype=torch.float32) * 0.2
    w3 = torch.randn(1, 4, 3, 3, device=DEV, dtype=torch.float32) * 0.2
    assert ops.aliked_score_head.explain(head, w1, w2, w3, apply_sigmoid=apply_sigmoid)["cuda"] == "selected"
    fast = ops.aliked_score_head.get("cuda")(head, w1, w2, w3, apply_sigmoid=apply_sigmoid)
    ref = ops.aliked_score_head.get("reference")(head, w1, w2, w3, apply_sigmoid=apply_sigmoid)
    assert fast.shape == ref.shape and fast.dtype == torch.float32
    torch.testing.assert_close(fast, ref, atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize("c,size,n", [(64, 3, 128), (128, 3, 4096), (128, 5, 37), (128, 3, 0)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("nhwc", [False, True])
def test_aliked_sample_patches_bit_identical(c, size, n, dtype, nhwc):
    torch.manual_seed(1)
    h, w = 64, 80
    feat = torch.randn(1, c, h, w, device=DEV, dtype=dtype)
    if nhwc:
        feat = feat.contiguous(memory_format=torch.channels_last)
    feat = feat[0]
    corners = torch.cat([torch.randint(0, w, (n, 1)), torch.randint(0, h, (n, 1))], 1).to(DEV)
    if n >= 4:
        corners[:4] = torch.tensor([[0, 0], [w - 1, h - 1], [0, h - 1], [w - 1, 0]], device=DEV)
    assert ops.aliked_sample_patches.explain(feat, corners, size)["triton"] == "selected"
    fast = ops.aliked_sample_patches.get("triton")(feat, corners, size)
    ref = ops.aliked_sample_patches.get("reference")(feat, corners, size)
    assert torch.equal(fast, ref)


@pytest.mark.parametrize("c,m", [(64, 512), (128, 4096 * 16), (128, 37), (128, 0)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("nhwc", [False, True])
def test_aliked_bilinear_sample(c, m, dtype, nhwc):
    torch.manual_seed(2)
    h, w = 64, 80
    feat = torch.randn(1, c, h, w, device=DEV, dtype=dtype)
    if nhwc:
        feat = feat.contiguous(memory_format=torch.channels_last)
    feat = feat[0]
    # Include out-of-bounds, border, and exact integer positions.
    pos = torch.empty(m, 2, device=DEV, dtype=dtype)
    if m > 0:
        pos[:, 0] = torch.rand(m, device=DEV) * (w + 4) - 2.5
        pos[:, 1] = torch.rand(m, device=DEV) * (h + 4) - 2.5
    if m >= 4:
        pos[:4] = torch.tensor([[0.0, 0.0], [w - 1.0, h - 1.0], [-1.0, 5.0], [w + 1.0, 5.0]], device=DEV, dtype=dtype)
    assert ops.aliked_bilinear_sample.explain(feat, pos)["triton"] == "selected"
    fast = ops.aliked_bilinear_sample.get("triton")(feat, pos)
    ref = ops.aliked_bilinear_sample.get("reference")(feat, pos)
    assert fast.shape == ref.shape and fast.dtype == ref.dtype
    tol = 1e-4 if dtype == torch.float32 else 2e-2
    torch.testing.assert_close(fast.float(), ref.float(), atol=tol, rtol=tol)


def test_aliked_bilinear_sample_bf16_fp32_pos():
    """SDDH keeps pixel coordinates in FP32 while sampling BF16 NHWC features."""
    torch.manual_seed(3)
    h, w, c, m = 480, 640, 128, 4096
    feat = torch.randn(1, c, h, w, device=DEV, dtype=torch.bfloat16).contiguous(memory_format=torch.channels_last)[0]
    pos = torch.stack([torch.rand(m, device=DEV) * (w - 1), torch.rand(m, device=DEV) * (h - 1)], dim=1)
    assert ops.aliked_bilinear_sample.explain(feat, pos)["triton"] == "selected"
    fast = ops.aliked_bilinear_sample.get("triton")(feat, pos)
    ref = ops.aliked_bilinear_sample.get("reference")(feat, pos)
    assert fast.dtype == torch.bfloat16
    torch.testing.assert_close(fast.float(), ref.float(), atol=2e-2, rtol=2e-2)
