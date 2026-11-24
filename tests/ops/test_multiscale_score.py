"""Correctness tests for ``ops.multiscale_score`` (4-scale bilinear fusion + 1-ch FP32 projection)."""

from __future__ import annotations

import pytest
import torch

from flashim import ops

pytestmark = pytest.mark.cuda
DEV = "cuda"


@pytest.mark.parametrize("c", [16, 24, 32])
@pytest.mark.parametrize("hw", [(64, 96), (96, 128), (832, 1184)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("nhwc", [False, True])
def test_multiscale_score(c, hw, dtype, nhwc, monkeypatch):
    monkeypatch.setattr(torch.backends.cudnn, "allow_tf32", False)  # kernel accumulates in full FP32, not TF32
    torch.manual_seed(0)
    h, w = hw
    fmt = torch.channels_last if nhwc else torch.contiguous_format
    f1 = torch.randn(1, c, h, w, device=DEV, dtype=dtype).contiguous(memory_format=fmt)
    f2 = torch.randn(1, c, h // 2, w // 2, device=DEV, dtype=dtype).contiguous(memory_format=fmt)
    f3 = torch.randn(1, c, h // 8, w // 8, device=DEV, dtype=dtype).contiguous(memory_format=fmt)
    f4 = torch.randn(1, c, h // 32, w // 32, device=DEV, dtype=dtype).contiguous(memory_format=fmt)
    weight = torch.randn(4 * c, device=DEV, dtype=torch.float32) * 0.1
    bias = torch.randn(1, device=DEV, dtype=torch.float32)
    assert ops.multiscale_score.explain(f1, f2, f3, f4, weight, bias)["triton"] == "selected"
    fast = ops.multiscale_score.get("triton")(f1, f2, f3, f4, weight, bias)
    ref = ops.multiscale_score.get("reference")(f1, f2, f3, f4, weight, bias)
    assert fast.shape == ref.shape and fast.dtype == torch.float32
    torch.testing.assert_close(fast, ref, atol=1e-4, rtol=1e-4)
