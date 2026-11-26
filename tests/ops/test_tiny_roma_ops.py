"""CUDA correctness tests for Tiny-RoMa fused ops (``tiny_roma_corr_pos_embed``, ``gray_instance_norm``)."""

from __future__ import annotations

import pytest
import torch

from flashim import ops

pytestmark = pytest.mark.cuda
DEV = "cuda"


@pytest.mark.parametrize("exact_softmax", [False, True])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize(
    "shape0,shape1",
    [
        ((1, 64, 4, 4), (1, 64, 4, 4)),
        ((2, 64, 17, 23), (2, 64, 16, 20)),
        ((1, 64, 88, 128), (1, 64, 128, 96)),  # sacre_coeur real call-site shape
        ((1, 64, 96, 128), (1, 64, 92, 128)),  # MegaDepth resize=1024 call-site shape
    ],
)
def test_tiny_roma_corr_pos_embed_matches_reference(shape0, shape1, dtype, exact_softmax):
    torch.manual_seed(0)
    f0 = torch.randn(shape0, device=DEV, dtype=dtype)
    f1 = torch.randn(shape1, device=DEV, dtype=dtype)
    assert ops.tiny_roma_corr_pos_embed.explain(f0, f1, exact_softmax)["triton"] == "selected"
    fast = ops.tiny_roma_corr_pos_embed.get("triton")(f0, f1, exact_softmax)
    ref = ops.tiny_roma_corr_pos_embed.get("reference")(f0, f1, exact_softmax)
    assert fast.shape == ref.shape and fast.dtype == torch.float32
    diff = (fast - ref).abs()
    if dtype == torch.float32:
        # Exact FP32 dot product (`input_precision="ieee"`): only FP32 summation order differences.
        assert diff.max().item() < 1e-5, f"fp32 max diff {diff.max().item():.3e}"
    else:
        # Triton keeps FP32 dot-product logits in registers (avoiding the reference's BF16 correlation volume
        # truncation before argmax/softmax), so it matches the exact FP32 correlation of those BF16 features.
        ref_fp32 = ops.tiny_roma_corr_pos_embed.get("reference")(f0.float(), f1.float(), exact_softmax)
        assert (fast - ref_fp32).abs().max().item() < 1e-5, f"vs fp32-dot max diff {(fast - ref_fp32).abs().max().item():.3e}"
        if exact_softmax:
            assert diff.max().item() < 1e-2 and diff.median().item() < 1e-3


@pytest.mark.parametrize("out_dtype", [None, torch.float32, torch.bfloat16])
@pytest.mark.parametrize("shape", [(1, 3, 32, 32), (2, 3, 137, 211), (1, 3, 704, 1024), (2, 3, 768, 1024)])
def test_gray_instance_norm_matches_reference(shape, out_dtype):
    torch.manual_seed(1)
    x = torch.rand(shape, device=DEV, dtype=torch.float32)
    assert ops.gray_instance_norm.explain(x, 1e-5, out_dtype)["triton"] == "selected"
    fast = ops.gray_instance_norm.get("triton")(x, 1e-5, out_dtype)
    ref = ops.gray_instance_norm.get("reference")(x, 1e-5, out_dtype)
    assert fast.shape == ref.shape and fast.dtype == ref.dtype
    tol = 2e-5 if out_dtype in (None, torch.float32) else 2e-2
    assert (fast.float() - ref.float()).abs().max().item() < tol
    if out_dtype == torch.bfloat16:
        assert (fast.float() - ref.float()).abs().mean().item() < 1e-4
