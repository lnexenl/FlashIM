"""``ops.linear_attention``: fused Triton ELU+1 attention against the torch reference."""

import pytest
import torch

from flashim import ops

pytestmark = pytest.mark.cuda
DEV = "cuda"


@pytest.mark.parametrize("shape", [
    (1, 16800, 16500, 8, 32),  # LoFTR coarse at 1200 px
    (2, 4800, 4800, 8, 32),    # LoFTR coarse batched self-attention
    (500, 25, 25, 8, 16),      # LoFTR fine 5x5 windows
    (10000, 25, 25, 8, 16),    # LoFTR fine with N*H > 65535 (gridDim.y limit)
    (1, 13, 19, 4, 32),        # tiny ragged
    (1, 300, 500, 2, 64),      # dim=64
])
@pytest.mark.parametrize("dtype,gemm_dtype", [
    (torch.float32, None),
    (torch.bfloat16, torch.bfloat16),
])
def test_linear_attention_matches_reference(shape, dtype, gemm_dtype):
    torch.manual_seed(0)
    n, l_len, s_len, heads, dim = shape
    q = torch.randn(n, l_len, heads, dim, device=DEV, dtype=dtype)
    k = torch.randn(n, s_len, heads, dim, device=DEV, dtype=dtype)
    v = torch.randn(n, s_len, heads, dim, device=DEV, dtype=dtype)
    with torch.inference_mode():
        assert ops.linear_attention.explain(q, k, v, 1e-6, gemm_dtype)["triton"] == "selected"
        fast = ops.linear_attention.get("triton")(q, k, v, 1e-6, gemm_dtype)
        ref = ops.linear_attention.get("reference")(q, k, v, 1e-6, gemm_dtype)
    assert fast.shape == ref.shape == q.shape and fast.dtype == q.dtype
    if dtype == torch.float32:
        torch.testing.assert_close(fast, ref, atol=1e-4, rtol=1e-4)
    else:
        rel = (fast.float() - ref.float()).norm() / ref.float().norm().clamp_min(1e-6)
        assert rel < 1e-2
        assert (fast.float() - ref.float()).abs().max() <= 0.05 * ref.float().abs().max() + 5e-2
