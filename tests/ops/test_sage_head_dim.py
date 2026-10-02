"""SageAttention ``sdpa`` impls (needs the optional ``sageattention`` package): head dim 96 is accepted (Sage pads it to 128 itself)."""

import pytest
import torch
import torch.nn.functional as F

pytest.importorskip("sageattention")
pytestmark = pytest.mark.cuda


@pytest.mark.parametrize("variant", ["sage", "sage-fp8"])
@pytest.mark.parametrize("head_dim", [64, 96, 128])
def test_sage_sdpa_matches_reference(variant, head_dim):
    from flashim import ops

    if variant == "sage-fp8" and torch.cuda.get_device_capability() < (8, 9):
        pytest.skip("sage-fp8 needs sm_89+")
    torch.manual_seed(0)
    q, k, v = (torch.randn(1, 1, 512, head_dim, device="cuda", dtype=torch.bfloat16) for _ in range(3))
    out = ops.sdpa.get(variant)(q, k, v)
    ref = F.scaled_dot_product_attention(q, k, v)
    assert out.shape == ref.shape
    cosine = F.cosine_similarity(out.float().flatten(), ref.float().flatten(), dim=0)
    assert cosine > 0.99
