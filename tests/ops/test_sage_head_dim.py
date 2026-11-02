"""SageAttention ``sdpa`` impls (needs the optional ``sageattention`` package): head dim 96 is accepted (Sage pads it to 128 itself)."""

import pytest
import torch
import torch.nn.functional as F

pytest.importorskip("sageattention")
pytestmark = pytest.mark.cuda


@pytest.mark.parametrize("variant", ["sage", "sage-fp8", "sage-fp8-warp", "sage-fp8-sm90", "sage-fp8-sm90-warp"])
@pytest.mark.parametrize("head_dim", [64, 96, 128])
def test_sage_sdpa_matches_reference(variant, head_dim):
    from flashim import ops

    capability = torch.cuda.get_device_capability()
    if variant.startswith("sage-fp8") and capability < (8, 9):
        pytest.skip("sage-fp8 needs sm_89+")
    if "sm90" in variant and capability != (9, 0):
        pytest.skip("the sm_90 kernel needs an sm_90 GPU")
    torch.manual_seed(0)
    q, k, v = (torch.randn(1, 1, 512, head_dim, device="cuda", dtype=torch.bfloat16) for _ in range(3))
    out = ops.sdpa.get(variant)(q, k, v)
    ref = F.scaled_dot_product_attention(q, k, v)
    assert out.shape == ref.shape
    cosine = F.cosine_similarity(out.float().flatten(), ref.float().flatten(), dim=0)
    assert cosine > 0.99


@pytest.mark.parametrize("variant", ["sage-fp8-warp", "sage-fp8-sm90-warp"])
def test_sage_vit_shape(variant):
    """DINOv2-L at 1200 px: 5377 tokens (not a multiple of any tile), 16 heads of 64, one image."""
    from flashim import ops

    capability = torch.cuda.get_device_capability()
    if capability < (8, 9) or ("sm90" in variant and capability != (9, 0)):
        pytest.skip(f"{variant} needs a matching GPU")
    torch.manual_seed(0)
    q, k, v = (torch.randn(1, 16, 5377, 64, device="cuda", dtype=torch.bfloat16) for _ in range(3))
    out, ref = ops.sdpa.get(variant)(q, k, v), F.scaled_dot_product_attention(q, k, v)
    assert (out.float() - ref.float()).norm() / ref.float().norm() < 0.05
