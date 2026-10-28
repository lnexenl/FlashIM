"""FlashIM's native ALIKED against the reference implementation (github.com/Shiaoming/ALIKED, vendored in
``tests/oracles/aliked``; the released checkpoints come from FlashIM's weight lookup).

The reference's compiled ``custom_ops.get_patches`` CUDA extension is not a build dependency here; both the
reference and FlashIM use the same pure-PyTorch patch-sampling fallback (see ``descriptor.sample_patches``),
so this compares the two implementations' math rather than the presence of a compiled kernel.
"""

import sys

import pytest
import torch

from flashim.models.aliked import weights
from flashim.paths import ORACLES

pytestmark = [pytest.mark.slow]

ALIKED_ROOT = ORACLES / "aliked"
VARIANTS = ["aliked-t16", "aliked-n16", "aliked-n16rot", "aliked-n32"]


@pytest.fixture(scope="module")
def oracle_module():
    sys.path.insert(0, str(ALIKED_ROOT))
    try:
        import nets.aliked as aliked_module
        import nets.blocks as blocks_module
    finally:
        sys.path.remove(str(ALIKED_ROOT))

    def get_patches_lg(tensor: torch.Tensor, required_corners: torch.Tensor, ps: int) -> torch.Tensor:
        c, h, w = tensor.shape
        corner = (required_corners - ps / 2 + 1).long()
        corner[:, 0] = corner[:, 0].clamp(min=0, max=w - 1 - ps)
        corner[:, 1] = corner[:, 1].clamp(min=0, max=h - 1 - ps)
        offset = torch.arange(0, ps)
        x, y = torch.meshgrid(offset, offset, indexing="ij")
        patches = torch.stack((x, y)).permute(2, 1, 0).unsqueeze(2)
        patches = patches.to(corner) + corner[None, None]
        pts = patches.reshape(-1, 2)
        sampled = tensor.permute(1, 2, 0)[tuple(pts.T)[::-1]]
        return sampled.reshape(ps, ps, -1, c).permute(2, 3, 0, 1)

    blocks_module.with_compiled_get_patches = False  # force the same fallback FlashIM uses (see module docstring)
    blocks_module.get_patches_lg = get_patches_lg
    return aliked_module


def _oracle(oracle_module, variant: str):
    oracle = oracle_module.ALIKED(model_name=variant, device="cpu", load_pretrained=False).eval()
    oracle.load_state_dict(weights.load_state_dict(variant))
    return oracle


@torch.inference_mode()
@pytest.mark.parametrize("variant", VARIANTS)
def test_dense_maps_and_sparse_features_match_reference(oracle_module, variant, monkeypatch):
    from flashim.models.aliked import ALIKED

    if not torch.cuda.is_available():  # the reference's forward times itself with torch.cuda.synchronize()
        monkeypatch.setattr(torch.cuda, "synchronize", lambda device=None: None)

    oracle = _oracle(oracle_module, variant)
    ours = ALIKED.from_pretrained(variant=variant, checkpoint=weights.find_checkpoint(variant), device="cpu")

    torch.manual_seed(0)
    image = torch.rand(1, 3, 176, 240)
    ref_feature_map, ref_score_map = oracle.extract_dense_map(image)
    ref = oracle.forward(image)
    feature_map, score_map = ours(image)

    assert torch.equal(score_map, ref_score_map)
    assert torch.equal(feature_map, ref_feature_map)

    out = ours.extract(image)
    ref_kpts, ref_desc = ref["keypoints"][0], ref["descriptors"][0]
    assert out.keypoints.shape[1] == ref_kpts.shape[0]
    pixel_ref = (ref_kpts + 1) / 2 * ref_kpts.new_tensor([image.shape[-1] - 1, image.shape[-2] - 1])
    assert (out.keypoints[0] - pixel_ref).abs().max() < 1e-4
    cosine = (out.descriptors[0] * ref_desc).sum(-1)
    assert cosine.min() > 0.9999  # deformable sampling: summation-order only (see RDD's parity test)


@torch.inference_mode()
def test_bf16_feature_map_tracks_fp32(oracle_module):
    import torch.nn.functional as F

    import flashim

    checkpoint = str(weights.find_checkpoint("aliked-n16"))
    reference = flashim.load("aliked", "fp32", device="cpu", overrides={"checkpoint": checkpoint})
    model = flashim.load("aliked", "bf16", device="cpu", overrides={"checkpoint": checkpoint})

    torch.manual_seed(1)
    image = torch.rand(1, 3, 176, 240)
    ref_features, _ = reference(image)
    features, _ = model(image)
    cosine = F.cosine_similarity(ref_features, features.float(), dim=1)
    assert cosine.mean() > 0.99, f"bf16: mean per-pixel feature cosine {cosine.mean():.4f}"
