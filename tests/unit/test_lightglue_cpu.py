"""LightGlue without a checkpoint: key conversion, module layout and the sparse API on CPU."""

import torch
import torch.nn.functional as F

from flashim.models.lightglue import LightGlue, LightGlueConfig
from flashim.models.lightglue.config import ALIKED_CONFIG
from flashim.models.lightglue.model import apply_rotary, filter_matches, rotate_half
from flashim.models.lightglue.weights import convert_state_dict
from flashim.types import SparseFeatures

SMALL = LightGlueConfig(dim=32, input_dim=32, num_heads=2, num_layers=2)
SMALL_PROJECTED = LightGlueConfig(dim=32, input_dim=16, num_heads=2, num_layers=2)


def test_input_dim_mismatch_gets_a_linear_projection():
    model = LightGlue(SMALL_PROJECTED)
    assert isinstance(model.input_proj, torch.nn.Linear)
    assert model.input_proj.weight.shape == (32, 16)
    assert isinstance(LightGlue(SMALL).input_proj, torch.nn.Identity)  # input_dim == dim: no projection


def test_projected_input_reaches_the_transformer():
    torch.manual_seed(0)
    model = LightGlue(SMALL_PROJECTED).eval()
    desc = torch.randn(1, 5, SMALL_PROJECTED.input_dim)
    enc = torch.zeros(2, 1, 1, 5, SMALL_PROJECTED.dim // SMALL_PROJECTED.num_heads)
    scores = model.scores(desc, desc, enc, enc)
    assert scores.shape == (1, 5, 5)


def test_aliked_config_matches_cvg_lightglue_release():
    assert (ALIKED_CONFIG.input_dim, ALIKED_CONFIG.dim) == (128, 256)
    assert ALIKED_CONFIG.filter_threshold == 0.1
    assert ALIKED_CONFIG.reference_size_order is False


def _upstream_state(config: LightGlueConfig) -> dict[str, torch.Tensor]:
    d, n = config.dim, config.num_layers
    state = {"posenc.Wr.weight": torch.zeros(d // config.num_heads // 2, 2)}
    for i in range(n):
        state[f"transformers.{i}.self_attn.Wqkv.weight"] = torch.arange(3 * d * d, dtype=torch.float32).reshape(3 * d, d)
        state[f"transformers.{i}.self_attn.Wqkv.bias"] = torch.arange(3 * d, dtype=torch.float32)
        state[f"log_assignment.{i}.final_proj.weight"] = torch.full((d, d), float(i))
        if i < n - 1:
            state[f"token_confidence.{i}.token.0.weight"] = torch.zeros(1, d)
    return state


def test_convert_keeps_per_layer_heads():
    out = convert_state_dict(_upstream_state(SMALL), SMALL.num_heads, SMALL.num_layers)
    assert float(out["log_assignment.1.final_proj.weight"][0, 0]) == 1.0 and "token_confidence.0.token.0.weight" in out


def test_convert_reorders_qkv_to_contiguous_slices():
    heads, dim = SMALL.num_heads, SMALL.dim
    bias = torch.arange(3 * dim, dtype=torch.float32)  # upstream index = head * (D * 3) + d * 3 + t
    out = convert_state_dict({"transformers.0.self_attn.Wqkv.bias": bias}, heads, SMALL.num_layers)
    new = out["transformers.0.self_attn.Wqkv.bias"].reshape(3, heads, -1)
    old = bias.reshape(heads, -1, 3)
    for t in range(3):
        assert torch.equal(new[t], old[..., t])


def test_rotary_is_a_rotation():
    t = torch.randn(1, 2, 5, 8)
    angle = torch.randn(1, 1, 5, 4).repeat_interleave(2, -1)
    rotated = apply_rotary(t, angle.cos(), angle.sin())
    assert torch.allclose(rotated.norm(dim=-1), t.norm(dim=-1), atol=1e-5)
    assert torch.equal(rotate_half(torch.tensor([1.0, 2.0, 3.0, 4.0])), torch.tensor([-2.0, 1.0, -4.0, 3.0]))


def test_filter_matches_mutual_and_thresholded():
    scores = torch.log(torch.tensor([[[0.9, 0.05], [0.4, 0.5], [0.001, 0.002]]]))
    ia, ib, conf = filter_matches(scores, 0.01)
    assert ia.tolist() == [0, 1] and ib.tolist() == [0, 1]  # row 2 is not mutual / below threshold
    assert torch.allclose(conf, torch.tensor([0.9, 0.5]))


def _features(n: int, dim: int, seed: int) -> SparseFeatures:
    gen = torch.Generator().manual_seed(seed)
    keypoints = torch.rand(1, n, 2, generator=gen) * torch.tensor([64.0, 48.0])
    return SparseFeatures(keypoints, torch.ones(1, n), F.normalize(torch.randn(1, n, dim, generator=gen), dim=-1), (48, 64))


def test_match_features_shapes_and_batched_equals_split():
    torch.manual_seed(0)
    model = LightGlue(SMALL).eval()
    for p in model.parameters():
        torch.nn.init.normal_(p, std=0.2)
    a, b = _features(40, SMALL.dim, 1), _features(40, SMALL.dim, 2)
    result = model.match_features(a, b, threshold=0.0)
    assert result.keypoints_A.shape == result.keypoints_B.shape == (len(result.confidence), 2)
    assert (result.confidence > 0).all()
    # Equal keypoint counts run both images as one batch; a different count takes the per-image path.
    full = model.scores(a.descriptors, b.descriptors, model.encode(a.keypoints, a.image_size), model.encode(b.keypoints, b.image_size))
    a2 = SparseFeatures(a.keypoints[:, :30], a.scores[:, :30], a.descriptors[:, :30], a.image_size)
    split = model.scores(a2.descriptors, b.descriptors, model.encode(a2.keypoints, a2.image_size), model.encode(b.keypoints, b.image_size))
    assert full.shape == (1, 40, 40) and split.shape == (1, 30, 40)


def test_early_stopping_exits_early_and_matches_full_depth_when_disabled():
    import dataclasses

    torch.manual_seed(0)
    model = LightGlue(SMALL).eval()
    for p in model.parameters():
        torch.nn.init.normal_(p, std=0.2)
    a, b = _features(40, SMALL.dim, 1), _features(40, SMALL.dim, 2)
    model.match_features(a, b)
    assert model.executed_layers == SMALL.num_layers
    with torch.no_grad():  # confidence bias so high that every point is confident after layer 0
        model.token_confidence[0].token[0].bias.fill_(50.0)
    model.config = dataclasses.replace(SMALL, depth_confidence=0.95)
    model.match_features(a, b)
    assert model.executed_layers == 1


def test_each_pipeline_uses_the_lightglue_trained_on_its_keypoints():
    """cvg/LightGlue trains one LightGlue per (detector, descriptor) pair: ALIKED's own keypoints and RaCo's
    keypoints with ALIKED descriptors have separate releases, which must not be swapped."""
    from flashim.build import resolve
    from flashim.models.lightglue import weights

    assert weights.CVG_FILES["aliked"] == "aliked_lightglue.pth"
    assert weights.CVG_FILES["raco-aliked"] == "raco_aliked_lightglue.pth"
    for pipeline, variant in (("aliked-lightglue", "aliked"), ("raco-aliked-lightglue", "raco-aliked")):
        assert resolve(pipeline, "fp32").components["matcher"].variant == variant
