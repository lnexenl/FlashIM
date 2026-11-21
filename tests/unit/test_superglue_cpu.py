"""SuperGlue without a checkpoint: weight conversion, BN folding, module layout, and config resolution on CPU."""

import pytest
import torch
import torch.nn.functional as F

from flashim.build import resolve
from flashim.models.superglue import SuperGlue, SuperGlueConfig
from flashim.models.superglue.model import filter_matches, normalize_keypoints
from flashim.models.superglue.weights import convert_state_dict
from flashim.types import SparseFeatures

SMALL = SuperGlueConfig(
    descriptor_dim=32,
    keypoint_encoder=(8, 16),
    gnn_layers=("self", "cross"),
    num_heads=4,
    sinkhorn_iterations=20,
    match_threshold=0.2,
)


def _features(n: int, dim: int, seed: int, size: tuple[int, int] = (48, 64)) -> SparseFeatures:
    gen = torch.Generator().manual_seed(seed)
    keypoints = torch.rand(1, n, 2, generator=gen) * torch.tensor([float(size[1]), float(size[0])])
    scores = torch.rand(1, n, generator=gen)
    descriptors = F.normalize(torch.randn(1, n, dim, generator=gen), dim=-1)
    return SparseFeatures(keypoints, scores, descriptors, size)


def test_normalize_keypoints_uses_superglue_scale():
    kpts = torch.tensor([[[32.0, 24.0], [64.0, 48.0]]])
    norm = normalize_keypoints(kpts, (48, 64))
    # Center is (32, 24), scaling is max(64, 48) * 0.7 = 44.8.
    assert torch.allclose(norm[0, 0], torch.zeros(2), atol=1e-6)
    assert torch.allclose(norm[0, 1], torch.tensor([32.0 / 44.8, 24.0 / 44.8]), atol=1e-6)


def test_convert_state_dict_matches_upstream_conv1d_dh_attention():
    """Converted ``nn.Linear`` + ``(H, D)`` attention matches upstream ``Conv1d(1)`` + ``(D, H)`` attention."""
    torch.manual_seed(0)
    d_model, num_heads = 32, 4
    head_dim = d_model // num_heads
    q_in = torch.randn(1, 10, d_model)
    k_in = torch.randn(1, 12, d_model)
    v_in = torch.randn(1, 12, d_model)

    # Build raw upstream Conv1d(1) weights.
    raw = {
        "gnn.layers.0.attn.merge.weight": torch.randn(d_model, d_model, 1),
        "gnn.layers.0.attn.merge.bias": torch.randn(d_model),
    }
    for i in range(3):
        raw[f"gnn.layers.0.attn.proj.{i}.weight"] = torch.randn(d_model, d_model, 1)
        raw[f"gnn.layers.0.attn.proj.{i}.bias"] = torch.randn(d_model)

    # Upstream channel-first Conv1d + (D, H) attention.
    def conv1d(x, prefix):
        return F.conv1d(x, raw[f"{prefix}.weight"], raw[f"{prefix}.bias"])

    q = conv1d(q_in.transpose(1, 2), "gnn.layers.0.attn.proj.0").view(1, head_dim, num_heads, -1)
    k = conv1d(k_in.transpose(1, 2), "gnn.layers.0.attn.proj.1").view(1, head_dim, num_heads, -1)
    v = conv1d(v_in.transpose(1, 2), "gnn.layers.0.attn.proj.2").view(1, head_dim, num_heads, -1)
    attn_scores = torch.einsum("bdhn,bdhm->bhnm", q, k) / (head_dim**0.5)
    prob = F.softmax(attn_scores, dim=-1)
    out = torch.einsum("bhnm,bdhm->bdhn", prob, v).contiguous().view(1, d_model, -1)
    expected = conv1d(out, "gnn.layers.0.attn.merge").transpose(1, 2)

    # Converted FlashIM MultiHeadedAttention.
    converted = convert_state_dict(raw, num_heads=num_heads)
    model = SuperGlue(SMALL).eval()
    sub = {k.removeprefix("gnn.layers.0.attn."): v for k, v in converted.items()}
    model.gnn.layers[0].attn.load_state_dict(sub, strict=True)
    got = model.gnn.layers[0].attn(q_in, k_in, v_in)
    assert torch.allclose(got, expected, rtol=1e-4, atol=1e-4)


def test_fold_batchnorm_preserves_scores():
    torch.manual_seed(1)
    model = SuperGlue(SMALL).eval()
    for m in model.modules():
        if isinstance(m, torch.nn.BatchNorm1d):
            m.running_mean.normal_()
            m.running_var.uniform_(0.5, 2.0)
            m.weight.data.normal_(1.0, 0.2)
            m.bias.data.normal_(0.0, 0.2)
    a, b = _features(16, SMALL.descriptor_dim, 1), _features(20, SMALL.descriptor_dim, 2)
    before = model.scores(a, b)
    folded = model.fold_batchnorm()
    assert len(folded) == 4  # 2 in kenc.encoder + 1 in each of the 2 gnn layers
    after = model.scores(a, b)
    assert torch.allclose(before, after, atol=1e-5)


def test_batched_and_split_gnn_agree():
    torch.manual_seed(2)
    model = SuperGlue(SMALL).eval()
    a, b = _features(18, SMALL.descriptor_dim, 3), _features(18, SMALL.descriptor_dim, 4)
    # Same-shape pair runs the batched path; slicing one point exercises the split path.
    full = model.scores(a, b)
    assert full.shape == (1, 19, 19)
    # Directly check AttentionalPropagation batched vs per-image calls.
    layer = model.gnn.layers[0]
    x0 = torch.randn(1, 18, SMALL.descriptor_dim)
    x1 = torch.randn(1, 18, SMALL.descriptor_dim)
    for cross in (False, True):
        y0, y1 = layer.forward_pair(x0, x1, cross=cross)
        ref0 = x0 + layer(x0, x1 if cross else x0)
        ref1 = x1 + layer(x1, x0 if cross else x1)
        assert torch.allclose(y0, ref0, atol=1e-5)
        assert torch.allclose(y1, ref1, atol=1e-5)


def test_match_features_and_empty_keypoints():
    torch.manual_seed(3)
    model = SuperGlue(SMALL).eval()
    a, b = _features(24, SMALL.descriptor_dim, 5), _features(20, SMALL.descriptor_dim, 6)
    res = model.match_features(a, b, threshold=0.0)
    assert res.keypoints_A.shape == res.keypoints_B.shape == (len(res.confidence), 2)
    assert res.indices_A is not None and res.indices_B is not None

    empty = _features(0, SMALL.descriptor_dim, 7)
    res_empty = model.match_features(empty, b)
    assert len(res_empty) == 0


def test_filter_matches_mutual_and_thresholded():
    scores = torch.log(torch.tensor([[[0.8, 0.1], [0.3, 0.6], [0.05, 0.06]]]))
    ia, ib, conf = filter_matches(scores, 0.2)
    assert ia.tolist() == [0, 1] and ib.tolist() == [0, 1]
    assert torch.allclose(conf, torch.tensor([0.8, 0.6]))


def test_resolve_superglue_and_pipeline():
    cfg = resolve("superglue", "fp32")
    assert cfg.inference["sinkhorn_iterations"] == 100
    assert cfg.inference["match_threshold"] == 0.2
    assert resolve("superglue", "bf16", overrides={"variant": "indoor"}).variant == "indoor"
    with pytest.raises(ValueError, match="Unknown SuperGlue variant"):
        resolve("superglue", "fp32", overrides={"variant": "bogus"})

    pipe = resolve("superpoint-superglue", "fp32")
    assert pipe.components["extractor"].model == "superpoint"
    assert pipe.components["extractor"].arch == {"nms_radius": 4, "detection_threshold": 0.005, "remove_borders": 4}
    assert pipe.components["matcher"].model == "superglue"
    assert pipe.components["matcher"].variant == "outdoor"
    assert pipe.inference == {"max_matches": 1024, "threshold": 0.2}

    # Quantized presets fall back to bf16 on superglue via build.NEAREST.
    fallbacks = {}
    assert resolve("superglue", "int8", fallbacks=fallbacks).preset == "bf16"
    assert "superglue" in fallbacks
    assert resolve("superglue", "fp8").preset == "bf16"
    assert resolve("superpoint-superglue", "bf16", overrides={"variant": "indoor"}).variant == "indoor"
