"""CPU unit tests for OmniGlue (architecture, BN folding, DINO mask, Sinkhorn, SuperPoint sp_v6, registry/specs)."""

from __future__ import annotations

import pytest
import torch

import flashim
from flashim.models.omniglue.config import OmniGlueConfig
from flashim.models.omniglue.extractor import (
    OmniGlueExtractor,
    dino_size,
    sample_dino_descriptors,
    superpoint_size,
)
from flashim.models.omniglue.model import (
    OmniGlueFeatures,
    OmniGlueMatcher,
    dino_attention_mask,
    log_optimal_transport,
    normalize_keypoints,
)
from flashim.models.superpoint.model import SuperPoint


def test_registry_and_resolve():
    for name in ("omniglue", "omniglue-extractor", "omniglue-matcher"):
        assert name in flashim.available_models()
    cfg = flashim.resolve("omniglue", "bf16")
    assert cfg.components["extractor"].model == "omniglue-extractor"
    assert cfg.components["matcher"].model == "omniglue-matcher"
    cfg_int8 = flashim.resolve("omniglue", "int8")
    assert cfg_int8.components["extractor"].preset == "int8"
    assert cfg_int8.components["matcher"].preset == "int8"
    fb: dict[str, str] = {}
    cfg_fp8 = flashim.resolve("omniglue", "fp8", fallbacks=fb)
    assert cfg_fp8.preset == "int8"
    assert cfg_fp8.components["extractor"].preset == "int8"
    assert cfg_fp8.components["matcher"].preset == "int8"
    assert "omniglue" in fb
    # Component extractor still exposes its calibrated fp8 preset when requested directly.
    assert flashim.resolve("omniglue-extractor", "fp8").preset == "fp8"
    assert flashim.resolve("superpoint", "fp32", overrides={"variant": "sp_v6"}).variant == "sp_v6"
    with pytest.raises(ValueError, match="unknown variant"):
        flashim.resolve("superpoint", "fp32", overrides={"variant": "nope"})


def test_size_helpers():
    assert superpoint_size(731, 1024) == (728, 1024)
    assert superpoint_size(732, 1024) == (736, 1024)
    assert superpoint_size(480, 640) == (480, 640)
    assert dino_size(731, 1024) == (448, 630)
    assert dino_size(1024, 768) == (630, 462)
    assert dino_size(300, 400) == (294, 392)


def test_dino_attention_mask_strict_half():
    torch.manual_seed(0)
    d0 = torch.randn(1, 16, 32)
    d1 = torch.randn(1, 10, 32)
    m0, m1 = dino_attention_mask(d0, d1)
    assert m0.shape == (1, 1, 16, 10) and m0.dtype == torch.bool
    assert m1.shape == (1, 1, 10, 16) and m1.dtype == torch.bool
    # Strict > ceil(0.5 * L)-th value => ceil(0.5*L) - 1 True entries per row (when distinct)
    assert (m0.sum(dim=-1) == 4).all()
    assert (m1.sum(dim=-1) == 7).all()


def test_log_optimal_transport_doubly_stochastic():
    torch.manual_seed(0)
    scores = torch.randn(1, 12, 15)
    p = log_optimal_transport(scores, torch.tensor(1.0), iters=100)
    assert p.shape == (1, 13, 16)
    assert (p >= 0).all()
    # Non-dustbin rows/cols sum to ~1 (marginals are 1 for keypoints, M/N for dustbins)
    assert torch.allclose(p[0, :-1].sum(-1), torch.ones(12), atol=1e-3)
    assert torch.allclose(p[0, :, :-1].sum(0), torch.ones(15), atol=1e-3)


def test_matcher_bn_fold_exact_and_shapes():
    torch.manual_seed(0)
    cfg = OmniGlueConfig(descriptor_dim=32, keypoint_encoder=(16, 32), num_heads=4, num_layers=1, sinkhorn_iterations=10)
    model = OmniGlueMatcher(cfg).eval()
    # Give BN non-trivial stats
    for m in model.modules():
        if isinstance(m, torch.nn.BatchNorm1d):
            m.running_mean.normal_()
            m.running_var.uniform_(0.5, 2.0)
            m.weight.data.uniform_(0.5, 1.5)
            m.bias.data.normal_()

    kp0 = torch.rand(1, 12, 2) * 100
    kp1 = torch.rand(1, 15, 2) * 100
    desc0, desc1 = torch.randn(1, 12, 32), torch.randn(1, 15, 32)
    sc0, sc1 = torch.rand(1, 12) + 0.1, torch.rand(1, 15) + 0.1
    dino0, dino1 = torch.randn(1, 12, 16), torch.randn(1, 15, 16)

    with torch.inference_mode():
        before = model.soft_assignment(kp0, kp1, desc0, desc1, sc0, sc1, dino0, dino1, (100, 100), (100, 100))
        folded = model.fold_batchnorm()
        assert len(folded) == len(cfg.keypoint_encoder) + 2 * cfg.num_layers
        packed = model.pack_qk()
        assert len(packed) == 2 * cfg.num_layers
        after = model.soft_assignment(kp0, kp1, desc0, desc1, sc0, sc1, dino0, dino1, (100, 100), (100, 100))
    assert (before - after).abs().max() < 1e-5

    # Also test equal keypoint count (batched path) and match_features
    fa = OmniGlueFeatures(kp0, sc0, desc0, (100, 100), dino0)
    fb = OmniGlueFeatures(kp0.clone(), sc0.clone(), desc0.clone(), (100, 100), dino0.clone())
    m = model.match_features(fa, fb, threshold=0.0)
    assert m.keypoints_A.shape[1] == 2 and len(m.confidence) == len(m.keypoints_A)

    # Empty keypoints edge case
    empty_f = OmniGlueFeatures(torch.zeros(1, 0, 2), torch.zeros(1, 0), torch.zeros(1, 0, 32), (100, 100), torch.zeros(1, 0, 16))
    m_empty = model.match_features(empty_f, fb)
    assert len(m_empty) == 0


def test_superpoint_sp_v6_bn_fold_exact_cpu():
    torch.manual_seed(0)
    sp = SuperPoint(variant="sp_v6").eval()
    for m in sp.modules():
        if isinstance(m, torch.nn.BatchNorm2d):
            m.running_mean.normal_()
            m.running_var.uniform_(0.5, 2.0)
            m.weight.data.uniform_(0.5, 1.5)
            m.bias.data.normal_()
    img = torch.rand(1, 3, 64, 64)
    with torch.inference_mode():
        s0, d0 = sp(img)
        folded = sp.fold_batchnorm()
        assert folded == ["convPb", "convDb"]
        relu_folded = sp.fold_relu_batchnorm()
        assert len(relu_folded) == 10
        s1, d1 = sp(img)
    assert (s0 - s1).abs().max() < 1e-6
    assert (d0 - d1).abs().max() < 1e-6


def test_sample_dino_descriptors_boundary():
    fmap = torch.arange(1, 1 + 4 * 2 * 3, dtype=torch.float32).reshape(1, 4, 2, 3)
    kpts = torch.tensor([[0.0, 0.0], [10.0, 5.0], [29.0, 19.0]])
    out = sample_dino_descriptors(fmap, kpts, orig_height=20, orig_width=30)
    assert out.shape == (3, 4)
    assert torch.allclose(out[0], fmap[0, :, 0, 0])
    assert normalize_keypoints(kpts[None], (20, 30)).shape == (1, 3, 2)
    assert OmniGlueExtractor(OmniGlueConfig(dino_depth=1, dino_dim=64, dino_heads=4)).batchnorm_pairs() == [
        ("superpoint.convPb", "superpoint.bnPb"),
        ("superpoint.convDb", "superpoint.bnDb"),
    ]
