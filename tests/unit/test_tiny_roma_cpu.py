"""CPU unit tests for Tiny-RoMa (no checkpoint download required)."""

from __future__ import annotations

import pytest
import torch

import flashim
from flashim.models.tiny_roma import TinyRoMa, TinyRoMaConfig


def _random_model(**kwargs) -> TinyRoMa:
    torch.manual_seed(0)
    model = TinyRoMa(TinyRoMaConfig(**kwargs)).eval()
    for m in model.modules():
        if isinstance(m, torch.nn.BatchNorm2d):
            m.running_mean.uniform_(-0.2, 0.2)
            m.running_var.uniform_(0.5, 1.5)
    return model


def test_resolve_presets_and_spec():
    for preset in ("fp32", "bf16", "int8", "fp8"):
        cfg = flashim.resolve("tiny-roma", preset)
        assert cfg.model == "tiny-roma"
    assert flashim.resolve("tiny-roma").preset == "bf16"


@pytest.mark.parametrize("exact_softmax", [False, True])
def test_forward_shapes_and_bn_fold_exact(exact_softmax):
    model = _random_model(exact_softmax=exact_softmax)
    assert len(model.batchnorm_pairs()) == 26
    a = torch.rand(1, 3, 130, 165)
    b = torch.rand(1, 3, 130, 165)
    ref = model(a, b)
    folded = model.fold_batchnorm()
    assert len(folded) == 26
    out = model(a, b)
    for scale, expected_hw in ((8, (16, 20)), (4, (32, 40))):
        assert ref[scale]["flow"].shape == (1, 2, *expected_hw)
        assert ref[scale]["certainty"].shape == (1, 1, *expected_hw)
        torch.testing.assert_close(out[scale]["flow"], ref[scale]["flow"], atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(out[scale]["certainty"], ref[scale]["certainty"], atol=1e-5, rtol=1e-5)


def test_different_aspect_ratios_and_match_api():
    model = _random_model()
    model.fold_batchnorm()
    a = torch.rand(1, 3, 135, 195)
    b = torch.rand(1, 3, 165, 130)
    dense = model.match_dense(a, b)
    assert dense.warp_AB.shape == (1, 135, 195, 2)
    assert dense.confidence_AB.shape == (1, 135, 195, 1)
    assert dense.overlap_AB.shape == (1, 135, 195, 1)
    assert dense.precision_AB is None
    assert not dense.bidirectional

    g1 = torch.Generator().manual_seed(42)
    g2 = torch.Generator().manual_seed(42)
    m1 = model.match(a, b, max_matches=128, generator=g1)
    m2 = model.match(a, b, max_matches=128, generator=g2)
    assert len(m1) == 128
    assert m1.keypoints_A.shape == m1.keypoints_B.shape == (128, 2)
    assert m1.confidence.shape == (128,)
    assert m1.dense is not None
    assert torch.equal(m1.keypoints_A, m2.keypoints_A)
    assert torch.equal(m1.keypoints_B, m2.keypoints_B)
    assert (m1.keypoints_A[:, 0] >= 0).all() and (m1.keypoints_A[:, 0] <= 195).all()
    assert (m1.keypoints_A[:, 1] >= 0).all() and (m1.keypoints_A[:, 1] <= 135).all()

    resized = model.match(a, b, resize=128, max_matches=64, threshold=0.1, generator=torch.Generator().manual_seed(0))
    assert len(resized) == 64
    assert resized.dense.warp_AB.shape[1] <= 128 and resized.dense.warp_AB.shape[2] <= 128


def test_bf16_runs_on_cpu():
    model = _random_model()
    model.fold_batchnorm()
    model.set_compute_dtype(torch.bfloat16)
    assert model.coarse_matcher[-1].weight.dtype == torch.float32
    assert model.fine_matcher[-1].weight.dtype == torch.float32
    assert model.coarse_matcher[0].layer[0].weight.dtype == torch.bfloat16
    a = torch.rand(1, 3, 128, 128)
    b = torch.rand(1, 3, 128, 128)
    m = model.match(a, b, max_matches=64, generator=torch.Generator().manual_seed(1))
    assert len(m) == 64 and torch.isfinite(m.keypoints_A).all() and torch.isfinite(m.keypoints_B).all()
