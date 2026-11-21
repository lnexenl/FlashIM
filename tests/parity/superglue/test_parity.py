"""FlashIM's SuperGlue and ``superpoint-superglue`` against upstream ``magicleap/SuperGluePretrainedNetwork``
(fetched on demand, see ``_reference.py``: Magic Leap's notice forbids copying it into this repo)."""

import importlib.util
import sys

import pytest
import torch

import flashim
from flashim.paths import ASSETS

from ._reference import reference_root

pytestmark = [pytest.mark.slow, pytest.mark.cuda, pytest.mark.oracle, pytest.mark.weights]

PAIRS = {
    "sacre_coeur": (ASSETS / "sacre_coeur1.jpg", ASSETS / "sacre_coeur2.jpg"),
    "toronto": (ASSETS / "toronto_A.jpg", ASSETS / "toronto_B.jpg"),
}


def _load_upstream_module(variant: str):
    root = reference_root(variant)
    path = root / "models" / "superglue.py"
    spec = importlib.util.spec_from_file_location("_upstream_superglue", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["_upstream_superglue"] = module
    spec.loader.exec_module(module)
    return module


def _upstream_scores_and_matches(upstream_mod, model, a, b):
    """Run upstream ``SuperGlue`` on FlashIM ``SparseFeatures`` and return both the full log-assignment matrix and
    the matches dict."""
    h0, w0 = a.image_size
    h1, w1 = b.image_size
    data = {
        "keypoints0": a.keypoints,
        "keypoints1": b.keypoints,
        "scores0": a.scores,
        "scores1": b.scores,
        "descriptors0": a.descriptors.transpose(1, 2).contiguous(),
        "descriptors1": b.descriptors.transpose(1, 2).contiguous(),
        "image0": torch.empty((1, 1, h0, w0), device=a.keypoints.device),
        "image1": torch.empty((1, 1, h1, w1), device=b.keypoints.device),
    }
    pred = model(data)
    kpts0 = upstream_mod.normalize_keypoints(data["keypoints0"], data["image0"].shape)
    kpts1 = upstream_mod.normalize_keypoints(data["keypoints1"], data["image1"].shape)
    desc0 = data["descriptors0"] + model.kenc(kpts0, data["scores0"])
    desc1 = data["descriptors1"] + model.kenc(kpts1, data["scores1"])
    desc0, desc1 = model.gnn(desc0, desc1)
    mdesc0, mdesc1 = model.final_proj(desc0), model.final_proj(desc1)
    sim = torch.einsum("bdn,bdm->bnm", mdesc0, mdesc1) / (model.config["descriptor_dim"] ** 0.5)
    scores = upstream_mod.log_optimal_transport(sim, model.bin_score, iters=model.config["sinkhorn_iterations"])
    return scores, pred


@pytest.mark.parametrize("variant", ["outdoor", "indoor"])
def test_fp32_scores_and_matches_match_upstream(variant):
    previous = (torch.backends.cudnn.allow_tf32, torch.backends.cuda.matmul.allow_tf32)
    torch.backends.cudnn.allow_tf32 = torch.backends.cuda.matmul.allow_tf32 = False
    try:
        extractor = flashim.load(
            "superpoint",
            "fp32",
            config={"arch": {"nms_radius": 4, "detection_threshold": 0.005, "remove_borders": 4}},
        )
        ours = flashim.load("superglue", "fp32", config={"variant": variant})
        upstream_mod = _load_upstream_module(variant)
        oracle = upstream_mod.SuperGlue(
            {"weights": variant, "sinkhorn_iterations": 100, "match_threshold": 0.2}
        ).cuda().eval()

        img_a, img_b = PAIRS["sacre_coeur"]
        with torch.inference_mode():
            fa = extractor.extract(str(img_a), resize=1024, top_k=1024)
            fb = extractor.extract(str(img_b), resize=1024, top_k=1024)
            ref_scores, ref_pred = _upstream_scores_and_matches(upstream_mod, oracle, fa, fb)
            our_scores = ours.scores(fa, fb)
            our_matches = ours.match_features(fa, fb)
    finally:
        torch.backends.cudnn.allow_tf32, torch.backends.cuda.matmul.allow_tf32 = previous

    prob_diff = float((our_scores.exp() - ref_scores.exp()).abs().max())
    log_diff = float((our_scores - ref_scores).abs().max())
    active = ref_scores > -5.0
    active_diff = float((our_scores[active] - ref_scores[active]).abs().max())
    assert prob_diff < 5e-4, f"{variant}: max prob abs diff = {prob_diff}"
    assert active_diff < 1e-3, f"{variant}: max log score diff (> -5) = {active_diff}"
    assert log_diff < 2e-2, f"{variant}: max log score diff = {log_diff}"

    m0 = ref_pred["matches0"][0]
    valid = m0 > -1
    expected = set(zip(torch.nonzero(valid)[:, 0].tolist(), m0[valid].tolist()))
    got = set(zip(our_matches.indices_A.tolist(), our_matches.indices_B.tolist()))
    min_matches = 100 if variant == "outdoor" else 10
    assert len(expected) >= min_matches
    assert got == expected, f"{variant}: {len(got ^ expected)} differing matches out of {len(expected)}"
    expected_scores = ref_pred["matching_scores0"][0][our_matches.indices_A]
    assert torch.allclose(our_matches.confidence, expected_scores, atol=5e-4)


def test_pipeline_matches_separate_components():
    img_a, img_b = PAIRS["sacre_coeur"]
    pipeline = flashim.load("superpoint-superglue", "fp32")
    extractor = flashim.load(
        "superpoint",
        "fp32",
        config={"arch": {"nms_radius": 4, "detection_threshold": 0.005, "remove_borders": 4}},
    )
    matcher = flashim.load("superglue", "fp32")
    with torch.inference_mode():
        res = pipeline.match(str(img_a), str(img_b), resize=1024)
        fa = extractor.extract(str(img_a), resize=1024, top_k=1024)
        fb = extractor.extract(str(img_b), resize=1024, top_k=1024)
        ref = matcher.match_features(fa, fb)
    assert len(res) == len(ref) > 100
    assert torch.equal(res.indices_A, ref.indices_A)
    assert torch.equal(res.indices_B, ref.indices_B)
    assert torch.allclose(res.confidence, ref.confidence, atol=1e-6)


def test_bf16_tracks_fp32():
    extractor = flashim.load(
        "superpoint",
        "fp32",
        config={"arch": {"nms_radius": 4, "detection_threshold": 0.005, "remove_borders": 4}},
    )
    m_fp32 = flashim.load("superglue", "fp32")
    m_bf16 = flashim.load("superglue", "bf16")
    for name, (img_a, img_b) in PAIRS.items():
        with torch.inference_mode():
            fa = extractor.extract(str(img_a), resize=1024, top_k=1024)
            fb = extractor.extract(str(img_b), resize=1024, top_k=1024)
            ref = m_fp32.match_features(fa, fb)
            bf16 = m_bf16.match_features(fa, fb)
        ref_set = set(zip(ref.indices_A.tolist(), ref.indices_B.tolist()))
        bf16_set = set(zip(bf16.indices_A.tolist(), bf16.indices_B.tolist()))
        agreement = len(ref_set & bf16_set) / max(len(ref_set), 1)
        assert agreement > 0.88, f"{name}: bf16 match agreement {agreement:.4f}"
