"""FlashIM's ``loma-extractor`` against upstream LoMa's ``detect_and_describe`` (vendored in ``tests/oracles/loma``),
both on image paths (upstream's PIL resizes vs :func:`flashim.models.loma.extractor.pil_resize`). One case per distinct
descriptor weight set: b128 (DeDoDe-B, 128-dim), b (= l = g, bit-equal descriptors), r. Downloads the checkpoints on
first use."""

import sys

import pytest
import torch
import torch.nn.functional as F

import flashim
from flashim.models.loma.weights import find_checkpoint
from flashim.paths import ASSETS, IMAGE_A, ORACLES

pytestmark = [pytest.mark.slow, pytest.mark.cuda]

ORACLE = ORACLES / "loma"
IMAGES = [ASSETS / "toronto_A.jpg", IMAGE_A]  # upstream's 0015_A.jpg is not in assets/; IMAGE_A is MegaDepth 0015
VARIANTS = ["b128", "b", "r"]


@pytest.fixture(scope="module")
def load_oracle():
    sys.path.insert(0, str(ORACLE))
    try:
        import loma.loma  # noqa: F401  (load_oracle imports it lazily, after sys.path is restored)
        from loma_oracle import load_oracle
    finally:
        sys.path.remove(str(ORACLE))
    return load_oracle


def _compare(load_oracle, variant: str, preset: str):
    """Per image: distance of each FlashIM keypoint to the nearest upstream one (pixels), and the cosine / relative
    error of the descriptors of the keypoints within 0.5 px of an upstream one."""
    amp = preset != "fp32"  # fp32: upstream with every autocast off and DINOv2 in FP32; bf16: upstream's defaults
    oracle = load_oracle(variant, find_checkpoint(variant), mp=amp, extractor_amp=amp)
    model = flashim.load("loma-extractor", preset, config={"variant": variant})
    out = []
    with torch.inference_mode():
        for path in IMAGES:
            ref_kpts, ref_desc, h, w = oracle.detect_and_describe(str(path))
            ref_px = (ref_kpts[0] + 1) / 2 * ref_kpts.new_tensor([w, h]) - 0.5  # upstream pixel edges -> centers
            features = model.extract(path)
            kpts, desc = features.keypoints[0], features.descriptors[0].float()
            assert kpts.shape == ref_px.shape and desc.shape == ref_desc[0].shape
            dist, index = torch.cdist(kpts, ref_px, compute_mode="donot_use_mm_for_euclid_dist").min(1)
            close = dist < 0.5
            ref = ref_desc[0][index[close]].float()
            cos = F.cosine_similarity(desc[close], ref, dim=-1)
            rel = (desc[close] - ref).norm(dim=-1) / ref.norm(dim=-1)
            out.append((dist, cos, rel))
    del oracle, model
    torch.cuda.empty_cache()
    return out


@pytest.mark.parametrize("variant", VARIANTS)
def test_fp32_matches_oracle(load_oracle, variant, monkeypatch):
    # measured (sm_120): 100% / 99.95% of the 2048 keypoints within 0.01 px (one borderline point flips on the
    # MegaDepth image), descriptor cosine 1.00000, relative error <= 0.003
    monkeypatch.setattr(torch.backends.cudnn, "allow_tf32", False)
    monkeypatch.setattr(torch.backends.cuda.matmul, "allow_tf32", False)
    for dist, cos, rel in _compare(load_oracle, variant, "fp32"):
        assert (dist < 0.01).float().mean() > 0.995
        assert cos.min() > 0.9999 and rel.max() < 0.01


@pytest.mark.parametrize("variant", VARIANTS)
def test_bf16_tracks_oracle_amp(load_oracle, variant):
    # BF16 against upstream's autocast (BF16 VGG / refiners / DINOv2): measured 92.6-93.5% of the keypoints within
    # 0.5 px, descriptor cosine mean >= 0.9996, min >= 0.993 (b128). For scale, upstream's own autocast against its
    # FP32 run: 91.6-92.1% within 0.5 px, cosine min 0.985 (b) / 0.992 (b128); FlashIM bf16 against that FP32 run:
    # 91.4-91.8%, cosine min 0.989 / 0.998.
    for dist, cos, rel in _compare(load_oracle, variant, "bf16"):
        assert (dist < 0.5).float().mean() > 0.9
        assert cos.mean() > 0.999 and cos.min() > 0.98
