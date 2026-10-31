"""End to end: ``flashim.load("loma-<variant>", preset).match(a, b)`` against upstream ``LoMa.match`` (vendored in
``tests/oracles/loma``) on image paths, and the reduced-precision presets against FlashIM's ``fp32``.

Upstream returns pixel-edge coordinates (``W (x + 1) / 2``); FlashIM's are pixel centers, so the oracle's matches are
shifted by -0.5 px. Matches are compared as sets of coordinate pairs: a match is recovered when the other side has one
with both endpoints within the radius. Thresholds from measurements on sm_120, every variant on both pairs
(``results/loma/parity.md``):

- ``fp32`` vs upstream with every autocast off (``mp=False``, ``extractor_amp=False``): 99.3 to 100% of the oracle's
  matches recovered within 0.05 px, and of FlashIM's in the oracle's (at most 2 of 285 to 477 matches differ:
  threshold-borderline keypoints, T3's extractor parity has 99.95% of the keypoints within 0.01 px); bound 0.99.
- ``bf16`` / ``int8`` / ``fp8`` vs FlashIM ``fp32``, fraction of the fp32 matches recovered within 2 px, minimum over
  the variants: toronto (a hard pair, ~290 matches of 2048 keypoints) 0.875 / 0.746 / 0.721, MegaDepth 0.948 / 0.908 /
  0.880 (``int8``: the rotated residual stream + smoothed ``ffn.3``; 0.745 / 0.901 before). Upstream's own autocast against its FP32 run, for scale: 0.808 / 0.928.
"""

import sys

import pytest
import torch

import flashim
from flashim.models.loma.weights import CHECKPOINTS
from flashim.models.weights_util import checkpoint_dir
from flashim.paths import ASSETS, MEGADEPTH_IMAGES, ORACLES

pytestmark = [pytest.mark.slow, pytest.mark.cuda]

ORACLE = ORACLES / "loma"
VARIANTS = ("b128", "b", "l", "g", "r")
PAIRS = {
    "toronto": (ASSETS / "toronto_A.jpg", ASSETS / "toronto_B.jpg"),
    "megadepth": (MEGADEPTH_IMAGES / "505127_40b48ea397_o.jpg", MEGADEPTH_IMAGES / "505128_561e05e146_o.jpg"),
}
RECALL_FP32 = 0.99
TRACKS = {  # preset -> pair -> bound
    "bf16": {"toronto": 0.85, "megadepth": 0.92},
    "int8": {"toronto": 0.7, "megadepth": 0.87},
    "fp8": {"toronto": 0.68, "megadepth": 0.85},
}


def recovered(ref: tuple[torch.Tensor, torch.Tensor], ours: tuple[torch.Tensor, torch.Tensor], radius: float) -> float:
    """Fraction of the matches ``ref`` (keypoints A, B) with a match in ``ours`` within ``radius`` px on both sides."""
    (ref_a, ref_b), (a, b) = ref, ours
    if len(a) == 0:
        return 0.0
    near_a = torch.cdist(ref_a.double(), a.double(), compute_mode="donot_use_mm_for_euclid_dist") < radius
    near_b = torch.cdist(ref_b.double(), b.double(), compute_mode="donot_use_mm_for_euclid_dist") < radius
    return float((near_a & near_b).any(1).double().mean())


def _flashim(variant: str, preset: str) -> dict:
    model = flashim.load(f"loma-{variant}", preset)
    assert not model.build_report.fallbacks, model.build_report.fallbacks
    with torch.inference_mode():
        out = {}
        for name, (a, b) in PAIRS.items():
            result = model.match(a, b)
            out[name] = (result.keypoints_A, result.keypoints_B)
    del model
    torch.cuda.empty_cache()
    return out


@pytest.fixture(scope="module", params=VARIANTS)
def fp32(request):
    """Per variant: upstream's FP32 matches (pixel centers) and FlashIM ``fp32``'s, per pair; TF32 off for both."""
    variant = request.param
    checkpoint = checkpoint_dir() / CHECKPOINTS[variant].filename
    if not torch.cuda.is_available() or not checkpoint.exists():
        pytest.skip(f"needs CUDA and {checkpoint}")
    sys.path.insert(0, str(ORACLE))
    try:
        import loma.loma  # noqa: F401  (load_oracle imports it lazily, after sys.path is restored)
        from loma_oracle import load_oracle
    finally:
        sys.path.remove(str(ORACLE))
    previous = torch.backends.cudnn.allow_tf32, torch.backends.cuda.matmul.allow_tf32
    torch.backends.cudnn.allow_tf32 = torch.backends.cuda.matmul.allow_tf32 = False
    try:
        oracle = load_oracle(variant, checkpoint, mp=False, extractor_amp=False)
        reference = {}
        with torch.inference_mode():
            for name, (a, b) in PAIRS.items():
                ka, kb = oracle.match(str(a), str(b))
                reference[name] = tuple(torch.as_tensor(k, device="cuda") - 0.5 for k in (ka, kb))
        del oracle
        torch.cuda.empty_cache()
        yield variant, reference, _flashim(variant, "fp32")
    finally:
        torch.backends.cudnn.allow_tf32, torch.backends.cuda.matmul.allow_tf32 = previous


def test_fp32_matches_upstream(fp32):
    variant, reference, ours = fp32
    for name in PAIRS:
        recall, precision = recovered(reference[name], ours[name], 0.05), recovered(ours[name], reference[name], 0.05)
        assert recall > RECALL_FP32 and precision > RECALL_FP32, (variant, name, recall, precision)


@pytest.mark.parametrize("preset", list(TRACKS))
def test_reduced_precision_tracks_fp32(fp32, preset):
    variant, _, ours = fp32
    for name, matches in _flashim(variant, preset).items():
        fraction = recovered(ours[name], matches, 2.0)
        assert fraction > TRACKS[preset][name], (variant, name, preset, fraction)
