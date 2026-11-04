"""FlashIM's native FlexGlue against the reference implementation (cvg/FlexGlue, /tmp/flexglue_upstream).

(a) fp32 parity: upstream ``flexglue.FlexGlue`` vs ``flashim.load('flexglue', 'fp32')`` on the same
MegaDepth pair and the same 2048 RDD keypoints; the mutual-NN match index sets must overlap almost
exactly (upstream is fp32-eager with autocast off, FlashIM's fp32 preset is the checkpoint precision).

(b) reduced-precision tracking: bf16 / fp8 / int8 against the fp32 FlashIM result on the same pair;
quantized transformer linears shift the search maps slightly and a few threshold-borderline matches
flip (a few indices shift every later one), so the requirement is a > 0.85 shared fraction.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

from ..conftest import MEGADEPTH_IMAGES

pytestmark = [pytest.mark.cuda, pytest.mark.slow]

UPSTREAM = Path("/tmp/flexglue_upstream")
# the stable MegaDepth-0015 pair (505127/505128, ~343 matches at 2048 RDD keypoints); the
# 33831624/299056923 pair used by the LightGlue test gives FlexGlue only ~56 matches, all of
# them within ~0.05 of the 0.1 confidence threshold, so reduced-precision noise flips half of
# them regardless of the implementation (measured IoU 0.32-0.45 vs fp32)
IMAGES = [MEGADEPTH_IMAGES / "505127_40b48ea397_o.jpg", MEGADEPTH_IMAGES / "505128_561e05e146_o.jpg"]


@pytest.fixture(scope="module")
def checkpoint():
    from flashim.models.flexglue.weights import find_checkpoint

    try:
        return find_checkpoint()
    except FileNotFoundError as exc:
        pytest.skip(str(exc))


@pytest.fixture(scope="module")
def rdd_keypoints():
    if not all(path.exists() for path in IMAGES):
        pytest.skip("needs MegaDepth images under assets/megadepth_0015")
    import flashim

    rdd = flashim.load("rdd", preset="bf16", device="cuda")
    kpts = [rdd.extract(str(path), top_k=2048).keypoints.reshape(-1, 2) for path in IMAGES]
    return [torch.from_numpy(k).float().cuda() if not isinstance(k, torch.Tensor) else k.float().cuda() for k in kpts]


def _load_image_tensor(path: Path) -> torch.Tensor:
    from flashim.models.romav2.model import load_image

    return load_image(str(path), torch.device("cuda"))


def _load_upstream(checkpoint: Path):
    if not (UPSTREAM / "flexglue" / "flexglue.py").exists():
        pytest.skip(f"needs the upstream FlexGlue checkout at {UPSTREAM}")
    sys.path.insert(0, str(UPSTREAM))
    try:
        import flexglue
    except ImportError as exc:
        pytest.skip(f"upstream flexglue does not import: {exc}")
    # mp=None: run upstream's transformer in fp32 (no autocast), matching FlashIM's fp32 preset
    model = flexglue.FlexGlue(weights=str(checkpoint), mp=None).cuda().eval()
    return model


@torch.inference_mode()
def test_fp32_matches_upstream(checkpoint, rdd_keypoints):
    upstream = _load_upstream(checkpoint)

    image_a, image_b = (_load_image_tensor(path) for path in IMAGES)
    kpts_a, kpts_b = rdd_keypoints

    expected = upstream.match(
        {"image": image_a, "keypoints": kpts_a[None], "image_size": torch.tensor([[image_a.shape[-1], image_a.shape[-2]]], device="cuda")},
        {"image": image_b, "keypoints": kpts_b[None], "image_size": torch.tensor([[image_b.shape[-1], image_b.shape[-2]]], device="cuda")},
        sparse=True,
    )
    import flashim

    ours = flashim.load("flexglue", "fp32", device="cuda")
    result = ours.match_keypoints(image_a, kpts_a, image_b, kpts_b)

    ref_pairs = {
        (i, int(j))
        for i, j in enumerate(expected["matches0"][0].tolist())
        if j > -1 and expected["matching_scores0"][0, i] > 0
    }
    pairs = set(zip(result.indices_A.tolist(), result.indices_B.tolist()))
    shared = len(ref_pairs & pairs) / max(len(ref_pairs), 1)
    iou = len(ref_pairs & pairs) / max(len(ref_pairs | pairs), 1)
    # the reference scores of the shared matches (aligned by the a-index)
    ref_scores = expected["matching_scores0"][0]
    our_scores = dict(zip(result.indices_A.tolist(), result.confidence.tolist()))
    common = sorted(ref_pairs & pairs)
    confidence_agreement = max(
        (abs(float(ref_scores[i]) - our_scores[i]) for i, _ in common), default=0.0
    )
    print(f"\nfp32 vs upstream: {len(ref_pairs & pairs)}/{len(ref_pairs)} shared (IoU {iou:.3f}), "
          f"confidence max diff {confidence_agreement:.4f}")
    assert shared > 0.9, f"only {shared:.3f} of the upstream matches are reproduced"
    # IoU too: a few FlashIM-only matches above upstream's gates must not slip through unchecked
    assert iou > 0.9, f"IoU vs upstream is only {iou:.3f} (too many FlashIM-only matches)"


@torch.inference_mode()
def test_dense_gates_match_upstream(checkpoint):
    """match() without keypoints: FlashIM's gated cells (scores > 0 among all samples) are upstream's dense_valid."""
    upstream = _load_upstream(checkpoint)
    import flashim

    image_a, image_b = (_load_image_tensor(path) for path in IMAGES)
    data = [{"image": im, "image_size": torch.tensor([[im.shape[-1], im.shape[-2]]], device="cuda")}
            for im in (image_a, image_b)]
    expected = upstream.match(*data)
    n_expected = int((expected["matches0"][0] > -1).sum())
    ours = flashim.load("flexglue", "fp32", device="cuda")
    result = ours.match(image_a, image_b, generator=torch.Generator(device="cuda").manual_seed(0))
    print(f"\ndense: FlashIM {len(result)} vs upstream {n_expected} gated samples")
    assert n_expected < 5000  # every valid cell is sampled, so the counts compare the gates
    assert abs(len(result) - n_expected) <= max(2, 0.01 * n_expected)
    stricter = ours.match(image_a, image_b, threshold=0.5, generator=torch.Generator(device="cuda").manual_seed(0))
    assert len(stricter) < len(result) and bool((stricter.confidence >= 0.5).all())


@torch.inference_mode()
@pytest.mark.parametrize("preset", ["bf16", "fp8", "int8"])
def test_reduced_precision_tracks_fp32(checkpoint, rdd_keypoints, preset):
    import flashim
    from flashim.models.flexglue.quant import FP8_CALIBRATION, INT8_CALIBRATION

    image_a, image_b = (_load_image_tensor(path) for path in IMAGES)
    kpts_a, kpts_b = rdd_keypoints
    fp32 = flashim.load("flexglue", "fp32", device="cuda")
    calibration = FP8_CALIBRATION if preset == "fp8" else INT8_CALIBRATION if preset == "int8" else None
    low = flashim.load("flexglue", preset, device="cuda", calibration=calibration)

    expected = fp32.match_keypoints(image_a, kpts_a, image_b, kpts_b)
    result = low.match_keypoints(image_a, kpts_a, image_b, kpts_b)
    ref_pairs = set(zip(expected.indices_A.tolist(), expected.indices_B.tolist()))
    pairs = set(zip(result.indices_A.tolist(), result.indices_B.tolist()))
    shared = len(ref_pairs & pairs) / max(len(ref_pairs), 1)
    print(f"\n{preset} vs fp32: {len(ref_pairs & pairs)}/{len(ref_pairs)} shared")
    assert shared > 0.85, f"{preset}: only {shared:.3f} of the fp32 matches are reproduced"
