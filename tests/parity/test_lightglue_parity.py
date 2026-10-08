"""FlashIM's native LightGlue against the reference implementation (RDD's ``src/matchers/lightglue.py``)."""

import importlib.util
from pathlib import Path

import pytest
import torch

from ..conftest import MEGADEPTH_IMAGES

pytestmark = [pytest.mark.cuda, pytest.mark.slow]

REFERENCE = Path.home() / "work" / "RDD" / "src" / "matchers" / "lightglue.py"
IMAGES = [MEGADEPTH_IMAGES / "505127_40b48ea397_o.jpg", MEGADEPTH_IMAGES / "505128_561e05e146_o.jpg"]


@pytest.fixture(scope="module")
def checkpoint():
    from flashim.models.lightglue.weights import find_checkpoint

    try:
        return find_checkpoint()
    except FileNotFoundError as exc:
        pytest.skip(str(exc))


@pytest.fixture(scope="module")
def features():
    if not IMAGES:
        pytest.skip("needs MegaDepth images under assets/megadepth_0015")
    import flashim

    rdd = flashim.load("rdd", preset="bf16")
    return [rdd.extract(str(path), resize=1600) for path in IMAGES]


@torch.inference_mode()
def test_fp32_matches_reference(checkpoint, features):
    if not REFERENCE.exists():
        pytest.skip(f"needs the RDD reference repository at {REFERENCE}")
    spec = importlib.util.spec_from_file_location("rdd_reference_lightglue", REFERENCE)
    reference = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(reference)
    ref = reference.LightGlue(
        "rdd", flash=False, weights=str(checkpoint), depth_confidence=-1, width_confidence=-1
    ).cuda().eval()

    from flashim.models.lightglue import LightGlue

    ours = LightGlue.from_pretrained(checkpoint=str(checkpoint), device="cuda")
    a, b = features
    data = lambda f: {"keypoints": f.keypoints, "descriptors": f.descriptors, "image_size": torch.tensor(f.image_size)[None]}
    expected = ref({"image0": data(a), "image1": data(b)})
    result = ours.match_features(a, b)
    assert {tuple(m) for m in expected["matches"][0].tolist()} == set(zip(result.indices_A.tolist(), result.indices_B.tolist()))
    assert (expected["scores"][0].sort().values - result.confidence.sort().values).abs().max() < 1e-3


@torch.inference_mode()
@pytest.mark.parametrize("preset", ["bf16", "fp8", "int8"])
def test_reduced_precision_tracks_fp32(features, preset):
    import flashim

    fp32 = flashim.load("lightglue", preset="fp32")
    # the preset as named, emulated on a GPU without FP8 (flashim.load builds its fallback there)
    low = flashim.build_config(flashim.resolve("lightglue", preset))
    a, b = features
    expected = fp32.match_features(a, b)
    result = low.match_features(a, b)
    ref_pairs = set(zip(expected.indices_A.tolist(), expected.indices_B.tolist()))
    pairs = set(zip(result.indices_A.tolist(), result.indices_B.tolist()))
    assert len(ref_pairs & pairs) / len(ref_pairs) > 0.9
