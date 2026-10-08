"""FlashIM's native RDD against the reference implementation (github.com/xtcpete/rdd vendored in tests/oracles/rdd).

The reference needs ``timm``, ``yacs`` and ``pyyaml`` (not FlashIM dependencies); tests skip without them.
"""

import sys

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from ...conftest import MEGADEPTH_IMAGES, ORACLES

pytestmark = [pytest.mark.cuda, pytest.mark.slow]

RDD_ROOT = ORACLES / "rdd"
IMAGES = [MEGADEPTH_IMAGES / "505127_40b48ea397_o.jpg", MEGADEPTH_IMAGES / "505128_561e05e146_o.jpg"]


@pytest.fixture(scope="module")
def checkpoint():
    from flashim.models.rdd.weights import find_checkpoint

    try:
        return find_checkpoint()
    except FileNotFoundError as exc:
        pytest.skip(str(exc))


@pytest.fixture(scope="module")
def oracle(checkpoint):
    for module in ("timm", "yacs", "yaml"):
        pytest.importorskip(module)
    if not (RDD_ROOT / "src" / "RDD.py").exists():
        pytest.skip(f"needs the RDD reference repository at {RDD_ROOT}")
    sys.path.insert(0, str(RDD_ROOT))
    try:
        from src.RDD import build
    finally:
        sys.path.remove(str(RDD_ROOT))
    model = build(weights=str(checkpoint)).cuda()
    model.eval()
    return model


def _images():
    if not IMAGES:
        pytest.skip("needs MegaDepth images under assets/megadepth_0015")
    from PIL import Image

    out = []
    for path in IMAGES:
        image = Image.open(path).convert("RGB")
        w, h = image.size
        image = image.crop((0, 0, w // 32 * 32, h // 32 * 32)).resize((640, 480))  # no resize inside extract
        out.append(torch.from_numpy(np.asarray(image)).permute(2, 0, 1)[None].float().cuda() / 255)
    return out


@torch.inference_mode()
def test_fp32_matches_reference(oracle):
    from flashim.models.rdd import RDD

    ours = RDD.from_pretrained(device="cuda")
    for image in _images():
        ref_desc, ref_scores, _ = oracle(image)
        desc, scores = ours(image)
        assert torch.equal(scores, ref_scores)
        assert ((desc - ref_desc).norm() / ref_desc.norm()) < 1e-4  # deformable sampling: summation order only
        ref = oracle.extract(image)[0]
        out = ours.extract(image)
        assert (out.keypoints[0] - ref["keypoints"]).abs().max() < 1e-3
        assert (out.descriptors[0] * ref["descriptors"]).sum(-1).min() > 0.9999


@torch.inference_mode()
@pytest.mark.parametrize("preset", ["bf16", "fp8", "fp8-full"])
def test_reduced_precision_tracks_fp32(checkpoint, preset):
    import flashim

    reference = flashim.load("rdd", preset="fp32")
    # the preset as named, emulated on a GPU without FP8 (flashim.load builds its fallback there)
    model = flashim.build_config(flashim.resolve("rdd", preset))
    if preset != "bf16":
        assert model.quant_report.fused_blocks == ["transformer.encoder"]
    for image in _images():
        a, b = reference.describe(image), model.describe(image).float()
        cosine = F.cosine_similarity(a, b, dim=1)
        assert cosine.mean() > 0.99, f"{preset}: mean descriptor cosine {cosine.mean():.4f}"
