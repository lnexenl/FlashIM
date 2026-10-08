"""FlashIM's SuperPoint against cvg/LightGlue's ``lightglue.SuperPoint`` (downloaded on demand, see ``_reference.py``: its Magic Leap notice forbids copying it here)."""

import sys

import pytest
import torch

import flashim
from flashim.paths import MEGADEPTH_IMAGES

from ._reference import reference_root

pytestmark = [pytest.mark.slow, pytest.mark.cuda]

IMAGE = MEGADEPTH_IMAGES / "3383162411_67cc9af85e_o.jpg"


@pytest.fixture(scope="module")
def oracle():
    if not IMAGE.exists():
        pytest.skip("needs the MegaDepth image")
    root = str(reference_root())
    sys.path.insert(0, root)
    try:
        from lightglue.superpoint import SuperPoint as Reference
    finally:
        sys.path.remove(root)
        for name in [m for m in sys.modules if m == "lightglue" or m.startswith("lightglue.")]:
            del sys.modules[name]  # the vendored LightGlue of test_cvg_lightglue_parity is the other 'lightglue'
    return Reference(max_num_keypoints=2048).eval().cuda()


def _image(size=640):
    from flashim.models.romav2.model import load_image

    if not IMAGE.exists():
        pytest.skip("needs a MegaDepth image")
    image = load_image(str(IMAGE), torch.device("cuda"))
    h, w = image.shape[-2:]
    scale = size / max(h, w)
    return torch.nn.functional.interpolate(image, size=(int(h * scale) // 8 * 8, int(w * scale) // 8 * 8), mode="bilinear", antialias=True)


def _by_position(kpts, scores, desc):
    return {tuple(k): (sc, d) for k, sc, d in zip(kpts.tolist(), scores.tolist(), desc)}


def test_fp32_matches_oracle(oracle):
    """Same keypoints (a handful sitting exactly on the 5e-4 threshold may flip), same scores and descriptors."""
    image = _image()
    with torch.inference_mode():
        ref = oracle({"image": image})
        kpts, scores, desc = flashim.load("superpoint", "fp32").detect(image, 2048)
    mine = _by_position(kpts, scores, desc)
    theirs = _by_position(ref["keypoints"][0], ref["keypoint_scores"][0], ref["descriptors"][0])
    common = mine.keys() & theirs.keys()
    assert len(common) >= 0.995 * len(theirs)
    for key in common:
        assert abs(mine[key][0] - theirs[key][0]) < 2e-2 * theirs[key][0] + 1e-6
        assert float(mine[key][1] @ theirs[key][1]) > 0.999


def test_bf16_tracks_fp32():
    image = _image()
    with torch.inference_mode():
        a = flashim.load("superpoint", "fp32").detect(image, 1024)
        b = flashim.load("superpoint", "bf16").detect(image, 1024)
    common = torch.cdist(a[0], b[0]).min(1).values
    assert (common < 1.5).float().mean() > 0.6
