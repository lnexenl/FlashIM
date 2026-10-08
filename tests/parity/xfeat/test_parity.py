"""FlashIM's XFeat against verlab/accelerated_features (vendored in ``tests/oracles/xfeat``)."""

import sys

import pytest
import torch

import flashim
from flashim.models.xfeat import weights
from flashim.paths import MEGADEPTH_IMAGES, ORACLES

pytestmark = [pytest.mark.slow, pytest.mark.cuda]

ORACLE = ORACLES / "xfeat"
IMAGE = MEGADEPTH_IMAGES / "3383162411_67cc9af85e_o.jpg"


@pytest.fixture(scope="module")
def oracle():
    if not (ORACLE / "modules" / "xfeat.py").exists() or not IMAGE.exists():
        pytest.skip("needs the MegaDepth image")
    sys.path.insert(0, str(ORACLE))
    try:
        from modules.xfeat import XFeat
    finally:
        sys.path.remove(str(ORACLE))
    return XFeat(weights=weights.load_state_dict(), top_k=2048)


def _image():
    from flashim.models.romav2.model import load_image

    image = load_image(str(IMAGE), torch.device("cuda"))
    return torch.nn.functional.interpolate(image, size=(480, 640), mode="bilinear", antialias=True)


def test_fp32_matches_oracle(oracle, monkeypatch):
    monkeypatch.setattr(torch.backends.cudnn, "allow_tf32", False)
    image = _image()
    with torch.inference_mode():
        ref = oracle.detectAndCompute(image, top_k=2048)[0]
        kpts, scores, desc = flashim.load("xfeat", "fp32").detect(image, 2048)
    ours = {tuple(k): (s, d) for k, s, d in zip(kpts.tolist(), scores.tolist(), desc)}
    theirs = {tuple(k): (s, d) for k, s, d in zip(ref["keypoints"].tolist(), ref["scores"].tolist(), ref["descriptors"])}
    common = ours.keys() & theirs.keys()
    assert len(common) >= 0.98 * len(theirs)
    assert all(abs(ours[k][0] - theirs[k][0]) < 2e-3 for k in common)
    assert min(float(ours[k][1] @ theirs[k][1]) for k in common) > 0.998
