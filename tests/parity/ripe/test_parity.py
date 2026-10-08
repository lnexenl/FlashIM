"""FlashIM's RIPE against fraunhoferhhi/RIPE (downloaded on demand, see ``_reference.py``: RIPE's license does not allow shipping its code here)."""

import sys

import pytest
import torch

import flashim
from flashim.models.ripe import weights
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
        from ripe.models.backbones.vgg import VGG
        from ripe.models.ripe import RIPE
        from ripe.models.upsampler.hypercolumn_features import HyperColumnFeatures
    finally:
        sys.path.remove(root)
    model = RIPE(net=VGG(pretrained=False), upsampler=HyperColumnFeatures())
    model.load_state_dict(weights.load_state_dict())
    return model.cuda().eval()


def test_fp32_matches_oracle(oracle, monkeypatch):
    from flashim.models.romav2.model import load_image

    monkeypatch.setattr(torch.backends.cudnn, "allow_tf32", False)
    image = load_image(str(IMAGE), torch.device("cuda"))
    image = torch.nn.functional.interpolate(image, size=(512, 664), mode="bilinear", antialias=True)
    model = flashim.load("ripe", "fp32")
    with torch.inference_mode():
        ref_k, ref_d, ref_s = oracle.detectAndCompute(image, threshold=0.5, top_k=1500)
        kpts, scores, desc = model.detect(image, 1500)
    ours = {tuple(k): (s, d) for k, s, d in zip(kpts.tolist(), scores.tolist(), desc)}
    theirs = {tuple(k): (s, d) for k, s, d in zip(ref_k.tolist(), ref_s.tolist(), ref_d)}
    common = ours.keys() & theirs.keys()
    assert len(common) >= 0.98 * len(theirs)
    assert all(abs(ours[k][0] - theirs[k][0]) < 2e-3 for k in common)
    assert min(float(ours[k][1] @ theirs[k][1]) for k in common) > 0.998
