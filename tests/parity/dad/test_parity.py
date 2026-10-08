"""FlashIM's DaD detector against Parskatt/dad (vendored in ``tests/oracles/dad``)."""

import sys

import pytest
import torch

import flashim
from flashim.paths import MEGADEPTH_IMAGES, ORACLES

pytestmark = [pytest.mark.slow, pytest.mark.cuda]

ORACLE = ORACLES / "dad"
IMAGE = MEGADEPTH_IMAGES / "3383162411_67cc9af85e_o.jpg"


@pytest.fixture(scope="module")
def dad():
    if not (ORACLE / "dad").exists() or not IMAGE.exists():
        pytest.skip("needs the MegaDepth image")
    sys.path.insert(0, str(ORACLE))
    try:
        import dad
    finally:
        sys.path.remove(str(ORACLE))
    return dad


def test_fp32_matches_oracle(dad, monkeypatch):
    from flashim.models.romav2.model import load_image

    monkeypatch.setattr(torch.backends.cudnn, "allow_tf32", False)
    image = load_image(str(IMAGE), torch.device("cuda"))
    image = torch.nn.functional.interpolate(image, size=(512, 664), mode="bilinear", antialias=True)
    normalized = (image - torch.tensor([0.485, 0.456, 0.406], device="cuda").view(1, 3, 1, 1)) / torch.tensor([0.229, 0.224, 0.225], device="cuda").view(1, 3, 1, 1)
    detector = dad.load_DaD().eval()
    for m in detector.modules():
        if hasattr(m, "amp"):
            m.amp = False
    model = flashim.load("dad", "fp32")
    with torch.inference_mode():
        ref = detector.detect({"image": normalized}, num_keypoints=1500)
        kpts, _, _ = model.detect(image, 1500)
    h, w = image.shape[-2:]
    ref_px = (ref["keypoints"][0] + 1) / 2 * ref["keypoints"].new_tensor([w, h]) - 0.5
    d = torch.cdist(kpts, ref_px, compute_mode="donot_use_mm_for_euclid_dist")  # the mm path loses ~0.1 px at |x| ~ 300
    assert (d.min(1).values < 0.01).float().mean() > 0.97  # same keypoints incl. sub-pixel offsets
