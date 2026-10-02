"""FlashIM's DISK against kornia's ``DISK.from_pretrained("depth")`` (same checkpoint, loaded from the local copy)."""

from pathlib import Path

import pytest
import torch

import flashim
from flashim.models.disk import weights

pytestmark = [pytest.mark.slow, pytest.mark.cuda]

IMAGE = Path.home() / "work/RoMaV2/data/megadepth/Undistorted_SfM/0015/images/3383162411_67cc9af85e_o.jpg"


@pytest.fixture(scope="module")
def oracle():
    kornia = pytest.importorskip("kornia")
    try:
        state = weights.load_state_dict()
    except FileNotFoundError:
        pytest.skip("needs the DISK checkpoint")
    model = kornia.feature.DISK().cuda().eval()
    model.load_state_dict(state)
    return model


def _image():
    from flashim.models.romav2.model import load_image

    image = load_image(str(IMAGE), torch.device("cuda"))
    return torch.nn.functional.interpolate(image, size=(480, 640), mode="bilinear", antialias=True)


def test_fp32_matches_oracle(oracle, monkeypatch):
    monkeypatch.setattr(torch.backends.cudnn, "allow_tf32", False)
    image = _image()
    with torch.inference_mode():
        ref = oracle(image, n=2048, window_size=5, score_threshold=0.0)[0]
        kpts, scores, desc = flashim.load("disk", "fp32").detect(image, 2048)
    ours = {tuple(k): (s, d) for k, s, d in zip(kpts.tolist(), scores.tolist(), desc)}
    theirs = {tuple(k): (s, d) for k, s, d in zip(ref.keypoints.tolist(), ref.detection_scores.tolist(), ref.descriptors)}
    common = ours.keys() & theirs.keys()
    assert len(common) >= 0.99 * len(theirs)
    assert all(abs(ours[k][0] - theirs[k][0]) < 1e-3 for k in common)
    assert min(float(ours[k][1] @ theirs[k][1]) for k in common) > 0.999
