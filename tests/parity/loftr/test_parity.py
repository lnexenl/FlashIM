"""FlashIM's LoFTR against kornia's ``LoFTR`` (same checkpoint, loaded from the local copy)."""

from pathlib import Path

import pytest
import torch

import flashim
from flashim.models.loftr import weights

pytestmark = [pytest.mark.slow, pytest.mark.cuda]

ROOT = Path.home() / "work/RoMaV2/data/megadepth/Undistorted_SfM/0015/images"
IMAGES = [ROOT / "3383162411_67cc9af85e_o.jpg", ROOT / "299056923_ecd980298c_o.jpg"]


@pytest.fixture(scope="module")
def oracle():
    kornia = pytest.importorskip("kornia")
    from kornia.feature.loftr.loftr import default_cfg

    try:
        state = weights.load_state_dict("outdoor")
    except FileNotFoundError:
        pytest.skip("needs the LoFTR checkpoint")
    model = kornia.feature.LoFTR(pretrained=None, config=default_cfg)
    model.load_state_dict(state)
    return model.cuda().eval()


def test_fp32_matches_oracle(oracle, monkeypatch):
    from flashim.models import sparse
    from flashim.models.romav2.model import load_image

    monkeypatch.setattr(torch.backends.cudnn, "allow_tf32", False)
    monkeypatch.setattr(torch.backends.cuda.matmul, "allow_tf32", False)
    grays = []
    for path in IMAGES:
        image = load_image(str(path), torch.device("cuda"))
        image = torch.nn.functional.interpolate(image, size=(480, 640), mode="bilinear", antialias=True)
        grays.append(sparse.rgb_to_gray(image))
    model = flashim.load("loftr", "fp32")
    with torch.inference_mode():
        ref = oracle({"image0": grays[0], "image1": grays[1]})
        k0, k1, conf = model.match_prepared(*grays)
    assert abs(len(conf) - len(ref["confidence"])) <= 0.02 * len(ref["confidence"]) + 2
    ours = {tuple(a): (tuple(b), c) for a, b, c in zip(k0.tolist(), k1.tolist(), conf.tolist())}
    common = 0
    for a, b, c in zip(ref["keypoints0"].tolist(), ref["keypoints1"].tolist(), ref["confidence"].tolist()):
        if tuple(a) in ours:
            common += 1
            assert max(abs(ours[tuple(a)][0][i] - b[i]) for i in (0, 1)) < 0.05
            assert abs(ours[tuple(a)][1] - c) < 5e-3
    assert common >= 0.97 * len(ref["confidence"])
