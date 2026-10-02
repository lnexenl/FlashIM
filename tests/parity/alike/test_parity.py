"""FlashIM's ALIKE against Shiaoming/ALIKE (checked out at ``~/work/oracles/ALIKE``)."""

import sys
from pathlib import Path

import pytest
import torch

import flashim

pytestmark = [pytest.mark.slow, pytest.mark.cuda]

ORACLE = Path.home() / "work" / "oracles" / "ALIKE"
IMAGE = Path.home() / "work/RoMaV2/data/megadepth/Undistorted_SfM/0015/images/3383162411_67cc9af85e_o.jpg"
VARIANTS = ["alike-t", "alike-n", "alike-l"]


@pytest.fixture(scope="module")
def alike_module():
    if not (ORACLE / "alike.py").exists() or not IMAGE.exists():
        pytest.skip("needs the ALIKE checkout and a MegaDepth image")
    sys.path.insert(0, str(ORACLE))
    try:
        import alike
    finally:
        sys.path.remove(str(ORACLE))
    return alike


def _image():
    from flashim.models.romav2.model import load_image

    image = load_image(str(IMAGE), torch.device("cuda"))
    return torch.nn.functional.interpolate(image, size=(480, 640), mode="bilinear", antialias=True)


@pytest.mark.parametrize("variant", VARIANTS)
def test_fp32_matches_oracle(alike_module, variant, monkeypatch):
    monkeypatch.setattr(torch.backends.cudnn, "allow_tf32", False)  # BN folding changes TF32 rounding
    image = _image()
    cfg = dict(alike_module.configs[variant])
    ref = alike_module.ALike(**cfg, top_k=1000, device="cuda").eval()
    model = flashim.load("alike", "fp32", overrides=[f"variant={variant}"])
    with torch.inference_mode():
        ref_desc, ref_scores = ref.extract_dense_map(image)
        scores, _ = model.dense(image)
        kpts, _, descriptors = model.detect(image, 1000)
        rk, *_ = ref.dkd(ref_scores, ref_desc, sub_pixel=False)
    torch.testing.assert_close(scores, ref_scores, atol=2e-4, rtol=1e-3)
    # keypoints: same set (upstream maps them back through ``.long()``, which can land one pixel off)
    rk_px = ((rk[0] + 1) / 2 * rk[0].new_tensor([[image.shape[-1] - 1, image.shape[-2] - 1]])).round().long()
    ours = set(map(tuple, kpts.long().tolist()))
    theirs = set(map(tuple, rk_px.tolist()))
    assert len(ours & theirs) >= 0.98 * len(theirs)
    # descriptors: upstream's dense (normalized) map read at our keypoints
    xy = kpts.long()
    expected = ref_desc[0, :, xy[:, 1], xy[:, 0]].t()
    assert (descriptors * expected).sum(-1).min() > 0.999
