"""FlashIM's DeDoDe (detector L v2, descriptors B and G) against Parskatt/DeDoDe (vendored in ``tests/oracles/dedode``)."""

import sys

import pytest
import torch

import flashim
from flashim.models.dedode import weights
from flashim.paths import MEGADEPTH_IMAGES, ORACLES

pytestmark = [pytest.mark.slow, pytest.mark.cuda]

ORACLE = ORACLES / "dedode"
IMAGE = MEGADEPTH_IMAGES / "3383162411_67cc9af85e_o.jpg"


@pytest.fixture(scope="module")
def zoo():
    if not (ORACLE / "DeDoDe").exists() or not IMAGE.exists():
        pytest.skip("needs the MegaDepth image")
    sys.path.insert(0, str(ORACLE))
    try:
        import DeDoDe.model_zoo.dedode_models as zoo
    finally:
        sys.path.remove(str(ORACLE))
    return zoo


def _image(size=(448, 560)):
    from flashim.models.romav2.model import load_image

    image = load_image(str(IMAGE), torch.device("cuda"))
    return torch.nn.functional.interpolate(image, size=size, mode="bilinear", antialias=True)


def _fp32(module):
    for m in module.modules():
        if hasattr(m, "amp"):
            m.amp = False
    return module


@pytest.mark.parametrize("variant", ["B", "G"])
def test_fp32_matches_oracle(zoo, variant, monkeypatch):
    monkeypatch.setattr(torch.backends.cudnn, "allow_tf32", False)
    monkeypatch.setattr(torch.backends.cuda.matmul, "allow_tf32", False)
    image = _image()
    normalized = (image - torch.tensor([0.485, 0.456, 0.406], device="cuda").view(1, 3, 1, 1)) / torch.tensor([0.229, 0.224, 0.225], device="cuda").view(1, 3, 1, 1)
    detector = _fp32(zoo.dedode_detector_L(device="cuda", weights=weights.load("detector-L")))
    if variant == "B":
        descriptor = _fp32(zoo.dedode_descriptor_B(device="cuda", weights=weights.load("descriptor-B")))
    else:
        descriptor = zoo.dedode_descriptor_G(device="cuda", weights=weights.load("descriptor-G"), dinov2_weights=weights.load("dinov2"))
        for m in descriptor.modules():
            if hasattr(m, "amp"):
                m.amp = False
        descriptor.encoder.frozen_dinov2.dinov2_vitl14[0].float().cuda()
        descriptor.encoder.frozen_dinov2.amp_dtype = torch.float32
    detector.eval(), descriptor.eval()  # upstream ``detect`` / ``describe_keypoints`` do this themselves
    model = flashim.load("dedode" if variant == "B" else "dedode-g", "fp32")
    with torch.inference_mode():
        ref_logits = detector.forward({"image": normalized})["keypoint_logits"]
        ref_desc = descriptor.forward({"image": normalized})["description_grid"]
        logits = model.detector(normalized)
        desc = model.descriptor(normalized)
    torch.testing.assert_close(logits, ref_logits, atol=2e-3, rtol=1e-3)
    cos = torch.nn.functional.cosine_similarity(desc, ref_desc, dim=1)
    assert cos.min() > 0.999


def test_sampler_matches_oracle(zoo, monkeypatch):
    monkeypatch.setattr(torch.backends.cudnn, "allow_tf32", False)
    image = _image()
    normalized = (image - torch.tensor([0.485, 0.456, 0.406], device="cuda").view(1, 3, 1, 1)) / torch.tensor([0.229, 0.224, 0.225], device="cuda").view(1, 3, 1, 1)
    detector = _fp32(zoo.dedode_detector_L(device="cuda", weights=weights.load("detector-L"))).eval()
    model = flashim.load("dedode", "fp32")
    with torch.inference_mode():
        ref = detector.detect({"image": normalized}, num_keypoints=2000)
        kpts, _, _ = model.detect(image, 2000)
    h, w = image.shape[-2:]
    ref_px = (ref["keypoints"][0] + 1) / 2 * ref["keypoints"].new_tensor([w, h]) - 0.5  # normalized -> pixel index
    ours, theirs = set(map(tuple, kpts.round().long().tolist())), set(map(tuple, ref_px.round().long().tolist()))
    assert len(ours & theirs) >= 0.98 * len(theirs)
