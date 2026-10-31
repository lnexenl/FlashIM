"""LoMa's extractor on CPU with random weights: resizing as upstream's PIL path, the detector -> descriptor coordinate
mapping, and the spec's variants / presets."""

import numpy as np
import pytest
import torch
import torch.nn.functional as F
from PIL import Image

import flashim
from flashim.build import get_spec
from flashim.models.dedode import blocks
from flashim.models.dedode.model import sample_keypoints
from flashim.models.loma.extractor import (
    DESCRIPTOR_B128,
    LOMA_SAMPLER,
    LoMaExtractor,
    detector_size,
    pil_resize,
)
from flashim.paths import ASSETS


def test_pil_resize_matches_pillow():
    pil = Image.open(ASSETS / "toronto_A.jpg").convert("RGB").resize((400, 300))
    image = torch.from_numpy(np.array(pil)).permute(2, 0, 1)[None].float() / 255
    for w, h in ((200, 144), (112, 112), (520, 384)):  # down in both, aspect changed, up
        ref = torch.from_numpy(np.array(pil.resize((w, h)))).permute(2, 0, 1)[None].float() / 255
        diff = (pil_resize(image, (h, w)) - ref).abs() * 255
        assert diff.max() <= 1.0001 and (diff < 1e-3).float().mean() > 0.995, (w, h)


def test_detector_size_is_upstreams():
    assert detector_size(1185, 1600, 1024) == (752, 1024)  # upstream's 0015_A.jpg
    assert detector_size(1200, 1600, 1024) == (768, 1024)
    assert detector_size(1600, 1063, 1024) == (1024, 680)


@pytest.fixture(scope="module")
def model():
    torch.manual_seed(0)
    model = LoMaExtractor(blocks.DETECTOR_S, DESCRIPTOR_B128, LOMA_SAMPLER).eval()
    model.descriptor_size = (64, 48)
    return model


def test_extract_samples_descriptors_at_upstreams_coordinates(model):
    torch.manual_seed(1)
    image = torch.rand(1, 3, 90, 130)
    features = model.extract(image, resize=96, top_k=50)
    kpts, desc = features.keypoints[0], features.descriptors[0]
    assert kpts.shape == (50, 2) and desc.shape == (50, 128) and features.image_size == (90, 130)
    assert ((kpts > -0.5) & (kpts < torch.tensor([129.5, 89.5]))).all()
    assert not torch.allclose(desc.norm(dim=-1), torch.ones(50))  # not L2-normalized
    # upstream: the dense descriptor map of the (unnormalized) descriptor image, grid_sample'd (align_corners=False)
    # at the keypoints normalized to the full image
    with torch.inference_mode():
        dense = model.descriptor(pil_resize(image, model.descriptor_size))
    normalized = (kpts + 0.5) * 2 / torch.tensor([130.0, 90.0]) - 1
    ref = F.grid_sample(dense, normalized[None, None], mode="bilinear", align_corners=False)[0, :, 0].t()
    torch.testing.assert_close(desc, ref, atol=1e-4, rtol=1e-4)


def test_detect_keypoints_follow_upstream_sampler(model):
    """Borders are kept (upstream's ``remove_borders=False``): with a peak in the corner pixel it is the first keypoint."""
    image = torch.rand(1, 3, 64, 64)
    with torch.no_grad():
        logits = model.detector(model.normalize(image))[0]
    logits[0, 0, 0] = logits.max() + 50
    kpts, _ = sample_keypoints(logits, 10, LOMA_SAMPLER)
    assert kpts[0].round().tolist() == [0.0, 0.0]


def test_spec_variants_and_presets():
    for variant in ("b128", "b", "l", "g", "r"):
        assert flashim.resolve("loma-extractor", "fp32", config={"variant": variant}).variant == variant
    spec = get_spec("loma-extractor")
    for variant, table in (("b128", "b128"), ("b", "b"), ("l", "b"), ("g", "b"), ("r", "r")):  # B / L / G: one descriptor
        for quant in ("int8", "fp8"):
            path = spec.calibrations[f"{variant}/{quant}"]
            assert path.name == f"loma-extractor_{table}_{quant}.safetensors" and path.is_file()
    assert flashim.resolve("loma-extractor", "fp8").precision.quant == "fp8"
    assert flashim.resolve("loma-extractor").preset == "bf16"
    with pytest.raises(ValueError, match="variant"):
        flashim.resolve("loma-extractor", "fp32", config={"variant": "x"})
