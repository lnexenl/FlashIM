"""RaCo keypoints + another model's descriptors: the extractor wiring (CPU) and ``describe_keypoints`` against each
model's own ``detect`` (CUDA, downloads the checkpoints on first use)."""

import pytest
import torch
from torch import nn

import flashim
from flashim.models.raco import RaCo, RaCoConfig
from flashim.models.raco.extractor import RaCoExtractor
from flashim.models.raco.spec import DESCRIBERS
from flashim.paths import ASSETS
from flashim.types import Matches, SparseFeatures


class _Descriptor(nn.Module):
    div_by = 8

    def __init__(self):
        super().__init__()
        self.seen = None

    def describe_keypoints(self, image, keypoints):
        self.seen = keypoints
        return torch.nn.functional.normalize(torch.ones(len(keypoints), 4), dim=-1)

    def match_features(self, a, b, threshold=None):
        return Matches(a.keypoints[0][:1], b.keypoints[0][:1], torch.ones(1), None, None)


def _raco():
    torch.manual_seed(0)
    return RaCo(RaCoConfig(num_keypoints=16, candidate_multiplier=2)).eval()


def test_keypoints_are_pixel_centers_and_lightglue_sees_raco_convention():
    descriptor = _Descriptor()
    extractor = RaCoExtractor(_raco(), descriptor)
    features = extractor.extract(torch.rand(1, 3, 64, 96), top_k=16)
    assert isinstance(features, SparseFeatures)
    assert features.keypoints.shape == (1, 16, 2) and features.descriptors.shape == (1, 16, 4)
    assert torch.equal(descriptor.seen, features.keypoints[0])  # pixel centers, as every model returns them
    assert extractor.match_features(features, features).keypoints_A.shape == (1, 2)  # the descriptor model's matcher

    class Spy(nn.Module):  # a LightGlue stand-in: records its inputs, matches keypoint i to i
        def match_features(self, a, b, threshold=None):
            self.seen = a.keypoints
            index = torch.arange(a.keypoints.shape[1])
            return Matches(a.keypoints[0], b.keypoints[0], torch.ones(len(index)), index, index)

    from flashim.models.lightglue.pipeline import SparsePipeline

    pipeline = SparsePipeline(extractor, Spy())
    matches = pipeline.match_features(features, features)
    assert torch.equal(pipeline.matcher.seen, features.keypoints + 0.5)  # RaCo's pixel-edge convention
    assert torch.equal(matches.keypoints_A, features.keypoints[0])


def test_alignment_follows_the_descriptor_model():
    descriptor = _Descriptor()
    descriptor.div_by = 56  # DeDoDe-G
    extractor = RaCoExtractor(_raco(), descriptor)
    assert extractor.align == 224
    with pytest.raises(ValueError, match="224-aligned"):
        extractor.extract(torch.rand(1, 3, 64, 96))
    assert extractor.extract(torch.rand(1, 3, 100, 130), resize=224, top_k=8).image_size == (100, 130)


def test_needs_a_model_that_can_describe_keypoints():
    with pytest.raises(TypeError, match="describe_keypoints"):
        RaCoExtractor(_raco(), nn.Identity())


@pytest.mark.parametrize("descriptor", DESCRIBERS)
def test_specs_resolve(descriptor):
    config = flashim.resolve(f"raco-{descriptor}")
    assert {name: c.model for name, c in config.components.items()} == {"raco": "raco", "descriptor": descriptor}
    assert config.components["raco"].preset == ("fp32" if config.components["descriptor"].preset == "fp32" else "bf16")


@pytest.mark.parametrize("descriptor", ["superpoint", "disk", "xfeat", "rdd"])
def test_lightglue_pairings_resolve(descriptor):
    config = flashim.resolve(f"raco-{descriptor}-lightglue")
    assert config.components["extractor"].model == f"raco-{descriptor}"
    assert config.components["matcher"].variant == descriptor


def test_classical_descriptors_cannot_describe_given_keypoints():
    """SIFT / ORB descriptors need a scale and an orientation per keypoint, which RaCo does not provide."""
    from flashim.models.sift.model import SIFT

    assert "sift" not in DESCRIBERS and "orb" not in DESCRIBERS
    with pytest.raises(NotImplementedError):
        SIFT.describe_keypoints(None, None, None)


@pytest.mark.cuda
@pytest.mark.parametrize("name", ["superpoint", "alike", "disk", "xfeat", "dedode", "ripe", "rdd"])
def test_describe_keypoints_reproduces_detect(name):
    """Describing the model's own keypoints gives its own descriptors (what makes a foreign keypoint set meaningful)."""
    from flashim.models.romav2.model import load_image

    model = flashim.load(name, "fp32")
    image = load_image(str(ASSETS / "sacre_coeur1.jpg"), model.device)
    image = torch.nn.functional.interpolate(image, size=(448, 448), mode="bilinear", antialias=True)
    with torch.inference_mode():
        if name == "rdd":
            features = model.extract(image, top_k=512)
            keypoints = features.keypoints[0]
            reference = features.descriptors[0]
            described = model.describe_keypoints(image, keypoints)
        else:
            keypoints, _, reference = model.detect(image, 512)
            described = model.describe_keypoints(image, keypoints)
    cosine = (described.float() * reference.float()).sum(-1)
    assert cosine.min() > 0.999, (name, cosine.min().item())
