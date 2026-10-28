"""RaCo keypoints + descriptors from another model: ALIKED (:class:`RaCoALIKED`, the pairing cvg's LightGlue weights
were trained on) or any model with ``describe_keypoints`` (:class:`RaCoExtractor`)."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from flashim.models.aliked.model import ALIKED
from flashim.models.base import FeatureMatcher
from flashim.models.raco.model import RaCo
from flashim.models.romav2.model import ImageLike, load_image
from flashim.types import Matches, SparseFeatures
from flashim.utils import device_constant

ALIGN = 32  # RaCo's stride


class _RaCoBase(FeatureMatcher, nn.Module):
    """``extract`` around a subclass' :meth:`_run` (``image -> keypoints [1, K, 2], scores [1, K], descriptors [1, K,
    C]``, keypoints in RaCo's pixel-edge convention: pixel index + 0.5): load, resize to the working resolution, map
    the keypoints back to the original image in the pixel-center convention of every other model."""

    raco: RaCo
    # what a LightGlue pipeline adds back: cvg's RaCo + ALIKED weights were trained on RaCo's pixel-edge keypoints,
    # and the raco-<model>-lightglue pairings were measured with them
    matcher_keypoint_offset = 0.5
    preset: str | None = None
    engine: str | None = None
    quant_report = None
    calibration_metadata: dict[str, str] | None = None

    @property
    def device(self) -> torch.device:
        return self.raco.device

    @property
    def align(self) -> int:
        """Working resolutions are multiples of this (RaCo's 32, or more when the descriptor model needs it)."""
        return ALIGN

    def _run(self, image: Tensor, top_k: int | None) -> tuple[Tensor, Tensor, Tensor]:
        raise NotImplementedError

    @torch.inference_mode()
    def extract(self, image: ImageLike, *, resize: int | None = None, top_k: int | None = None) -> SparseFeatures:
        """``resize``: long side of the working resolution, both sides floored to multiples of :attr:`align`
        (keypoints come back in the original image's coordinates). Without it the image must already be aligned."""
        image = load_image(image, self.device)
        if image.shape[0] != 1:
            raise ValueError(f"{type(self).__name__}.extract processes one image at a time (batch size 1)")
        height, width = image.shape[-2:]
        new_h, new_w = height, width
        align = self.align
        if resize is not None:
            aspect = width / height
            new_h, new_w = (resize, int(resize * aspect)) if height > width else (int(resize / aspect), resize)
            new_h, new_w = max(align, (new_h // align) * align), max(align, (new_w // align) * align)
        elif height % align or width % align:
            raise ValueError(f"{type(self).__name__} needs {align}-aligned images, got {height}x{width}; pass resize=")
        if (new_h, new_w) != (height, width):
            image = F.interpolate(image, size=(new_h, new_w), mode="bilinear", align_corners=False, antialias=True)
        keypoints, scores, descriptors = self._run(image, top_k)
        # pixel-edge coordinates scale with the image as they are; pixel-center ones are 0.5 less
        keypoints = keypoints * device_constant([width / new_w, height / new_h], keypoints) - 0.5
        return SparseFeatures(keypoints, scores, descriptors, (height, width))


class RaCoExtractor(_RaCoBase):
    """RaCo (detector + ranker) chooses the keypoints; ``descriptor`` (any model with ``describe_keypoints(image,
    keypoints)``: SuperPoint, ALIKE, DISK, XFeat, DeDoDe, DaD, RIPE, RDD) describes them on the same working image.
    ``scores`` are the ranker scores, and matching is the descriptor model's own (``match_features``: mutual NN, dual softmax, ...).
    The descriptors are the model's, read at RaCo's sub-pixel positions: the descriptor model never saw these
    keypoints in training, so a matcher trained for its own detector (LightGlue) is applied out of distribution."""

    def __init__(self, raco: RaCo, descriptor: nn.Module) -> None:
        super().__init__()
        if not hasattr(descriptor, "describe_keypoints"):
            raise TypeError(f"{type(descriptor).__name__} has no describe_keypoints(image, keypoints)")
        self.raco, self.descriptor = raco, descriptor

    @property
    def align(self) -> int:
        return math.lcm(ALIGN, getattr(self.descriptor, "div_by", ALIGN))  # DeDoDe-G: 56 -> 224

    def _run(self, image: Tensor, top_k: int | None) -> tuple[Tensor, Tensor, Tensor]:
        raco = self.raco
        x = raco.normalize(image)
        keypoints, scores = raco.rank(raco.ranker_map(x), raco.detect(x, top_k), top_k)
        descriptors = self.descriptor.describe_keypoints(image, keypoints[0] - 0.5)  # descriptors want pixel indices
        return keypoints, scores, descriptors[None]

    @torch.inference_mode()
    def match_features(self, a: SparseFeatures, b: SparseFeatures, threshold: float | None = None) -> Matches:
        return self.descriptor.match_features(a, b, threshold)


class RaCoALIKED(_RaCoBase):
    """RaCo (detector + ranker) chooses the keypoints; ALIKED's encoder + SDDH head (its own score head and DKD are
    not used) describes them. ``extract`` returns :class:`SparseFeatures` whose ``scores`` are the ranker scores."""

    def __init__(self, raco: RaCo, aliked: ALIKED) -> None:
        super().__init__()
        self.raco, self.aliked = raco, aliked
        self.concurrent = False  # overlap the independent CNNs on CUDA streams (measured no gain at 1024^2)

    @torch.inference_mode()
    def match_features(self, a: SparseFeatures, b: SparseFeatures, threshold: float | None = None) -> Matches:
        return self.aliked.match_features(a, b, threshold)

    def _side_streams(self, device: torch.device) -> tuple[torch.cuda.Stream, torch.cuda.Stream]:
        streams = getattr(self, "_streams_", None)
        if streams is None:
            streams = (torch.cuda.Stream(device), torch.cuda.Stream(device))
            self._streams_ = streams
        return streams

    def _run(self, image: Tensor, top_k: int | None) -> tuple[Tensor, Tensor, Tensor]:
        """RaCo's detector, RaCo's ranker and ALIKED's encoder depend only on the image, so on CUDA they overlap:
        the two RaCo CNNs on side streams, ALIKED's encoder on the caller's stream. ALIKED must stay on the
        caller's stream: ``torchvision.ops.deform_conv2d`` (its ``block3``/``block4``) ignores the current stream and
        returns wrong values when its input was produced on a side stream. The keypoint-dependent steps (ranking,
        SDDH) follow once their inputs are ready."""
        raco, aliked = self.raco, self.aliked
        x = raco.normalize(image)
        if not image.is_cuda or not self.concurrent:
            keypoints, scores = raco.rank(raco.ranker_map(x), raco.detect(x, top_k), top_k)
            return keypoints, scores, aliked.describe(image, keypoints)
        main = torch.cuda.current_stream(image.device)
        detect_stream, rank_stream = self._side_streams(image.device)
        detect_stream.wait_stream(main)
        rank_stream.wait_stream(main)
        with torch.cuda.stream(detect_stream):
            candidates = raco.detect(x, top_k)
        with torch.cuda.stream(rank_stream):
            ranker_map = raco.ranker_map(x)
        feature_map = aliked.encode(image)
        main.wait_stream(detect_stream)
        main.wait_stream(rank_stream)
        candidates.record_stream(main)
        ranker_map.record_stream(main)
        keypoints, scores = raco.rank(ranker_map, candidates, top_k)
        return keypoints, scores, aliked.describe_at(feature_map, keypoints)
