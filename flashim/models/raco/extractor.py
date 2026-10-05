"""RaCo keypoints + ALIKED descriptors: the ``raco_aliked`` extractor of cvg/LightGlue's ALIKED pairing."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from flashim.models.aliked.model import ALIKED
from flashim.models.raco.model import RaCo
from flashim.models.romav2.model import ImageLike, load_image
from flashim.types import SparseFeatures
from flashim.utils import device_constant


class RaCoALIKED(nn.Module):
    """RaCo (detector + ranker) chooses the keypoints; ALIKED's encoder + SDDH head (its own score head and DKD are
    not used) describes them. ``extract`` returns :class:`SparseFeatures` whose keypoints follow RaCo's pixel-edge
    convention (pixel index + 0.5) and whose ``scores`` are the ranker scores."""

    def __init__(self, raco: RaCo, aliked: ALIKED) -> None:
        super().__init__()
        self.raco, self.aliked = raco, aliked
        self.concurrent = False  # overlap the independent CNNs on CUDA streams (measured no gain at 1024^2)
        self.preset: str | None = None
        self.engine: str | None = None
        self.quant_report = None
        self.calibration_metadata: dict[str, str] | None = None

    @property
    def device(self) -> torch.device:
        return self.raco.device

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

    @torch.inference_mode()
    def extract(self, image: ImageLike, *, resize: int | None = None, top_k: int | None = None) -> SparseFeatures:
        """``resize``: long side of the working resolution, both sides floored to multiples of 32 (keypoints come
        back in the original image's coordinates). Without it the image must already be 32-aligned."""
        image = load_image(image, self.device)
        if image.shape[0] != 1:
            raise ValueError("RaCoALIKED.extract processes one image at a time (batch size 1)")
        height, width = image.shape[-2:]
        new_h, new_w = height, width
        if resize is not None:
            aspect = width / height
            new_h, new_w = (resize, int(resize * aspect)) if height > width else (int(resize / aspect), resize)
            new_h, new_w = (new_h // 32) * 32, (new_w // 32) * 32
        elif height % 32 or width % 32:
            raise ValueError(f"RaCo needs 32-aligned images, got {height}x{width}; pass resize=")
        if (new_h, new_w) != (height, width):
            image = F.interpolate(image, size=(new_h, new_w), mode="bilinear", align_corners=False, antialias=True)
        keypoints, scores, descriptors = self._run(image, top_k)
        keypoints = keypoints * device_constant([width / new_w, height / new_h], keypoints)
        return SparseFeatures(keypoints, scores, descriptors, (height, width))
