"""TraqPoint: track-aware RL sparse feature detector + DINOv3-ConvNeXt deformable-transformer descriptor."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from flashim.models.base import FeatureMatcher
from flashim.models.rdd import keypoints as kp
from flashim.models.rdd.model import BasicLayer, target_size
from flashim.models.rdd.transformer import DeformableTransformer
from flashim.models.romav2.model import ImageLike, load_image
from flashim.models.traqpoint import weights
from flashim.models.traqpoint.backbone import DINOv3ConvNeXt
from flashim.models.traqpoint.config import TraqPointConfig
from flashim.models.traqpoint.detector import Detector
from flashim.platforms import default_device
from flashim.types import Matches, SparseFeatures
from flashim.utils import device_constant, grid_sample

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def sample_descriptors(features: Tensor, keypoints: Tensor, height: int, width: int) -> Tensor:
    """Bicubic samples of ``features`` ``[B, C, h, w]`` at ``keypoints`` ``[B, K, 2]`` (``align_corners=True``),
    L2-normalized: ``[B, K, C]``."""
    scale = device_constant([width - 1, height - 1], keypoints)
    grid = (2.0 * (keypoints / scale) - 1.0)[:, :, None].to(features.dtype)
    sampled = grid_sample(features, grid, mode="bicubic", align_corners=True)
    return F.normalize(sampled.permute(0, 2, 3, 1).squeeze(-2), dim=-1)


class TraqPoint(FeatureMatcher, nn.Module):
    """``detector`` (RL-trained score map) and ``backbone`` -> ``input_proj`` -> ``transformer`` (descriptor map).

    Module paths follow the TraqPoint checkpoint after :func:`weights.convert_state_dict`, so it loads strictly.
    ``extract`` returns keypoints and descriptors; ``match`` runs dual-softmax mutual-NN matching.
    """

    def __init__(self, config: TraqPointConfig | None = None) -> None:
        super().__init__()
        self.config = cfg = config or TraqPointConfig()
        dims = cfg.backbone_dims
        self.backbone = DINOv3ConvNeXt(cfg.backbone_depths, dims)
        self.input_proj = nn.ModuleList(
            [nn.Sequential(nn.Conv2d(c, cfg.d_model, 1), nn.GroupNorm(32, cfg.d_model)) for c in dims]
        )
        self.transformer = DeformableTransformer(
            cfg.d_model, cfg.num_heads, cfg.num_layers, cfg.d_ffn, cfg.num_levels, cfg.num_points, activation="relu"
        )
        hidden = max(cfg.d_model // 2, 64)
        self.matchibility_head = nn.Sequential(
            BasicLayer(cfg.d_model, hidden),
            BasicLayer(hidden, max(hidden // 2, 32)),
            nn.Conv2d(max(hidden // 2, 32), 1, 1),
            nn.Sigmoid(),
        )
        self.detector = Detector(cfg.detector_dims)
        # Filled in by the builder (``flashim.build``).
        self.preset: str | None = None
        self.engine: str | None = None
        self.quant_report = None
        self.calibration_metadata: dict[str, str] | None = None
        self.compute_dtype = torch.float32

    @classmethod
    def from_pretrained(
        cls,
        config: TraqPointConfig | None = None,
        *,
        checkpoint: str | None = None,
        device: torch.device | str | None = None,
        state_dict: dict[str, Tensor] | None = None,
    ) -> TraqPoint:
        device = torch.device(device or default_device())
        with torch.device("meta"):
            model = cls(config)
        state = state_dict if state_dict is not None else weights.load_state_dict(checkpoint)
        model.load_state_dict(state, strict=True, assign=True)
        model.to(device)
        return model.eval()

    @property
    def device(self) -> torch.device:
        return self.detector.block1.conv1.weight.device

    def set_compute_dtype(self, dtype: torch.dtype) -> None:
        """Run the descriptor branch (backbone, projections, transformer) in ``dtype``; the detector stays FP32."""
        for module in (self.backbone, self.input_proj, self.transformer, self.matchibility_head):
            module.to(dtype)
        self.transformer.level_embed.data = self.transformer.level_embed.data.float()
        self.compute_dtype = dtype

    # -- forward ---------------------------------------------------------------------------------

    def describe(self, image: Tensor) -> Tensor:
        """Descriptor map ``[B, d_model, H/4, W/4]`` for an image ``[B, 3, H, W]`` in ``[0, 1]`` (H, W % 32 == 0)."""
        mean, std = (device_constant(v, image).view(1, 3, 1, 1) for v in (IMAGENET_MEAN, IMAGENET_STD))
        x = ((image - mean) / std).to(self.compute_dtype)
        features = self.backbone(x)
        sources = [proj(feat) for proj, feat in zip(self.input_proj, features)]
        levels = self.transformer(sources)
        fused = levels[0]
        for level in levels[1:]:
            fused = fused + F.interpolate(level, size=fused.shape[-2:], mode="bilinear", align_corners=True)
        return fused

    def describe_keypoints(self, image: Tensor, keypoints: Tensor) -> Tensor:
        """L2-normalized bicubic descriptors ``[K, C]`` at ``keypoints`` ``[K, 2]``."""
        descriptor_map = self.describe(image)
        height, width = image.shape[-2:]
        return sample_descriptors(F.normalize(descriptor_map.float(), dim=1), keypoints[None], height, width)[0]

    def matchability(self, descriptor_map: Tensor) -> Tensor:
        return self.matchibility_head(descriptor_map)

    def forward(self, image: Tensor) -> tuple[Tensor, Tensor]:
        """``(descriptor map [B, C, H/4, W/4], keypoint score map [B, 1, H, W])``. On CUDA, ``detector`` runs on a
        side stream concurrent with ``describe``."""
        if not image.is_cuda:
            return self.describe(image), self.detector(image)
        current = torch.cuda.current_stream(image.device)
        side = self._detector_stream(image.device)
        side.wait_stream(current)
        with torch.cuda.stream(side):
            score_map = self.detector(image)
        descriptor_map = self.describe(image)
        current.wait_stream(side)
        return descriptor_map, score_map

    def _detector_stream(self, device: torch.device) -> torch.cuda.Stream:
        stream = getattr(self, "_detector_stream_", None)
        if stream is None:
            stream = torch.cuda.Stream(device)
            self._detector_stream_ = stream
        return stream

    # -- sparse API ------------------------------------------------------------------------------

    @torch.inference_mode()
    def extract(self, image: ImageLike, *, resize: int | None = None, top_k: int | None = None) -> SparseFeatures:
        """Keypoints and descriptors. ``resize``: long side of the working resolution (upstream MegaDepth uses 1600);
        keypoints are returned in the original image's pixel coordinates."""
        cfg = self.config
        image = load_image(image, self.device)
        height, width = image.shape[-2:]
        new_h, new_w = target_size(height, width, resize)
        if (new_h, new_w) != (height, width):
            image = F.interpolate(image, size=(new_h, new_w), mode="bilinear", align_corners=False, antialias=True)
        descriptor_map, score_map = self(image)
        keypoints, scores = kp.detect_keypoints(
            score_map, top_k or cfg.top_k, cfg.nms_radius, cfg.softargmax_temperature
        )
        # Upstream SoftDetect returns [-1, 1] via /(W-1, H-1)*2 - 1, then to_pixel_coords maps [-1, 1] -> [0, W].
        norm = keypoints / device_constant([new_w - 1, new_h - 1], keypoints) * 2.0 - 1.0
        keypoints = torch.stack((new_w * (norm[..., 0] + 1.0) / 2.0, new_h * (norm[..., 1] + 1.0) / 2.0), dim=-1)
        descriptors = sample_descriptors(F.normalize(descriptor_map.float(), dim=1), keypoints, new_h, new_w)
        keypoints = keypoints * device_constant([width / new_w, height / new_h], keypoints)
        if cfg.detection_threshold > 0.0:
            valid = scores[0] > cfg.detection_threshold
            keypoints, scores, descriptors = keypoints[:, valid], scores[:, valid], descriptors[:, valid]
        return SparseFeatures(keypoints, scores, descriptors, (height, width))

    @torch.inference_mode()
    def match_features(
        self, a: SparseFeatures, b: SparseFeatures, threshold: float | None = None
    ) -> Matches:
        cfg = self.config
        ia, ib, conf = kp.dual_softmax_match(
            a.descriptors[0],
            b.descriptors[0],
            cfg.match_inv_temperature,
            cfg.match_threshold if threshold is None else threshold,
        )
        return Matches(a.keypoints[0][ia], b.keypoints[0][ib], conf, ia, ib)
