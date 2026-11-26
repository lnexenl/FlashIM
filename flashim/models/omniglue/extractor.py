"""OmniGlue feature extractor: SuperPoint ``sp_v6`` keypoints/descriptors + DINOv2 ViT-B/14 guidance features.

Reuses :class:`~flashim.models.superpoint.model.SuperPoint` (``variant="sp_v6"``) and
:class:`~flashim.models.dedode.blocks.DinoV2` (built from :class:`~flashim.layers.TransformerBlock`),
so FlashIM's ViT quantization and fusion passes apply by layer type.
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn.functional as F
from torch import Tensor

from flashim.models import sparse
from flashim.models.dedode.blocks import DinoV2
from flashim.models.omniglue import weights
from flashim.models.omniglue.config import OmniGlueConfig
from flashim.models.omniglue.model import OmniGlueFeatures
from flashim.models.romav2.model import ImageLike, load_image
from flashim.models.superpoint.model import SuperPoint
from flashim.platforms import default_device
from flashim.utils import device_constant

MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)


def superpoint_size(height: int, width: int, long_side: int | None = None) -> tuple[int, int]:
    """Working resolution for SuperPoint ``sp_v6``: optional long-side scale, then nearest multiple of 8
    (matching upstream ``SuperPointExtract._resize_input_image``: ``mod < 4`` rounds down, ``>= 4`` rounds up)."""
    if long_side is not None:
        scale = long_side / max(height, width)
        height, width = round(height * scale), round(width * scale)

    def round8(v: int) -> int:
        r = v % 8
        return max(8, v - r if r < 4 else v + (8 - r))

    return round8(height), round8(width)


def dino_size(height: int, width: int, max_size: int = 630, patch: int = 14) -> tuple[int, int]:
    """Working resolution for DINOv2 ViT-B/14 (matching upstream ``DINOExtract._resize_input_image``)."""
    if max(height, width) > max_size:
        if height > width:
            h_t = max_size
            w_t = int(max_size * width / height)
        else:
            w_t = max_size
            h_t = int(max_size * height / width)
    else:
        h_t, w_t = height, width
    return max(patch, (h_t // patch) * patch), max(patch, (w_t // patch) * patch)


def sample_dino_descriptors(feature_map: Tensor, keypoints: Tensor, orig_height: int, orig_width: int) -> Tensor:
    """Bilinearly sample ``feature_map`` ``[1, C, Hf, Wf]`` at original-image ``keypoints`` ``[K, 2]`` ``(x, y)``.

    Reproduces upstream ``dino_extract.get_dino_descriptors`` + ``utils.lookup_descriptor_bilinear`` exactly
    (including zero-decay past ``[Wf - 1, Wf)`` and ``[Hf - 1, Hf)``).
    """
    if keypoints.shape[0] == 0:
        return feature_map.new_zeros((0, feature_map.shape[1]), dtype=torch.float32)
    _, _, hf, wf = feature_map.shape
    img_size = device_constant([orig_width, orig_height], keypoints, torch.float32)
    feat_size = device_constant([wf, hf], keypoints, torch.float32)
    kp_feat = keypoints.float() / img_size * feat_size
    denom = device_constant([max(wf - 1, 1), max(hf - 1, 1)], keypoints, torch.float32)
    grid = (2.0 * kp_feat / denom - 1.0).view(1, 1, -1, 2)
    return F.grid_sample(
        feature_map.float(), grid, mode="bilinear", padding_mode="zeros", align_corners=True
    )[0, :, 0].t()


class OmniGlueExtractor(sparse.SparseExtractor):
    """SuperPoint ``sp_v6`` + DINOv2 ViT-B/14 feature extractor returning :class:`OmniGlueFeatures`."""

    default_top_k = 1024  # upstream SuperPointExtract default num_features
    div_by = 8

    def __init__(self, config: OmniGlueConfig | None = None, *, sp_variant: str = "sp_v6") -> None:
        super().__init__()
        self.config = cfg = config or OmniGlueConfig()
        self.superpoint = SuperPoint(variant=sp_variant)
        self.dinov2 = DinoV2(
            dim=cfg.dino_dim, depth=cfg.dino_depth, heads=cfg.dino_heads, patch=cfg.dino_patch_size, img_size=518
        )
        self.register_buffer("mean", torch.tensor(MEAN).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("std", torch.tensor(STD).view(1, 3, 1, 1), persistent=False)

    @classmethod
    def from_pretrained(
        cls,
        config: OmniGlueConfig | None = None,
        *,
        sp_variant: str = "sp_v6",
        sp_checkpoint: str | Path | None = None,
        dinov2_checkpoint: str | Path | None = None,
        device: torch.device | str | None = None,
    ) -> OmniGlueExtractor:
        from flashim.models.superpoint import weights as sp_weights

        with torch.device("meta"):
            model = cls(config, sp_variant=sp_variant)
        state = {f"superpoint.{k}": v for k, v in sp_weights.load_state_dict(sp_checkpoint, variant=sp_variant).items()}
        state |= {f"dinov2.{k}": v for k, v in weights.load_dinov2_state_dict(dinov2_checkpoint).items()}
        model.load_state_dict(state, strict=True, assign=True)
        model.mean = torch.tensor(MEAN).view(1, 3, 1, 1)
        model.std = torch.tensor(STD).view(1, 3, 1, 1)
        return model.to(torch.device(device or default_device())).eval()

    def batchnorm_pairs(self) -> list[tuple[str, str]]:
        return [(f"superpoint.{c}", f"superpoint.{b}") for c, b in self.superpoint.batchnorm_pairs()]

    def set_compute_dtype(self, dtype: torch.dtype) -> None:
        self.superpoint.set_compute_dtype(dtype)
        self.dinov2.to(dtype)
        self.compute_dtype = dtype

    def normalize_dino(self, image: Tensor) -> Tensor:
        return ((image.float() - self.mean.to(image.device)) / self.std.to(image.device)).to(self.compute_dtype)

    def extract_dino_map(self, image: Tensor) -> Tensor:
        """``[1, 3, H, W]`` RGB in ``[0, 1]`` -> DINOv2 patch feature map ``[1, 768, Hf, Wf]``."""
        h, w = image.shape[-2:]
        dh, dw = dino_size(h, w, self.config.dino_max_size, self.config.dino_patch_size)
        if (dh, dw) != (h, w):
            image = F.interpolate(image, size=(dh, dw), mode="bilinear", align_corners=False)
        return self.dinov2(self.normalize_dino(image))

    def detect(self, image: Tensor, top_k: int | None) -> tuple[Tensor, Tensor, Tensor]:
        return self.superpoint.detect(image, self.default_top_k if top_k is None else top_k)

    @torch.inference_mode()
    def extract(self, image: ImageLike, *, resize: int | None = None, top_k: int | None = None) -> OmniGlueFeatures:
        img = load_image(image, self.device)
        if img.shape[0] != 1:
            raise ValueError(f"{type(self).__name__}.extract processes one image at a time (batch size 1)")
        height, width = img.shape[-2:]
        sp_h, sp_w = superpoint_size(height, width, resize)
        sp_img = img if (sp_h, sp_w) == (height, width) else F.interpolate(
            img, size=(sp_h, sp_w), mode="bilinear", align_corners=False
        )
        keypoints, scores, descriptors = self.superpoint.detect(
            sp_img, self.default_top_k if top_k is None else top_k
        )
        # Upstream SuperPointExtract keypoint rescaling: (new - 1) / (orig - 1)
        if (sp_h, sp_w) != (height, width):
            scale_x = (sp_w - 1.0) / max(width - 1.0, 1.0)
            scale_y = (sp_h - 1.0) / max(height - 1.0, 1.0)
            keypoints = keypoints / device_constant([scale_x, scale_y], keypoints, torch.float32)

        dino_map = self.extract_dino_map(img)
        dino_desc = sample_dino_descriptors(dino_map, keypoints, height, width)
        return OmniGlueFeatures(
            keypoints[None], scores.float()[None], descriptors[None], (height, width), dino_desc[None]
        )
