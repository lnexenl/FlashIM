"""ALIKED: sparse detector + descriptor with deformable convolutions (Zhao et al., 2023), native FlashIM port."""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torchvision.models import resnet

from flashim import ops
from flashim.models.aliked import keypoints as kp
from flashim.models.aliked import weights
from flashim.models.aliked.blocks import ConvBlock, ResBlock
from flashim.models.aliked.config import VARIANTS, ALIKEDConfig
from flashim.models.aliked.descriptor import SDDH
from flashim.models.base import FeatureMatcher
from flashim.models.romav2.model import ImageLike, load_image
from flashim.platforms import default_device
from flashim.types import Matches, SparseFeatures
from flashim.utils import device_constant


def _pad_to(image: Tensor, div_by: int) -> tuple[Tensor, tuple[int, int, int, int]]:
    """Replicate-pads ``image`` ``[B, 3, H, W]`` so both dims are multiples of ``div_by``; returns the pad box
    ``(left, right, top, bottom)`` used to crop the corresponding output back."""
    h, w = image.shape[-2:]
    pad_h, pad_w = (-h) % div_by, (-w) % div_by
    box = (pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2)
    return F.pad(image, box, mode="replicate"), box


def _unpad(x: Tensor, box: tuple[int, int, int, int]) -> Tensor:
    left, right, top, bottom = box
    h, w = x.shape[-2:]
    return x[..., top : h - bottom, left : w - right]


def target_size(height: int, width: int, long_side: int | None) -> tuple[int, int]:
    """Working resolution: the long side scaled to ``long_side`` (if given), both sides floored to a
    multiple of 32 (matches ``flashim.models.rdd.model.target_size``)."""
    if long_side is not None:
        aspect = width / height
        height, width = (long_side, int(long_side * aspect)) if height > width else (int(long_side / aspect), long_side)
    return (height // 32) * 32, (width // 32) * 32


class ALIKED(FeatureMatcher, nn.Module):
    """``block1..4`` (conv / deformable-conv encoder, strides 1/2/8/32) -> multi-scale fusion -> ``score_head``
    (keypoint score map) and ``desc_head`` (``SDDH``, sampled at the detected keypoints).

    Module paths follow the released checkpoints (``aliked-t16`` / ``aliked-n16`` / ``aliked-n16rot`` /
    ``aliked-n32``), so it loads strictly with no key renaming. ``extract`` returns keypoints and descriptors;
    ``match`` runs mutual-nearest-neighbour cosine matching between two images.
    """

    def __init__(self, config: ALIKEDConfig | None = None) -> None:
        super().__init__()
        self.config = cfg = config or VARIANTS["aliked-n16"]
        gate, norm = nn.SELU(inplace=True), nn.BatchNorm2d
        self.gate = gate
        self.block1 = ConvBlock(3, cfg.c1, gate, norm)
        self.block2 = ResBlock(cfg.c1, cfg.c2, nn.Conv2d(cfg.c1, cfg.c2, 1), gate, norm, conv_type="conv")
        self.block3 = ResBlock(cfg.c2, cfg.c3, nn.Conv2d(cfg.c2, cfg.c3, 1), gate, norm, conv_type="dcn")
        self.block4 = ResBlock(cfg.c3, cfg.c4, nn.Conv2d(cfg.c3, cfg.c4, 1), gate, norm, conv_type="dcn")
        quarter = cfg.dim // 4
        self.conv1 = resnet.conv1x1(cfg.c1, quarter)
        self.conv2 = resnet.conv1x1(cfg.c2, quarter)
        self.conv3 = resnet.conv1x1(cfg.c3, quarter)
        self.conv4 = resnet.conv1x1(cfg.dim, quarter)
        self.score_head = nn.Sequential(
            resnet.conv1x1(cfg.dim, 8),
            gate,
            resnet.conv3x3(8, 4),
            gate,
            resnet.conv3x3(4, 4),
            gate,
            resnet.conv3x3(4, 1),
        )
        self.desc_head = SDDH(cfg.dim, cfg.K, cfg.M, gate)
        # Filled in by the builder (``flashim.build``).
        self.preset: str | None = None
        self.engine: str | None = None
        self.compute_dtype = torch.float32
        self.channels_last = False
        self.fused_batchnorm: list[str] = []
        self.quantized_deform_convs: list[str] = []

    @classmethod
    def from_pretrained(
        cls,
        config: ALIKEDConfig | None = None,
        *,
        variant: str = "aliked-n16",
        checkpoint: str | Path | None = None,
        device: torch.device | str | None = None,
        state_dict: dict[str, Tensor] | None = None,
    ) -> ALIKED:
        device = torch.device(device or default_device())
        config = config or VARIANTS[variant]
        with torch.device("meta"):
            model = cls(config)
        state = state_dict if state_dict is not None else weights.load_state_dict(variant, checkpoint)
        model.load_state_dict(state, strict=True, assign=True)
        model.to(device)
        return model.eval()

    @property
    def device(self) -> torch.device:
        return self.block1.conv1.weight.device

    def set_compute_dtype(self, dtype: torch.dtype) -> None:
        """Run the encoder and descriptor head in ``dtype``; the score head stays FP32 (keypoint localization)."""
        for module in (self.block1, self.block2, self.block3, self.block4, self.conv1, self.conv2, self.conv3, self.conv4, self.desc_head):
            module.to(dtype)
        self.compute_dtype = dtype

    def set_channels_last(self, enabled: bool = True) -> None:
        """Encoder conv weights to channels-last, so cuDNN can pick its channels-last algorithm for the plain
        convs of ``block1``/``block2`` (``deform_conv2d`` in ``block3``/``block4`` accepts either layout and
        always returns contiguous NCHW, verified bit-identical either way). Measured *no gain* on an RTX 5070
        Ti at ALIKED's channel counts (<=128) — kept available, opt-in, for hardware/variants where it may help."""
        memory_format = torch.channels_last if enabled else torch.contiguous_format
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                module.weight.data = module.weight.data.to(memory_format=memory_format)
        self.channels_last = enabled

    # -- forward -----------------------------------------------------------------------------------------

    def _encode(self, image: Tensor, with_head: bool = True) -> tuple[Tensor | None, Tensor, tuple[int, int, int, int]]:
        """Encoder + multi-scale fusion: ``(head [B, 8, H', W'], feature_map [B, dim, H', W'], pad box)`` at the padded
        resolution (a multiple of ``config.div_by``). ``head`` is the score head's first layer (1x1 conv + SELU, FP32),
        computed inside the fusion kernel; ``with_head=False`` skips it (descriptor-only use)."""
        head_weight = self.score_head[0].weight.reshape(self.score_head[0].out_channels, -1) if with_head else None
        padded, box = _pad_to(image, self.config.div_by)
        x = padded.to(self.compute_dtype)
        if self.channels_last:
            x = x.to(memory_format=torch.channels_last)
        x1 = self.block1(x)
        x2 = self.block2(F.avg_pool2d(x1, 2))
        x3 = self.block3(F.avg_pool2d(x2, 4))
        x4 = self.block4(F.avg_pool2d(x3, 4))
        f1, f2, f3, f4 = (
            self.gate(conv(feat)) for conv, feat in zip((self.conv1, self.conv2, self.conv3, self.conv4), (x1, x2, x3, x4))
        )
        head, feature_map = ops.multiscale_fuse(f1, f2, f3, f4, head_weight, normalize=True)
        return head, feature_map, box

    def dense(self, image: Tensor) -> tuple[Tensor, Tensor]:
        """Image ``[B, 3, H, W]`` in ``[0, 1]`` -> L2-normalized feature map ``[B, dim, H, W]`` and score map
        ``[B, 1, H, W]`` in ``(0, 1)``, both at the input's own resolution (internally padded to a multiple of
        ``config.div_by``)."""
        head, feature_map, box = self._encode(image)
        score_map = torch.sigmoid(self.score_head[2:](head))
        return _unpad(feature_map, box), _unpad(score_map, box)

    def encode(self, image: Tensor) -> Tensor:
        """L2-normalized feature map ``[1, dim, H, W]`` at the input's resolution (the score head is skipped)."""
        _, feature_map, box = self._encode(image, with_head=False)
        return _unpad(feature_map, box)

    def describe_at(self, feature_map: Tensor, keypoints: Tensor) -> Tensor:
        """Descriptors ``[1, N, dim]`` sampled from ``feature_map`` (:meth:`encode`) at ``keypoints`` ``[1, N, 2]``,
        ``(x, y)`` in the map's pixel coordinates."""
        h, w = feature_map.shape[-2:]
        normalized = 2 * keypoints[0] / device_constant([w - 1, h - 1], keypoints) - 1
        return self.desc_head(feature_map[0], normalized)[None]

    def describe(self, image: Tensor, keypoints: Tensor) -> Tensor:
        """Descriptors ``[1, N, dim]`` (L2-normalized) at externally chosen ``keypoints`` ``[1, N, 2]`` — ``(x, y)``
        in the image's pixel coordinates (e.g. from a different detector such as RaCo). The score head is skipped."""
        return self.describe_at(self.encode(image), keypoints)

    def forward(self, image: Tensor) -> tuple[Tensor, Tensor]:
        return self.dense(image)

    # -- sparse API --------------------------------------------------------------------------------------

    @torch.inference_mode()
    def extract(self, image: ImageLike, *, resize: int | None = None, top_k: int | None = None) -> SparseFeatures:
        """Keypoints and descriptors of a single image (batch size must be 1). ``resize``: long side of the
        working resolution; keypoints are returned in the original image's pixel coordinates."""
        cfg = self.config
        image = load_image(image, self.device)
        if image.shape[0] != 1:
            raise ValueError("ALIKED.extract processes one image at a time (batch size 1)")
        height, width = image.shape[-2:]
        new_h, new_w = target_size(height, width, resize) if resize is not None else (height, width)
        if (new_h, new_w) != (height, width):
            image = F.interpolate(image, size=(new_h, new_w), mode="bilinear", align_corners=False, antialias=True)
        feature_map, score_map = self(image)
        keypoints, scores, _ = kp.detect_keypoints(
            score_map, cfg.dkd_radius, cfg.dkd_temperature, cfg.top_k if top_k is None else top_k, cfg.scores_th, cfg.n_limit
        )
        descriptors = self.desc_head(feature_map[0], keypoints[0])
        pixel_kpts = (keypoints[0] + 1) / 2 * device_constant([new_w - 1, new_h - 1], keypoints[0])
        pixel_kpts = pixel_kpts * device_constant([width / new_w, height / new_h], pixel_kpts)
        return SparseFeatures(pixel_kpts[None], scores[0][None], descriptors[None], (height, width))

    @torch.inference_mode()
    def match_features(self, a: SparseFeatures, b: SparseFeatures, threshold: float | None = None) -> Matches:
        ia, ib, conf = kp.mutual_nn_match(a.descriptors[0], b.descriptors[0], self.config.match_threshold if threshold is None else threshold)
        return Matches(a.keypoints[0][ia], b.keypoints[0][ib], conf, ia, ib)
