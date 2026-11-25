"""SuperPoint (DeTone et al., 2018): VGG-style encoder, 65-way cell-softmax keypoint head, 256-d descriptor head.

Supports two variants:

- ``v1`` (default): Magic Leap's ``superpoint_v1.pth`` (no BatchNorm, ``simple_nms`` radius 4, border removal 4).
- ``sp_v6``: rpautrat/SuperPoint's ``sp_v6`` (Conv -> ReLU -> BatchNorm2d(eps=1e-3) in the trunk and first head convs,
  Conv -> BatchNorm2d(eps=1e-3) in ``convPb`` / ``convDb`` folded at load, TF ``box_nms`` 4x4 IoU 0.1 at threshold
  0.01, and TF1 ``ResizeBilinear(align_corners=False, half_pixel_centers=False)`` descriptor sampling).
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from flashim import ops
from flashim.layers.conv import conv_relu
from flashim.models import sparse
from flashim.models.superpoint import weights
from flashim.platforms import default_device
from flashim.quant.methods.conv2d import StaticInt8Conv2d, StaticMXFP8Conv2d
from flashim.quant.tensor import Int8Tensor
from flashim.utils import device_constant


def _sample_descriptors_sp_v6(keypoints: Tensor, coarse_descriptors: Tensor, stride: int = 8) -> Tensor:
    """Bilinear descriptor lookup matching TF1 ``ResizeBilinear(align_corners=False, half_pixel_centers=False)`` of a
    ``[1, C, h, w]`` coarse map to ``(h * stride, w * stride)`` followed by L2 normalization (``eps=1e-12``)."""
    if keypoints.shape[0] == 0:
        return coarse_descriptors.new_zeros((0, coarse_descriptors.shape[1]), dtype=torch.float32)
    _, _, h, w = coarse_descriptors.shape
    coords = keypoints.float() / float(stride)
    grid = (2.0 * coords / device_constant([w - 1, h - 1], keypoints, torch.float32) - 1.0).view(1, 1, -1, 2)
    out = F.grid_sample(
        coarse_descriptors.float(), grid, mode="bilinear", padding_mode="border", align_corners=True
    )[0, :, 0].t()
    return out * torch.rsqrt((out * out).sum(dim=-1, keepdim=True).clamp(min=1e-12))


def _box_nms_sp_v6(
    scores_hw: Tensor, threshold: float, size: int = 4, iou_threshold: float = 0.1
) -> tuple[Tensor, Tensor]:
    """TF ``sp_v6`` ``box_nms``: filter ``scores >= threshold``, run 4x4 box NMS (IoU 0.1), return ``(kpts_xy, scores)``."""
    return ops.box_nms_2d(scores_hw, threshold, size, iou_threshold)


class _ReLUBatchNorm2d(nn.Module):
    """Eval-mode ``BatchNorm2d(ReLU(x))`` folded into one ``ops.relu_affine`` pass."""

    def __init__(self, bn: nn.BatchNorm2d) -> None:
        super().__init__()
        scale = bn.weight.float() * torch.rsqrt(bn.running_var.float() + bn.eps)
        shift = bn.bias.float() - bn.running_mean.float() * scale
        self.register_buffer("scale", scale.contiguous())
        self.register_buffer("shift", shift.contiguous())

    def forward(self, x: Tensor) -> Tensor:
        return ops.relu_affine(x, self.scale, self.shift)


def _conv_relu_bn(conv: nn.Module, bn: nn.Module, x: Tensor) -> Tensor:
    if isinstance(bn, _ReLUBatchNorm2d):
        return bn(conv(x))
    return bn(conv_relu(conv, x))


class SuperPoint(sparse.SparseExtractor):
    """Module paths follow ``superpoint_v1.pth`` (``conv1a`` ... ``convDb``); ``sp_v6`` adds ``bn1a`` ... ``bnDb``.

    ``detect``: grayscale -> encoder (1/8 resolution) -> ``convPa/convPb`` cell scores (softmax over the 64 cell
    positions + dustbin, FP32) -> pixel score map -> NMS -> top-k -> bilinearly sampled L2-normalized descriptors.
    The score softmax, NMS and sampling are FP32 in every precision preset."""

    div_by = 8
    fp32_modules = ()
    fused_stem_op: ops.BoundOp | None = None  # bound ``ops.conv3x3_small_cin_relu`` (pass ``fused_stem_conv``)
    fused_pool_epilogue: bool = False  # enabled by ``fused_pool_epilogue`` pass (int8, bit-identical)

    def __init__(
        self,
        descriptor_dim: int = 256,
        nms_radius: int = 4,
        detection_threshold: float | None = None,
        remove_borders: int | None = None,
        *,
        variant: str = "v1",
    ) -> None:
        super().__init__()
        if variant not in ("v1", "sp_v6"):
            raise ValueError(f"Unknown SuperPoint variant {variant!r}; choose from ('v1', 'sp_v6')")
        self.variant = variant
        self.has_bn = variant == "sp_v6"
        self.nms_radius = nms_radius
        self.detection_threshold = (0.01 if self.has_bn else 0.0005) if detection_threshold is None else detection_threshold
        self.remove_borders = (0 if self.has_bn else 4) if remove_borders is None else remove_borders
        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        c1, c2, c3, c4, c5 = 64, 64, 128, 128, 256
        self.conv1a = nn.Conv2d(1, c1, 3, 1, 1)
        self.conv1b = nn.Conv2d(c1, c1, 3, 1, 1)
        self.conv2a = nn.Conv2d(c1, c2, 3, 1, 1)
        self.conv2b = nn.Conv2d(c2, c2, 3, 1, 1)
        self.conv3a = nn.Conv2d(c2, c3, 3, 1, 1)
        self.conv3b = nn.Conv2d(c3, c3, 3, 1, 1)
        self.conv4a = nn.Conv2d(c3, c4, 3, 1, 1)
        self.conv4b = nn.Conv2d(c4, c4, 3, 1, 1)
        self.convPa = nn.Conv2d(c4, c5, 3, 1, 1)
        self.convPb = nn.Conv2d(c5, 65, 1, 1, 0)
        self.convDa = nn.Conv2d(c4, c5, 3, 1, 1)
        self.convDb = nn.Conv2d(c5, descriptor_dim, 1, 1, 0)
        if self.has_bn:
            # TF sp_v6 BatchNorm epsilon = 1e-3 (FusedBatchNorm default in TF1)
            self.bn1a = nn.BatchNorm2d(c1, eps=1e-3)
            self.bn1b = nn.BatchNorm2d(c1, eps=1e-3)
            self.bn2a = nn.BatchNorm2d(c2, eps=1e-3)
            self.bn2b = nn.BatchNorm2d(c2, eps=1e-3)
            self.bn3a = nn.BatchNorm2d(c3, eps=1e-3)
            self.bn3b = nn.BatchNorm2d(c3, eps=1e-3)
            self.bn4a = nn.BatchNorm2d(c4, eps=1e-3)
            self.bn4b = nn.BatchNorm2d(c4, eps=1e-3)
            self.bnPa = nn.BatchNorm2d(c5, eps=1e-3)
            self.bnPb = nn.BatchNorm2d(65, eps=1e-3)
            self.bnDa = nn.BatchNorm2d(c5, eps=1e-3)
            self.bnDb = nn.BatchNorm2d(descriptor_dim, eps=1e-3)

    @classmethod
    def from_pretrained(
        cls,
        *,
        checkpoint: str | Path | None = None,
        device: torch.device | str | None = None,
        variant: str = "v1",
        **kwargs,
    ) -> SuperPoint:
        with torch.device("meta"):
            model = cls(variant=variant, **kwargs)
        model.load_state_dict(weights.load_state_dict(checkpoint, variant=variant), strict=True, assign=True)
        return model.to(torch.device(device or default_device())).eval()

    def batchnorm_pairs(self) -> list[tuple[str, str]]:
        # In sp_v6, convPb -> bnPb and convDb -> bnDb have no ReLU in between and fold exactly.
        # The trunk / convPa / convDa layers are Conv -> ReLU -> BN, so BN stays after ReLU.
        return [("convPb", "bnPb"), ("convDb", "bnDb")] if self.has_bn else []

    def fold_relu_batchnorm(self) -> list[str]:
        """Fold eval-mode ``ReLU -> BatchNorm2d`` in ``sp_v6`` into fused ``ops.relu_affine`` modules."""
        if not self.has_bn:
            return []
        folded: list[str] = []
        for name in ("bn1a", "bn1b", "bn2a", "bn2b", "bn3a", "bn3b", "bn4a", "bn4b", "bnPa", "bnDa"):
            mod = getattr(self, name)
            if isinstance(mod, nn.BatchNorm2d):
                setattr(self, name, _ReLUBatchNorm2d(mod))
                folded.append(name)
        return folded

    def _conv_pool(self, conv: nn.Module, x: Tensor, consumer: nn.Module) -> Tensor:
        if self.fused_pool_epilogue and isinstance(conv, StaticInt8Conv2d) and conv.pools_into(self.pool, consumer):
            q, _ = conv._quantized_input(x)
            pooled = ops.int8_conv2d_maxpool_only(
                q.values, conv.weight_int8, conv.input_scale, conv.weight_scale, conv.bias,
                *conv._geometry_args(), relu=True, out_scale=consumer.input_scale,
            )
            return Int8Tensor(pooled, consumer.input_scale)
        return self.pool(conv_relu(conv, x))

    def encode(self, image: Tensor) -> Tensor:
        gray = sparse.rgb_to_gray(image) if image.shape[1] == 3 else image
        x = gray.to(self.compute_dtype)
        if not self.has_bn:
            if self.fused_stem_op is not None:
                if isinstance(self.conv1b, StaticInt8Conv2d):
                    x = self.fused_stem_op(x, out_scale=self.conv1b.input_scale)
                elif isinstance(self.conv1b, StaticMXFP8Conv2d):
                    x = self.fused_stem_op(x, out_exponent=self.conv1b.channel_exponent)
                else:
                    x = self.fused_stem_op(x)
            else:
                x = conv_relu(self.conv1a, x, feeds=self.conv1b)
            x = self._conv_pool(self.conv1b, x, self.conv2a)
            x = conv_relu(self.conv2a, x, feeds=self.conv2b)
            x = self._conv_pool(self.conv2b, x, self.conv3a)
            x = conv_relu(self.conv3a, x, feeds=self.conv3b)
            x = self._conv_pool(self.conv3b, x, self.conv4a)
            return conv_relu(self.conv4b, conv_relu(self.conv4a, x, feeds=self.conv4b))
        if isinstance(self.conv1a, nn.Conv2d) and self.conv1a.weight.is_contiguous(memory_format=torch.channels_last):
            x = x.to(memory_format=torch.channels_last)
        x = _conv_relu_bn(self.conv1a, self.bn1a, x)
        x = self.pool(_conv_relu_bn(self.conv1b, self.bn1b, x))
        x = _conv_relu_bn(self.conv2a, self.bn2a, x)
        x = self.pool(_conv_relu_bn(self.conv2b, self.bn2b, x))
        x = _conv_relu_bn(self.conv3a, self.bn3a, x)
        x = self.pool(_conv_relu_bn(self.conv3b, self.bn3b, x))
        x = _conv_relu_bn(self.conv4a, self.bn4a, x)
        return _conv_relu_bn(self.conv4b, self.bn4b, x)

    def score_map(self, x: Tensor) -> Tensor:
        """Dense per-pixel scores ``[1, H, W]`` (FP32), before NMS."""
        if not self.has_bn:
            logits = self.convPb(conv_relu(self.convPa, x, feeds=self.convPb)).float()
        else:
            logits = self.bnPb(self.convPb(_conv_relu_bn(self.convPa, self.bnPa, x))).float()
        scores = F.softmax(logits, 1)[:, :-1]
        b, _, h, w = scores.shape
        return scores.permute(0, 2, 3, 1).reshape(b, h, w, 8, 8).permute(0, 1, 3, 2, 4).reshape(b, h * 8, w * 8)

    def _coarse_descriptors(self, x: Tensor) -> Tensor:
        if not self.has_bn:
            return self.convDb(conv_relu(self.convDa, x, feeds=self.convDb)).float()
        return self.bnDb(self.convDb(_conv_relu_bn(self.convDa, self.bnDa, x))).float()

    def dense_descriptors(self, x: Tensor) -> Tensor:
        return F.normalize(self._coarse_descriptors(x), dim=1)

    def _detect_trunk(self, image: Tensor) -> tuple[Tensor, Tensor]:
        x = self.encode(image)
        scores = sparse.simple_nms(self.score_map(x), self.nms_radius)
        if self.remove_borders:
            pad = self.remove_borders
            scores[:, :pad] = -1
            scores[:, :, :pad] = -1
            scores[:, -pad:] = -1
            scores[:, :, -pad:] = -1
        return scores, self.dense_descriptors(x)

    def detect(self, image: Tensor, top_k: int | None) -> tuple[Tensor, Tensor, Tensor]:
        if not self.has_bn:
            scores, dense_desc = self._run_graphed("detect_trunk", self._detect_trunk, image)
            _, ys, xs = torch.where(scores > self.detection_threshold)
            keypoints = torch.stack((xs, ys), -1)
            keypoints, kp_scores, _ = sparse.select_top_k(keypoints, scores[0, ys, xs], top_k)
            keypoints = keypoints.float()
            return keypoints, kp_scores, sparse.sample_descriptors(keypoints, dense_desc, 8)
        x = self.encode(image)
        score_map = self.score_map(x)[0]
        keypoints, kp_scores = _box_nms_sp_v6(score_map, self.detection_threshold, size=self.nms_radius)
        keypoints, kp_scores, _ = sparse.select_top_k(keypoints, kp_scores, top_k)
        descriptors = _sample_descriptors_sp_v6(keypoints, self._coarse_descriptors(x), 8)
        return keypoints, kp_scores, descriptors

    def describe_keypoints(self, image: Tensor, keypoints: Tensor) -> Tensor:
        x = self.encode(image)
        if not self.has_bn:
            return sparse.sample_descriptors(keypoints.float(), self.dense_descriptors(x), 8)
        return _sample_descriptors_sp_v6(keypoints.float(), self._coarse_descriptors(x), 8)

    def forward(self, image: Tensor) -> tuple[Tensor, Tensor]:
        """Dense outputs: ``(score_map [1, H, W], descriptors [1, C, H/8, W/8])`` (post-softmax, pre-NMS)."""
        x = self.encode(image)
        return self.score_map(x), self.dense_descriptors(x)
