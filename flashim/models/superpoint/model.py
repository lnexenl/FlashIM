"""SuperPoint (DeTone et al., 2018): VGG-style encoder, 65-way cell-softmax keypoint head, 256-d descriptor head."""

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


class SuperPoint(sparse.SparseExtractor):
    """Module paths follow ``superpoint_v1.pth`` (``conv1a`` ... ``convDb``); loads strictly.

    ``detect``: grayscale -> encoder (1/8 resolution) -> ``convPa/convPb`` cell scores (softmax over the 64 cell
    positions + dustbin, FP32) -> pixel score map -> ``simple_nms`` (radius 4) -> border removal -> threshold ->
    top-k; descriptors are L2-normalized densely, bilinearly sampled at the keypoints and re-normalized. The score
    softmax, NMS and sampling are FP32 in every precision preset."""

    div_by = 8
    fp32_modules = ()
    fused_stem_op: ops.BoundOp | None = None  # bound ``ops.conv3x3_small_cin_relu`` (pass ``fused_stem_conv``)
    fused_pool_epilogue: bool = False  # enabled by ``fused_pool_epilogue`` pass (int8, bit-identical)

    def __init__(self, descriptor_dim: int = 256, nms_radius: int = 4, detection_threshold: float = 0.0005,
                 remove_borders: int = 4) -> None:
        super().__init__()
        self.nms_radius, self.detection_threshold, self.remove_borders = nms_radius, detection_threshold, remove_borders
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

    @classmethod
    def from_pretrained(cls, *, checkpoint: str | Path | None = None, device: torch.device | str | None = None,
                        **kwargs) -> SuperPoint:
        with torch.device("meta"):
            model = cls(**kwargs)
        model.load_state_dict(weights.load_state_dict(checkpoint), strict=True, assign=True)
        return model.to(torch.device(device or default_device())).eval()

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
        gray = sparse.rgb_to_gray(image).to(self.compute_dtype)
        if self.fused_stem_op is not None:
            if isinstance(self.conv1b, StaticInt8Conv2d):
                x = self.fused_stem_op(gray, out_scale=self.conv1b.input_scale)
            elif isinstance(self.conv1b, StaticMXFP8Conv2d):
                x = self.fused_stem_op(gray, out_exponent=self.conv1b.channel_exponent)
            else:
                x = self.fused_stem_op(gray)
        else:
            x = conv_relu(self.conv1a, gray, feeds=self.conv1b)
        x = self._conv_pool(self.conv1b, x, self.conv2a)
        x = conv_relu(self.conv2a, x, feeds=self.conv2b)
        x = self._conv_pool(self.conv2b, x, self.conv3a)
        x = conv_relu(self.conv3a, x, feeds=self.conv3b)
        x = self._conv_pool(self.conv3b, x, self.conv4a)
        return conv_relu(self.conv4b, conv_relu(self.conv4a, x, feeds=self.conv4b))

    def score_map(self, x: Tensor) -> Tensor:
        """Dense per-pixel scores ``[1, H, W]`` (FP32), before NMS."""
        scores = F.softmax(self.convPb(conv_relu(self.convPa, x, feeds=self.convPb)).float(), 1)[:, :-1]
        b, _, h, w = scores.shape
        return scores.permute(0, 2, 3, 1).reshape(b, h, w, 8, 8).permute(0, 1, 3, 2, 4).reshape(b, h * 8, w * 8)

    def dense_descriptors(self, x: Tensor) -> Tensor:
        return F.normalize(self.convDb(conv_relu(self.convDa, x, feeds=self.convDb)).float(), dim=1)

    def detect(self, image: Tensor, top_k: int | None) -> tuple[Tensor, Tensor, Tensor]:
        x = self.encode(image)
        scores = sparse.simple_nms(self.score_map(x), self.nms_radius)
        if self.remove_borders:
            pad = self.remove_borders
            scores[:, :pad] = -1
            scores[:, :, :pad] = -1
            scores[:, -pad:] = -1
            scores[:, :, -pad:] = -1
        _, ys, xs = torch.where(scores > self.detection_threshold)
        keypoints = torch.stack((xs, ys), -1)
        keypoints, kp_scores, _ = sparse.select_top_k(keypoints, scores[0, ys, xs], top_k)
        keypoints = keypoints.float()
        return keypoints, kp_scores, sparse.sample_descriptors(keypoints, self.dense_descriptors(x), 8)

    def describe_keypoints(self, image: Tensor, keypoints: Tensor) -> Tensor:
        return sparse.sample_descriptors(keypoints.float(), self.dense_descriptors(self.encode(image)), 8)

    def forward(self, image: Tensor) -> tuple[Tensor, Tensor]:
        """Dense outputs: ``(score_map [1, H, W], descriptors [1, C, H/8, W/8])`` (post-softmax, pre-NMS)."""
        x = self.encode(image)
        return self.score_map(x), self.dense_descriptors(x)
