"""RIPE++ (Kunzel et al., LIMIT@ECCV 2026): reinforced keypoint learning from positive pairs only.

Shares RIPE's VGG19 + ConvRefiner detector backbone (``blocks.DeDoDeNet(blocks.RIPE_DETECTOR)``) and 960 -> 256 1x1
hypercolumn descriptor reduction, with four inference differences read from the released checkpoints' embedded config:

1. ImageNet mean/std normalization instead of per-image InstanceNorm (``use_instance_norm=False`` in the checkpoint
   config; ImageNet ``Normalize`` is applied during training, validation, Glue Factory eval, and the HF demo).
2. Subpixel keypoint refinement (3x3 softmax around each NMS peak at ``sub_pixel_temp=0.5``, zero-padded borders).
3. Hypercolumn descriptors sampled with ``align_corners=False`` at subpixel coordinates (RIPE used ``align_corners=True``
   at integer coordinates).
4. Upstream ``detectAndCompute`` returns corner-origin coordinates in ``[0.5, W - 0.5]`` (pixel ``i``'s center is at
   ``i + 0.5``). :meth:`RIPEPP.detect` converts to FlashIM's integer-is-pixel-center convention (``upstream - 0.5``) and
   sets ``matcher_keypoint_offset = 0.5`` so ``ripepp-lightglue`` feeds LightGlue the exact ``[0.5, W - 0.5]``
   coordinates its weights were trained on.
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from flashim import ops
from flashim.models import sparse
from flashim.models.dedode import blocks
from flashim.models.dedode.model import MEAN, STD
from flashim.models.ripepp import weights
from flashim.platforms import default_device
from flashim.types import Matches, SparseFeatures
from flashim.utils import device_constant


class RIPEPP(sparse.SparseExtractor):
    """Module paths follow ``ripe++.ckpt`` (``net.encoder.layers.N``, ``net.decoder.layers.*``,
    ``conv_dim_reduction_coarse_desc``); loads strictly."""

    div_by = 8
    default_top_k = 2048  # upstream KeypointSampler default
    match_threshold = 0.0  # upstream: mutual nearest neighbour, no threshold
    matcher_keypoint_offset = 0.5  # ripe++lightglue.tar was trained on upstream [0.5, W - 0.5] keypoints

    def __init__(
        self,
        threshold: float = 0.5,
        nms_size: int = 3,
        subpixel_sampling: bool = True,
        sub_pixel_temp: float = 0.5,
    ) -> None:
        super().__init__()
        self.threshold = threshold
        self.nms_size = nms_size
        self.subpixel_sampling = subpixel_sampling
        self.sub_pixel_temp = sub_pixel_temp
        self.net = blocks.DeDoDeNet(blocks.RIPE_DETECTOR)
        self.conv_dim_reduction_coarse_desc = nn.Conv1d(960, 256, kernel_size=1)
        self.register_buffer("mean", torch.tensor(MEAN).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("std", torch.tensor(STD).view(1, 3, 1, 1), persistent=False)

    @classmethod
    def from_pretrained(
        cls,
        *,
        variant: str = "default",
        checkpoint: str | Path | None = None,
        device: torch.device | str | None = None,
        **kwargs,
    ) -> RIPEPP:
        with torch.device("meta"):
            model = cls(**kwargs)
        model.load_state_dict(weights.load_state_dict(variant=variant, path=checkpoint), strict=True, assign=True)
        model.mean, model.std = (torch.tensor(v).view(1, 3, 1, 1) for v in (MEAN, STD))  # meta-built buffers
        return model.to(torch.device(device or default_device())).eval()

    def batchnorm_pairs(self) -> list[tuple[str, str]]:
        return self.net.batchnorm_pairs("net.")

    def normalize(self, image: Tensor) -> Tensor:
        return ((image - self.mean) / self.std).to(self.compute_dtype)

    def heatmap(self, image: Tensor) -> tuple[Tensor, list[Tensor]]:
        out, features = self.net(self.normalize(image), return_features=True)
        return out.float(), features

    def detect(self, image: Tensor, top_k: int | None) -> tuple[Tensor, Tensor, Tensor]:
        h, w = image.shape[-2:]
        heat, features = self.heatmap(image)
        if self.nms_size > 0:
            keep = (heat == F.max_pool2d(heat, self.nms_size, 1, self.nms_size // 2)) & (heat > self.threshold)
        else:
            keep = heat > self.threshold
        ys, xs = keep[0, 0].nonzero(as_tuple=True)
        scores = heat[0, 0, ys, xs]
        if top_k and top_k > 0 and len(scores) > top_k:
            scores, order = torch.topk(scores, top_k, dim=0, sorted=True)
            xs, ys = xs[order], ys[order]
        elif len(scores) > 0:
            order = torch.argsort(-scores)
            xs, ys, scores = xs[order], ys[order], scores[order]

        keypoints = torch.stack((xs, ys), -1).float()
        if self.subpixel_sampling and self.nms_size > 0 and len(xs) > 0:
            k = self.nms_size
            pad = k // 2
            steps = torch.arange(k, device=heat.device)
            padded = F.pad(heat[0, 0], (pad,) * 4)
            patches = padded[(ys[:, None] + steps)[:, :, None], (xs[:, None] + steps)[:, None, :]].reshape(-1, k * k).t()
            offsets = torch.stack(torch.meshgrid(steps - pad, steps - pad, indexing="ij"), -1)
            offsets = offsets.reshape(k * k, 2).flip(-1).float()  # (dx, dy), row-major like the patch unfolding
            keypoints = keypoints + torch.einsum("kn,kd->nd", (patches / self.sub_pixel_temp).softmax(0), offsets)

        if len(scores) > 0:
            scores = scores / heat.max()
        return keypoints, scores, self._describe(features, keypoints, h, w)

    def describe_keypoints(self, image: Tensor, keypoints: Tensor) -> Tensor:
        h, w = image.shape[-2:]
        return self._describe(self.heatmap(image)[1], keypoints, h, w)

    def _describe(self, features: list[Tensor], keypoints: Tensor, h: int, w: int) -> Tensor:
        if keypoints.shape[0] == 0:
            return keypoints.new_empty((0, self.conv_dim_reduction_coarse_desc.out_channels), dtype=torch.float32)
        grid = (keypoints.float() + 0.5) * (2.0 / device_constant([w, h], keypoints, torch.float32)) - 1.0
        hyper = ops.bilinear_sample_points(
            features,
            grid,
            align_corners=False,
            out_dtype=self.conv_dim_reduction_coarse_desc.weight.dtype,
        )  # [K, 960]
        # Keep the 1x1 Conv1d call (instead of F.linear): cuDNN Conv1d and cuBLAS GEMM have different BF16 rounding
        # orders (see RIPE's note).
        desc = self.conv_dim_reduction_coarse_desc(hyper.t().contiguous()[None])[0].t()
        return F.normalize(desc.float(), dim=-1)

    @torch.inference_mode()
    def match_features(self, a: SparseFeatures, b: SparseFeatures, threshold: float | None = None) -> Matches:
        limit = self.match_threshold if threshold is None else threshold
        # Keep FP32 descriptors (TF32 tiled Triton kernel; see RIPE's note).
        ia, ib, score = ops.mutual_nn_match(a.descriptors[0], b.descriptors[0], limit)
        return Matches(a.keypoints[0][ia], b.keypoints[0][ib], score, ia, ib)
