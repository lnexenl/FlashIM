"""RIPE (Kunzel et al., ICCV 2025): DeDoDe-style VGG19 + ConvRefiner detector heatmap; descriptors are hypercolumns of the
VGG feature maps sampled at the keypoints and reduced by a 1x1 conv."""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from flashim import ops
from flashim.models import sparse
from flashim.models.dedode import blocks
from flashim.models.ripe import weights
from flashim.platforms import default_device
from flashim.types import Matches, SparseFeatures
from flashim.utils import device_constant


class RIPE(sparse.SparseExtractor):
    """Module paths follow ``ripe_weights.pth`` (``net.encoder.layers.N``, ``net.decoder.layers.*``,
    ``conv_dim_reduction_coarse_desc``); loads strictly.

    ``detect`` (upstream ``detectAndCompute``): instance norm -> VGG19 + decoder heatmap (raw logits) -> 3x3 max-pool NMS
    above ``threshold`` -> top-k by heatmap value -> hypercolumn descriptors (bilinear samples of the four VGG maps at the
    keypoints, ``align_corners=True``, concatenated 960-d) -> 1x1 conv to 256-d -> L2 normalization; the score is the
    heatmap value over the map's maximum."""

    div_by = 8
    match_threshold = 0.0  # upstream: kornia mutual nearest neighbour, no threshold

    def __init__(self, threshold: float = 0.5) -> None:
        super().__init__()
        self.threshold = threshold
        self.net = blocks.DeDoDeNet(blocks.RIPE_DETECTOR)
        self.conv_dim_reduction_coarse_desc = nn.Conv1d(960, 256, kernel_size=1)

    @classmethod
    def from_pretrained(cls, *, checkpoint: str | Path | None = None, device: torch.device | str | None = None,
                        **kwargs) -> RIPE:
        with torch.device("meta"):
            model = cls(**kwargs)
        model.load_state_dict(weights.load_state_dict(checkpoint), strict=True, assign=True)
        return model.to(torch.device(device or default_device())).eval()

    def batchnorm_pairs(self) -> list[tuple[str, str]]:
        return self.net.batchnorm_pairs("net.")

    def heatmap(self, image: Tensor) -> tuple[Tensor, list[Tensor]]:
        x = F.instance_norm(image).to(self.compute_dtype)
        out, features = self.net(x, return_features=True)
        return out.float(), features

    def detect(self, image: Tensor, top_k: int | None) -> tuple[Tensor, Tensor, Tensor]:
        h, w = image.shape[-2:]
        heat, features = self.heatmap(image)
        keep = (heat == F.max_pool2d(heat, 3, 1, 1)) & (heat > self.threshold)
        ys, xs = keep[0, 0].nonzero(as_tuple=True)
        keypoints = torch.stack((xs, ys), -1)
        scores = heat[0, 0, ys, xs]
        order = torch.argsort(-scores)[: top_k if top_k else None]
        keypoints, scores = keypoints[order], scores[order]
        return keypoints.float(), scores / heat.max(), self._describe(features, keypoints, h, w)

    def describe_keypoints(self, image: Tensor, keypoints: Tensor) -> Tensor:
        h, w = image.shape[-2:]
        return self._describe(self.heatmap(image)[1], keypoints, h, w)

    def _describe(self, features: list[Tensor], keypoints: Tensor, h: int, w: int) -> Tensor:
        grid = 2.0 * keypoints.float() / device_constant([w - 1, h - 1], keypoints, torch.float32) - 1.0
        hyper = ops.bilinear_sample_points(
            features, grid, align_corners=True, out_dtype=self.conv_dim_reduction_coarse_desc.weight.dtype
        )  # [K, 960]
        # Keep the 1x1 Conv1d call (instead of F.linear): cuDNN Conv1d and cuBLAS GEMM have different BF16
        # rounding orders (max_diff 2.4e-3) which shifted MegaDepth RIPE BF16 AUC@5 by -0.0042.
        desc = self.conv_dim_reduction_coarse_desc(hyper.t().contiguous()[None])[0].t()
        return F.normalize(desc.float(), dim=-1)

    @torch.inference_mode()
    def match_features(self, a: SparseFeatures, b: SparseFeatures, threshold: float | None = None) -> Matches:
        limit = self.match_threshold if threshold is None else threshold
        # Keep FP32 descriptors (TF32 tiled Triton kernel: 2.30x faster and 13.5x less VRAM than dense FP32 at 32k^2,
        # <0.4 ms at 4096^2): casting to BF16 dropped ScanNet MNN overlap from 99.95% to 99.47%.
        ia, ib, score = ops.mutual_nn_match(a.descriptors[0], b.descriptors[0], limit)
        return Matches(a.keypoints[0][ia], b.keypoints[0][ib], score, ia, ib)
