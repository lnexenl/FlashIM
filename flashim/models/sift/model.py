"""SIFT (Lowe, 2004): OpenCV's implementation as a FlashIM sparse extractor (fp32 baseline; no learned weights).

Detection / description run on the CPU in OpenCV (there is nothing to quantize: the hand-crafted pipeline has no
weights, so ``bf16`` / ``int8`` / ``fp8`` are not offered); matching runs on the model's device: mutual nearest
neighbours with Lowe's ratio test on L2 distances.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import Tensor

from flashim.models import sparse
from flashim.types import Matches, SparseFeatures


def gray_uint8(image: Tensor) -> np.ndarray:
    """``[1, 3, H, W]`` in ``[0, 1]`` -> ``[H, W]`` uint8 luma."""
    return (sparse.rgb_to_gray(image)[0, 0] * 255.0).round().clamp(0, 255).byte().cpu().numpy()


def ratio_mutual_match(dist: Tensor, ratio: float) -> tuple[Tensor, Tensor, Tensor]:
    """``dist`` ``[N, M]`` (smaller = closer): mutual nearest neighbours passing ``d1 < ratio * d2``."""
    if dist.shape[0] == 0 or dist.shape[1] < 2:
        empty = torch.empty(0, dtype=torch.long, device=dist.device)
        return empty, empty, dist.new_empty(0)
    best, index = dist.topk(2, dim=1, largest=False)
    ids = torch.arange(dist.shape[0], device=dist.device)
    keep = (best[:, 0] < ratio * best[:, 1]) & (dist.argmin(dim=0)[index[:, 0]] == ids)
    ia, ib = ids[keep], index[keep, 0]
    return ia, ib, 1.0 - best[keep, 0] / best[keep, 1]


class SIFT(sparse.SparseExtractor):
    """``extract`` (inherited): resize to the working resolution, OpenCV ``SIFT_create(nfeatures=top_k)``; descriptors
    are L2-normalized; ``scores`` are OpenCV's keypoint responses. Presets: ``fp32`` only."""

    div_by = 1
    ratio = 0.8

    def __init__(self, contrast_threshold: float = 0.04, edge_threshold: float = 10.0, sigma: float = 1.6) -> None:
        super().__init__()
        self.contrast_threshold, self.edge_threshold, self.sigma = contrast_threshold, edge_threshold, sigma
        self.register_buffer("_device_anchor", torch.zeros(()), persistent=False)  # carries the device of matching

    @classmethod
    def from_pretrained(cls, *, device: torch.device | str | None = None, **kwargs) -> SIFT:
        from flashim.platforms import default_device

        return cls(**kwargs).to(torch.device(device or default_device())).eval()

    @property
    def device(self) -> torch.device:
        return self._device_anchor.device

    def detect(self, image: Tensor, top_k: int | None) -> tuple[Tensor, Tensor, Tensor]:
        import cv2

        sift = cv2.SIFT_create(
            nfeatures=top_k or 0, contrastThreshold=self.contrast_threshold, edgeThreshold=self.edge_threshold, sigma=self.sigma
        )
        keypoints, descriptors = sift.detectAndCompute(gray_uint8(image), None)
        if not keypoints:
            return image.new_zeros(0, 2), image.new_zeros(0), image.new_zeros(0, 128)
        xy = torch.tensor([kp.pt for kp in keypoints], dtype=torch.float32, device=image.device)
        scores = torch.tensor([kp.response for kp in keypoints], dtype=torch.float32, device=image.device)
        desc = torch.from_numpy(descriptors).to(image.device)
        return xy, scores, torch.nn.functional.normalize(desc, dim=-1)

    @torch.inference_mode()
    def match_features(self, a: SparseFeatures, b: SparseFeatures, threshold: float | None = None) -> Matches:
        dist = torch.cdist(a.descriptors[0].float(), b.descriptors[0].float())
        ia, ib, conf = ratio_mutual_match(dist, self.ratio if threshold is None else threshold)
        return Matches(a.keypoints[0][ia], b.keypoints[0][ib], conf, ia, ib)
