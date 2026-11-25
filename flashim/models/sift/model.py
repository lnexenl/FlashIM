"""SIFT (Lowe, 2004): OpenCV's implementation as a FlashIM sparse extractor (fp32 baseline; no learned weights).

Detection / description run on the CPU in OpenCV (there is nothing to quantize: the hand-crafted pipeline has no
weights, so ``bf16`` / ``int8`` / ``fp8`` are not offered); matching runs on the model's device: mutual nearest
neighbours with Lowe's ratio test on L2 distances.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import Tensor

from flashim import ops
from flashim.models import sparse
from flashim.types import Matches, SparseFeatures
from flashim.utils import import_cv2


def gray_uint8(image: Tensor) -> np.ndarray:
    """``[1, 3, H, W]`` in ``[0, 1]`` -> ``[H, W]`` uint8 luma."""
    return (sparse.rgb_to_gray(image)[0, 0] * 255.0).round().clamp(0, 255).byte().cpu().numpy()


def ratio_mutual_match(dist: Tensor, ratio: float) -> tuple[Tensor, Tensor, Tensor]:
    """``dist`` ``[N, M]`` (smaller = closer): mutual nearest neighbours passing ``d1 < ratio * d2``."""
    return ops.ratio_mutual_match(dist, ratio)


class SIFT(sparse.SparseExtractor):
    """``extract`` (inherited): resize to the working resolution, OpenCV ``SIFT_create(nfeatures=top_k)``; descriptors
    are L2-normalized; ``scores`` are OpenCV's keypoint responses. Presets: ``fp32`` only."""

    div_by = 1
    ratio = 0.8

    def __init__(self, contrast_threshold: float = 0.04, edge_threshold: float = 10.0, sigma: float = 1.6) -> None:
        super().__init__()
        import_cv2()  # fail at load time, not at the first match
        self.contrast_threshold, self.edge_threshold, self.sigma = contrast_threshold, edge_threshold, sigma
        self._detectors: dict[int, object] = {}
        self.register_buffer("_device_anchor", torch.zeros(()), persistent=False)  # carries the device of matching

    @classmethod
    def from_pretrained(cls, *, device: torch.device | str | None = None, **kwargs) -> SIFT:
        from flashim.platforms import default_device

        return cls(**kwargs).to(torch.device(device or default_device())).eval()

    @property
    def device(self) -> torch.device:
        return self._device_anchor.device

    def _detector(self, nfeatures: int):
        sift = self._detectors.get(nfeatures)
        if sift is None:
            cv2 = import_cv2()
            sift = cv2.SIFT_create(
                nfeatures=nfeatures, contrastThreshold=self.contrast_threshold, edgeThreshold=self.edge_threshold, sigma=self.sigma
            )
            self._detectors[nfeatures] = sift
        return sift

    def detect(self, image: Tensor, top_k: int | None) -> tuple[Tensor, Tensor, Tensor]:
        keypoints, descriptors = self._detector(top_k or 0).detectAndCompute(gray_uint8(image), None)
        if not keypoints:
            return image.new_zeros(0, 2), image.new_zeros(0), image.new_zeros(0, 128)
        kpt_arr = np.array([(kp.pt[0], kp.pt[1], kp.response) for kp in keypoints], dtype=np.float32)
        kpt_t = torch.from_numpy(kpt_arr).to(image.device)
        desc = torch.from_numpy(descriptors).to(image.device)
        return kpt_t[:, :2].contiguous(), kpt_t[:, 2].contiguous(), torch.nn.functional.normalize(desc, dim=-1)

    @torch.inference_mode()
    def match_features(self, a: SparseFeatures, b: SparseFeatures, threshold: float | None = None) -> Matches:
        dist = torch.cdist(a.descriptors[0].float(), b.descriptors[0].float())
        ia, ib, conf = ratio_mutual_match(dist, self.ratio if threshold is None else threshold)
        return Matches(a.keypoints[0][ia], b.keypoints[0][ib], conf, ia, ib)
