"""ORB (Rublee et al., 2011): OpenCV's implementation as a FlashIM sparse extractor (fp32 baseline; binary descriptors).

Detection / description run on the CPU in OpenCV; the 256-bit descriptors are kept packed (``uint8 [K, 32]``) and matched
on the model's device by Hamming distance (mutual nearest neighbours + ratio test). No weights, hence no ``bf16`` /
``int8`` / ``fp8`` presets.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import Tensor

from flashim import ops
from flashim.models.sift.model import SIFT, gray_uint8
from flashim.types import Matches, SparseFeatures
from flashim.utils import import_cv2


def hamming_distances(a: Tensor, b: Tensor, chunk: int = 1024) -> Tensor:
    """``a`` ``[N, 32]``, ``b`` ``[M, 32]`` uint8 -> Hamming distances ``[N, M]`` (float)."""
    return ops.hamming_distances(a, b)


class ORB(SIFT):
    """``extract``: OpenCV ``ORB_create(nfeatures=top_k)`` (8 pyramid levels, scale 1.2, FAST threshold 20); the
    ``descriptors`` of :class:`SparseFeatures` are packed bits ``[K, 32]`` uint8; ``scores`` are FAST/Harris responses."""

    ratio = 0.8

    def __init__(self, fast_threshold: int = 20, n_levels: int = 8, scale_factor: float = 1.2) -> None:
        super().__init__()
        import_cv2()  # fail at load time, not at the first match
        self.fast_threshold, self.n_levels, self.scale_factor = fast_threshold, n_levels, scale_factor
        self._detectors: dict[int, object] = {}

    def _detector(self, nfeatures: int):
        orb = self._detectors.get(nfeatures)
        if orb is None:
            cv2 = import_cv2()
            orb = cv2.ORB_create(
                nfeatures=nfeatures, scaleFactor=self.scale_factor, nlevels=self.n_levels, fastThreshold=self.fast_threshold
            )
            self._detectors[nfeatures] = orb
        return orb

    def detect(self, image: Tensor, top_k: int | None) -> tuple[Tensor, Tensor, Tensor]:
        keypoints, descriptors = self._detector(top_k or 500).detectAndCompute(gray_uint8(image), None)
        if not keypoints:
            return image.new_zeros(0, 2), image.new_zeros(0), torch.zeros(0, 32, dtype=torch.uint8, device=image.device)
        kpt_arr = np.array([(kp.pt[0], kp.pt[1], kp.response) for kp in keypoints], dtype=np.float32)
        kpt_t = torch.from_numpy(kpt_arr).to(image.device)
        return kpt_t[:, :2].contiguous(), kpt_t[:, 2].contiguous(), torch.from_numpy(descriptors).to(image.device)

    @torch.inference_mode()
    def match_features(self, a: SparseFeatures, b: SparseFeatures, threshold: float | None = None) -> Matches:
        ratio = self.ratio if threshold is None else threshold
        ia, ib, conf = ops.hamming_ratio_mutual_match(a.descriptors[0], b.descriptors[0], ratio)
        return Matches(a.keypoints[0][ia], b.keypoints[0][ib], conf, ia, ib)
