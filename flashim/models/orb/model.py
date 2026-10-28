"""ORB (Rublee et al., 2011): OpenCV's implementation as a FlashIM sparse extractor (fp32 baseline; binary descriptors).

Detection / description run on the CPU in OpenCV; the 256-bit descriptors are kept packed (``uint8 [K, 32]``) and matched
on the model's device by Hamming distance (mutual nearest neighbours + ratio test). No weights, hence no ``bf16`` /
``int8`` / ``fp8`` presets.
"""

from __future__ import annotations

import torch
from torch import Tensor

from flashim.models.sift.model import SIFT, gray_uint8, ratio_mutual_match
from flashim.types import Matches, SparseFeatures
from flashim.utils import import_cv2

_POPCOUNT = torch.tensor([i.bit_count() for i in range(256)], dtype=torch.uint8)


def hamming_distances(a: Tensor, b: Tensor, chunk: int = 1024) -> Tensor:
    """``a`` ``[N, 32]``, ``b`` ``[M, 32]`` uint8 -> Hamming distances ``[N, M]`` (float)."""
    table = _POPCOUNT.to(a.device)
    rows = [
        table[(a[i : i + chunk, None, :] ^ b[None, :, :]).long()].sum(-1, dtype=torch.int16)
        for i in range(0, a.shape[0], chunk)
    ]
    return torch.cat(rows).float() if rows else a.new_empty(0, b.shape[0]).float()


class ORB(SIFT):
    """``extract``: OpenCV ``ORB_create(nfeatures=top_k)`` (8 pyramid levels, scale 1.2, FAST threshold 20); the
    ``descriptors`` of :class:`SparseFeatures` are packed bits ``[K, 32]`` uint8; ``scores`` are FAST/Harris responses."""

    ratio = 0.8

    def __init__(self, fast_threshold: int = 20, n_levels: int = 8, scale_factor: float = 1.2) -> None:
        super().__init__()
        import_cv2()  # fail at load time, not at the first match
        self.fast_threshold, self.n_levels, self.scale_factor = fast_threshold, n_levels, scale_factor

    def detect(self, image: Tensor, top_k: int | None) -> tuple[Tensor, Tensor, Tensor]:
        cv2 = import_cv2()

        orb = cv2.ORB_create(
            nfeatures=top_k or 500, scaleFactor=self.scale_factor, nlevels=self.n_levels, fastThreshold=self.fast_threshold
        )
        keypoints, descriptors = orb.detectAndCompute(gray_uint8(image), None)
        if not keypoints:
            return image.new_zeros(0, 2), image.new_zeros(0), torch.zeros(0, 32, dtype=torch.uint8, device=image.device)
        xy = torch.tensor([kp.pt for kp in keypoints], dtype=torch.float32, device=image.device)
        scores = torch.tensor([kp.response for kp in keypoints], dtype=torch.float32, device=image.device)
        return xy, scores, torch.from_numpy(descriptors).to(image.device)

    @torch.inference_mode()
    def match_features(self, a: SparseFeatures, b: SparseFeatures, threshold: float | None = None) -> Matches:
        dist = hamming_distances(a.descriptors[0], b.descriptors[0])
        ia, ib, conf = ratio_mutual_match(dist, self.ratio if threshold is None else threshold)
        return Matches(a.keypoints[0][ia], b.keypoints[0][ib], conf, ia, ib)
