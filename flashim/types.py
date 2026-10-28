"""Result containers returned by FlashIM models: :class:`Matches` from every model's ``match``, the per-image
:class:`SparseFeatures` of detector/descriptor models and the full :class:`DenseMatchResult` of dense ones."""

from __future__ import annotations

from dataclasses import dataclass, fields

from torch import Tensor


@dataclass
class DenseMatchResult:
    """Dense correspondences from image A to B (and B to A when bidirectional).

    ``warp_*``: ``[B, H, W, 2]`` target coordinates in normalized ``[-1, 1]``
    (``align_corners=False``); ``confidence_*``: raw ``[B, H, W, 4]`` (overlap
    logit + precision parameters); ``overlap_*``: ``[B, H, W, 1]`` in ``[0, 1]``;
    ``precision_*``: ``[B, H, W, 2, 2]`` precision matrices.
    """

    warp_AB: Tensor
    confidence_AB: Tensor
    overlap_AB: Tensor
    precision_AB: Tensor
    warp_BA: Tensor | None = None
    confidence_BA: Tensor | None = None
    overlap_BA: Tensor | None = None
    precision_BA: Tensor | None = None

    @property
    def bidirectional(self) -> bool:
        return self.warp_BA is not None

    def as_dict(self) -> dict[str, Tensor | None]:
        return {f.name: getattr(self, f.name) for f in fields(self)}


@dataclass
class SparseFeatures:
    """Keypoints and descriptors of one image batch.

    ``keypoints`` ``[B, K, 2]``: ``(x, y)`` in pixel-index coordinates of the *original* image (integer =
    pixel center); ``scores`` ``[B, K]``: detector score map value at each keypoint; ``descriptors``
    ``[B, K, C]``: L2-normalized; ``image_size``: ``(H, W)`` of the original image.
    """

    keypoints: Tensor
    scores: Tensor
    descriptors: Tensor
    image_size: tuple[int, int]


@dataclass
class Matches:
    """Correspondences between images A and B: what every model's ``match`` returns.

    ``keypoints_A`` / ``keypoints_B``: ``[N, 2]`` ``(x, y)`` in pixel coordinates of each *original* image (integer =
    pixel center); ``confidence``: ``[N]``, the
    model's own match score. Detector/descriptor models also return ``indices_A`` / ``indices_B`` into the keypoints
    of ``features_A`` / ``features_B`` (every keypoint of each image; ``None`` from ``match_features``, whose caller
    has them); dense models attach their full prediction as ``dense``.
    """

    keypoints_A: Tensor
    keypoints_B: Tensor
    confidence: Tensor
    indices_A: Tensor | None = None
    indices_B: Tensor | None = None
    features_A: SparseFeatures | None = None
    features_B: SparseFeatures | None = None
    dense: DenseMatchResult | None = None

    def __len__(self) -> int:
        return self.keypoints_A.shape[0]
