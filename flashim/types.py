"""Result containers returned by FlashIM matchers."""

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
