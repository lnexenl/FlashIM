"""RoMa (v1) dense matcher configuration."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass


def _as_hw(value: int | Sequence[int], name: str) -> tuple[int, int]:
    if isinstance(value, int) and not isinstance(value, bool):
        return (value, value)
    if isinstance(value, Sequence) and len(value) == 2 and all(isinstance(v, int) and not isinstance(v, bool) for v in value):
        return (int(value[0]), int(value[1]))
    raise ValueError(f"{name} must be an int or (H, W) pair of ints, got {value!r}")


@dataclass(frozen=True)
class RoMaConfig:
    """Architecture and inference settings for :class:`~flashim.models.roma.model.RoMa`.

    Defaults match upstream ``roma_outdoor`` / ``roma_indoor``: coarse 560x560 (multiple of 14 for DINOv2 ViT-L/14),
    upsampled to 864x864 with bidirectional (symmetric) matching, certainty attenuation, and thresholded balanced
    sampling at 0.05.
    """

    coarse_res: tuple[int, int] = (560, 560)
    upsample_res: tuple[int, int] = (864, 864)
    upsample_preds: bool = True
    symmetric: bool = True
    attenuate_cert: bool = True
    sample_thresh: float = 0.05

    def __post_init__(self) -> None:
        coarse = _as_hw(self.coarse_res, "coarse_res")
        upsample = _as_hw(self.upsample_res, "upsample_res")
        if coarse[0] % 14 != 0 or coarse[1] % 14 != 0:
            raise ValueError(f"coarse_res must be divisible by 14 for DINOv2, got {coarse}")
        object.__setattr__(self, "coarse_res", coarse)
        object.__setattr__(self, "upsample_res", upsample)
