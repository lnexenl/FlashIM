from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Setting:
    """Resolution / direction preset (RoMaV2 ``apply_setting``)."""

    h_lr: int
    w_lr: int
    h_hr: int | None
    w_hr: int | None
    bidirectional: bool
    threshold: float | None


_BENCH = Setting(800, 800, 1024, 1024, True, 0.05)

SETTINGS: dict[str, Setting] = {
    "mega1500": _BENCH,
    "scannet1500": _BENCH,
    "wxbs": _BENCH,
    "satast": _BENCH,
    "turbo": Setting(320, 320, None, None, False, None),
    "fast": Setting(512, 512, None, None, False, None),
    "base": Setting(640, 640, None, None, False, None),
    "precise": Setting(800, 800, 1280, 1280, True, None),
}


@dataclass(frozen=True)
class RoMaV2Config:
    setting: str = "precise"
    anchor_width: int = 512
    anchor_height: int = 512
    # Drop DINOv3 blocks after the last used layer (17); outputs are unchanged.
    truncate_descriptor: bool = True
    # Run the refiner fine features of A and B as one batch.
    batch_refiner_features: bool = True
