"""DaD checkpoint (``dad.pth`` of Parskatt/dad release v0.1.0; ``encoder.*`` / ``decoder.*`` keys)."""

from __future__ import annotations

from pathlib import Path

import torch

from flashim.models.weights_util import Checkpoint, find_weights

CHECKPOINT = Checkpoint(
    "dad.pth",
    "https://github.com/Parskatt/dad/releases/download/v0.1.0/dad.pth",
    "7d827844649faf5213ba97e7adb43e16b936dd3d11ab2d0fc0d602c67fae781f",
)
LICENSES = {"code": "MIT (Parskatt/dad); FlashIM re-implementation", "weights": "DaD v0.1.0 release (dad.pth)"}


def load_state_dict(path: str | Path | None = None, map_location="cpu") -> dict[str, torch.Tensor]:
    file = find_weights(CHECKPOINT, path=path, env="FLASHIM_DAD_CHECKPOINT")
    return torch.load(file, map_location=map_location, weights_only=True)
