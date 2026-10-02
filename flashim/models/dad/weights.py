"""DaD checkpoint (``dad.pth`` of Parskatt/dad release v0.1.0; ``encoder.*`` / ``decoder.*`` keys)."""

from __future__ import annotations

from pathlib import Path

import torch

from flashim.models.weights_util import find_weights

HELP = "Download https://github.com/Parskatt/dad/releases/download/v0.1.0/dad.pth"
LICENSES = {"code": "MIT (Parskatt/dad); FlashIM re-implementation", "weights": "DaD v0.1.0 release (dad.pth)"}


def load_state_dict(path: str | Path | None = None, map_location="cpu") -> dict[str, torch.Tensor]:
    return torch.load(find_weights("dad.pth", path=path, env="FLASHIM_DAD_CHECKPOINT", help=HELP), map_location=map_location, weights_only=False)
