"""ALIKE checkpoints (``models/alike-{t,s,n,l}.pth`` of Shiaoming/ALIKE; keys match the module paths)."""

from __future__ import annotations

from pathlib import Path

import torch

from flashim.models.weights_util import find_weights

HELP = "Download <variant>.pth from the `models/` folder of https://github.com/Shiaoming/ALIKE"
LICENSES = {"code": "BSD-3-Clause (Shiaoming/ALIKE); FlashIM re-implementation", "weights": "ALIKE release checkpoints"}


def load_state_dict(variant: str, path: str | Path | None = None, map_location="cpu") -> dict[str, torch.Tensor]:
    file = find_weights(
        f"{variant}.pth", path=path, env="FLASHIM_ALIKE_CHECKPOINT", help=HELP,
        extra=[Path.home() / "work" / "oracles" / "ALIKE" / "models" / f"{variant}.pth"],
    )
    return torch.load(file, map_location=map_location, weights_only=True)
