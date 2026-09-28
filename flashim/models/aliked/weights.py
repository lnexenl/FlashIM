"""ALIKED checkpoint location and loading (module names match the released checkpoints one-to-one)."""

from __future__ import annotations

import os
from pathlib import Path

import torch

CHECKPOINT_ENV = "FLASHIM_ALIKED_CHECKPOINT"
CHECKPOINT_HELP = (
    "Download <variant>.pth from the `models/` folder of https://github.com/Shiaoming/ALIKED, then pass "
    f"checkpoint=..., set {CHECKPOINT_ENV}, or place it at <torch hub dir>/checkpoints/<variant>.pth."
)

LICENSES = {
    "code": "BSD-3-Clause (ALIKED, Shiaoming/ALIKED); FlashIM re-implementation",
    "weights": "ALIKED release checkpoints (aliked-t16 / aliked-n16 / aliked-n16rot / aliked-n32)",
}


def find_checkpoint(variant: str, path: str | Path | None = None) -> Path:
    candidates = [
        path,
        os.environ.get(CHECKPOINT_ENV),
        Path(torch.hub.get_dir()) / "checkpoints" / f"{variant}.pth",
        Path.home() / "work" / "ALIKED" / "models" / f"{variant}.pth",  # sibling checkout of the reference repo
    ]
    for candidate in candidates:
        if candidate and Path(candidate).is_file():
            return Path(candidate)
    raise FileNotFoundError(f"ALIKED checkpoint not found for {variant!r}. {CHECKPOINT_HELP}")


def load_state_dict(
    variant: str, path: str | Path | None = None, map_location: str | torch.device = "cpu"
) -> dict[str, torch.Tensor]:
    """The released checkpoints are already a plain ``state_dict`` whose keys match FlashIM's module paths."""
    return torch.load(find_checkpoint(variant, path), map_location=map_location, weights_only=True)
