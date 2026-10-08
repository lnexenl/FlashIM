"""ALIKED checkpoint location and loading (module names match the released checkpoints one-to-one)."""

from __future__ import annotations

from pathlib import Path

import torch

from flashim.models.weights_util import Checkpoint, find_weights

CHECKPOINT_ENV = "FLASHIM_ALIKED_CHECKPOINT"
_SHA256 = {
    "aliked-t16": "56782336919db220144a884e710e2b6af13b3b8f8da15b915a239078ab471085",
    "aliked-n16": "5be8704840ed662d9d8c561bf7279c222092674e7eb05fd0feab94899e9d82f2",
    "aliked-n16rot": "ddf3abbf38e86f6a74540d214e1a9712c54b2d8551abc864542199f2347d7332",
    "aliked-n32": "055283fd1624d3ab46a05c3b617b1a79a3f28d1f551737eef2d653e8c8232890",
}
_REPO = "https://github.com/Shiaoming/ALIKED/raw/main/models"
CHECKPOINTS = {name: Checkpoint(f"{name}.pth", f"{_REPO}/{name}.pth", sha) for name, sha in _SHA256.items()}

LICENSES = {
    "code": "BSD-3-Clause (ALIKED, Shiaoming/ALIKED); FlashIM re-implementation",
    "weights": "ALIKED release checkpoints (aliked-t16 / aliked-n16 / aliked-n16rot / aliked-n32)",
}


def find_checkpoint(variant: str, path: str | Path | None = None) -> Path:
    return find_weights(CHECKPOINTS[variant], path=path, env=CHECKPOINT_ENV)


def load_state_dict(
    variant: str, path: str | Path | None = None, map_location: str | torch.device = "cpu"
) -> dict[str, torch.Tensor]:
    """The released checkpoints are already a plain ``state_dict`` whose keys match FlashIM's module paths."""
    return torch.load(find_checkpoint(variant, path), map_location=map_location, weights_only=True)
