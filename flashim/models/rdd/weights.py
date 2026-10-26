"""RDD checkpoint location and loading."""

from __future__ import annotations

import re
from pathlib import Path

import torch

from flashim.models.weights_util import Checkpoint, find_weights

CHECKPOINT_ENV = "FLASHIM_RDD_CHECKPOINT"
# ``weights/rdd.pth`` of the Google Drive folder linked from https://github.com/xtcpete/rdd (its only
# release); ``confirm=t`` skips Drive's page for files too large to virus-scan, the SHA-256 pins the content.
CHECKPOINT = Checkpoint(
    "rdd.pth",
    "https://drive.usercontent.google.com/download?id=18Xyzwtrk7dqppZzPz01Tpo6cdMBk_h6J"
    "&export=download&confirm=t",
    "b9f72267db638d7daa3a896d2c688d5fe02ea0a2acf39a1d4218305b4153b1d6",
)

LICENSES = {
    "code": "Apache-2.0 (RDD, xtcpete/rdd); FlashIM re-implementation",
    "weights": "RDD release checkpoint; ConvNeXt backbone initialised from Meta's ImageNet-22k weights",
}

# Upstream module paths -> FlashIM paths (the wrapper modules `descriptor.` / `Joiner` / `encoder.` are flattened).
_RENAMES = (
    (re.compile(r"^descriptor\.backbone\.0\.encoder\.model\."), "backbone."),
    (re.compile(r"^descriptor\."), ""),
)


def convert_state_dict(state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    out = {}
    for key, value in state.items():
        for pattern, replacement in _RENAMES:
            new = pattern.sub(replacement, key)
            if new != key:
                key = new
                break
        out[key] = value
    return out


def find_checkpoint(path: str | Path | None = None) -> Path:
    return find_weights(CHECKPOINT, path=path, env=CHECKPOINT_ENV)


def load_state_dict(path: str | Path | None = None, map_location: str | torch.device = "cpu") -> dict[str, torch.Tensor]:
    state = torch.load(find_checkpoint(path), map_location=map_location, weights_only=True)
    for wrapper in ("model", "state_dict"):
        if wrapper in state and isinstance(state[wrapper], dict):
            state = state[wrapper]
    return convert_state_dict(state)
