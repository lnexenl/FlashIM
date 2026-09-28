"""RDD checkpoint location and loading."""

from __future__ import annotations

import os
import re
from pathlib import Path

import torch

CHECKPOINT_ENV = "FLASHIM_RDD_CHECKPOINT"
CHECKPOINT_HELP = (
    "Download rdd.pth from the Google Drive folder linked in https://github.com/xtcpete/rdd (README), then pass "
    f"checkpoint=..., set {CHECKPOINT_ENV}, or place it at <torch hub dir>/checkpoints/rdd.pth."
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
    candidates = [path, os.environ.get(CHECKPOINT_ENV), Path(torch.hub.get_dir()) / "checkpoints" / "rdd.pth"]
    candidates.append(Path.home() / "work" / "RDD" / "weights" / "rdd.pth")  # sibling checkout of the reference repo
    for candidate in candidates:
        if candidate and Path(candidate).is_file():
            return Path(candidate)
    raise FileNotFoundError(f"RDD checkpoint not found. {CHECKPOINT_HELP}")


def load_state_dict(path: str | Path | None = None, map_location: str | torch.device = "cpu") -> dict[str, torch.Tensor]:
    state = torch.load(find_checkpoint(path), map_location=map_location, weights_only=True)
    for wrapper in ("model", "state_dict"):
        if wrapper in state and isinstance(state[wrapper], dict):
            state = state[wrapper]
    return convert_state_dict(state)
