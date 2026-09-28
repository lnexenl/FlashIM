"""RaCo checkpoint location and loading."""

from __future__ import annotations

import os
from pathlib import Path

import torch

CHECKPOINT_ENV = "FLASHIM_RACO_CHECKPOINT"
CHECKPOINT_URL = "https://github.com/cvg/RaCo/releases/download/v1.0.0/raco.pth"
CHECKPOINT_HELP = (
    f"Download raco.pth from {CHECKPOINT_URL}, then pass checkpoint=..., set {CHECKPOINT_ENV}, or place it at "
    "<torch hub dir>/checkpoints/raco.pth."
)
LICENSES = {
    "code": "Apache-2.0 (RaCo, cvg/RaCo); FlashIM re-implementation",
    "weights": "cvg/RaCo release raco.pth",
}
# The released checkpoint also carries a covariance-estimator head that keypoint extraction does not use.
_UNUSED_PREFIX = "covariance_estimator_head."


def find_checkpoint(path: str | Path | None = None) -> Path:
    candidates = [path, os.environ.get(CHECKPOINT_ENV), Path(torch.hub.get_dir()) / "checkpoints" / "raco.pth"]
    for candidate in candidates:
        if candidate and Path(candidate).is_file():
            return Path(candidate)
    raise FileNotFoundError(f"RaCo checkpoint not found. {CHECKPOINT_HELP}")


def load_state_dict(path: str | Path | None = None, map_location: str | torch.device = "cpu") -> dict[str, torch.Tensor]:
    state = torch.load(find_checkpoint(path), map_location=map_location, weights_only=True)
    return {key: value for key, value in state.items() if not key.startswith(_UNUSED_PREFIX)}
