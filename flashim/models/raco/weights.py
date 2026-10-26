"""RaCo checkpoint location and loading."""

from __future__ import annotations

from pathlib import Path

import torch

from flashim.models.weights_util import Checkpoint, find_weights

CHECKPOINT_ENV = "FLASHIM_RACO_CHECKPOINT"
CHECKPOINT = Checkpoint(
    "raco.pth",
    "https://github.com/cvg/RaCo/releases/download/v1.0.0/raco.pth",
    "79f0d862bb41abedd8dda33a9b0120f4dcd619c43a90398175bad1a019192e3a",
)
LICENSES = {
    "code": "Apache-2.0 (RaCo, cvg/RaCo); FlashIM re-implementation",
    "weights": "cvg/RaCo release raco.pth",
}
# The released checkpoint also carries a covariance-estimator head that keypoint extraction does not use.
_UNUSED_PREFIX = "covariance_estimator_head."


def find_checkpoint(path: str | Path | None = None) -> Path:
    return find_weights(CHECKPOINT, path=path, env=CHECKPOINT_ENV)


def load_state_dict(path: str | Path | None = None, map_location: str | torch.device = "cpu") -> dict[str, torch.Tensor]:
    state = torch.load(find_checkpoint(path), map_location=map_location, weights_only=True)
    return {key: value for key, value in state.items() if not key.startswith(_UNUSED_PREFIX)}
