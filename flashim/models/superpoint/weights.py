"""SuperPoint checkpoint (``superpoint_v1.pth`` of cvg/LightGlue's release; keys match the module paths)."""

from __future__ import annotations

from pathlib import Path

import torch

from flashim.models.weights_util import Checkpoint, find_weights

CHECKPOINT_ENV = "FLASHIM_SUPERPOINT_CHECKPOINT"
CHECKPOINT = Checkpoint(
    "superpoint_v1.pth",
    "https://github.com/cvg/LightGlue/releases/download/v0.1_arxiv/superpoint_v1.pth",
    "52b6708629640ca883673b5d5c097c4ddad37d8048b33f09c8ca0d69db12c40e",
)

LICENSES = {
    "code": "FlashIM re-implementation of SuperPoint (Magic Leap); no upstream code copied (LightGlue's superpoint.py carries a Magic Leap no-redistribution notice)",
    "weights": "superpoint_v1.pth (Magic Leap; non-commercial research license, see the SuperPoint release)",
}


def load_state_dict(path: str | Path | None = None, map_location="cpu") -> dict[str, torch.Tensor]:
    file = find_weights(CHECKPOINT, path=path, env=CHECKPOINT_ENV)
    return torch.load(file, map_location=map_location, weights_only=True)
