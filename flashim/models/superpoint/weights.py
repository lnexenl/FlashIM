"""SuperPoint checkpoint (``superpoint_v1.pth`` of cvg/LightGlue's release; keys match the module paths)."""

from __future__ import annotations

from pathlib import Path

import torch

from flashim.models.weights_util import find_weights

CHECKPOINT_ENV = "FLASHIM_SUPERPOINT_CHECKPOINT"
HELP = "Download https://github.com/cvg/LightGlue/releases/download/v0.1_arxiv/superpoint_v1.pth"

LICENSES = {
    "code": "MIT-style (Magic Leap SuperPoint) via cvg/LightGlue's port; FlashIM re-implementation",
    "weights": "superpoint_v1.pth (Magic Leap; non-commercial research license, see the SuperPoint release)",
}


def load_state_dict(path: str | Path | None = None, map_location="cpu") -> dict[str, torch.Tensor]:
    return torch.load(find_weights("superpoint_v1.pth", path=path, env=CHECKPOINT_ENV, help=HELP), map_location=map_location, weights_only=True)
