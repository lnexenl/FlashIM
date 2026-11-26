"""DISK checkpoint: cvlab-epfl/disk's ``depth-save.pth`` (the ``extractor`` state dict; keys match the module paths)."""

from __future__ import annotations

from pathlib import Path

import torch

from flashim.models.weights_util import Checkpoint, ModelInfo, find_weights

CHECKPOINT = Checkpoint(
    "disk_depth.pth",
    "https://raw.githubusercontent.com/cvlab-epfl/disk/master/depth-save.pth",
    "9c2ee4ded238892dfa51569941372601e35e4a74aa6f84ea80053d2ab1c07abe",
)
LICENSES = {
    "code": "Apache-2.0 (DISK, cvlab-epfl/disk; kornia); FlashIM re-implementation",
    "weights": "depth-save.pth inherits Apache-2.0 from cvlab-epfl/disk's LICENSE.txt",
}
MODEL_INFO = ModelInfo(("Apache-2.0",), ("Apache-2.0",))


def load_state_dict(path: str | Path | None = None, map_location="cpu") -> dict[str, torch.Tensor]:
    file = find_weights(CHECKPOINT, path=path, env="FLASHIM_DISK_CHECKPOINT")
    return torch.load(file, map_location=map_location, weights_only=True)["extractor"]
