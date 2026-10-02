"""DISK checkpoint: cvlab-epfl/disk's ``depth-save.pth`` (the ``extractor`` state dict; keys match the module paths)."""

from __future__ import annotations

from pathlib import Path

import torch

from flashim.models.weights_util import find_weights

HELP = "Download https://raw.githubusercontent.com/cvlab-epfl/disk/master/depth-save.pth (saved as disk_depth.pth)"
LICENSES = {"code": "Apache-2.0 (DISK, cvlab-epfl/disk; kornia); FlashIM re-implementation", "weights": "DISK 'depth' checkpoint"}


def load_state_dict(path: str | Path | None = None, map_location="cpu") -> dict[str, torch.Tensor]:
    file = find_weights("disk_depth.pth", path=path, env="FLASHIM_DISK_CHECKPOINT", help=HELP)
    return torch.load(file, map_location=map_location, weights_only=True)["extractor"]
