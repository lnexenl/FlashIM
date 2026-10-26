"""ALIKE checkpoints (``models/alike-{t,s,n,l}.pth`` of Shiaoming/ALIKE; keys match the module paths)."""

from __future__ import annotations

from pathlib import Path

import torch

from flashim.models.weights_util import Checkpoint, find_weights

_SHA256 = {
    "alike-t": "c0840329a6b88518d914b03af2be956f5607055a389ba17441db02bb94f7d12e",
    "alike-s": "a9c0789ff0a09f576cc24afe4924d3233471499d1ce3b0248d650c8794e99a94",
    "alike-n": "8bd4789272eec779be280f8fc1007608ff604241440a0a3377c1559199412ee3",
    "alike-l": "bed5fbbf352ab1c3e92e2241881f8b84edce949984fa23bc7f2517eab93938a0",
}
_REPO = "https://github.com/Shiaoming/ALIKE/raw/main/models"
CHECKPOINTS = {name: Checkpoint(f"{name}.pth", f"{_REPO}/{name}.pth", sha) for name, sha in _SHA256.items()}
LICENSES = {"code": "BSD-3-Clause (Shiaoming/ALIKE); FlashIM re-implementation", "weights": "ALIKE release checkpoints"}


def load_state_dict(variant: str, path: str | Path | None = None, map_location="cpu") -> dict[str, torch.Tensor]:
    file = find_weights(CHECKPOINTS[variant], path=path, env="FLASHIM_ALIKE_CHECKPOINT")
    return torch.load(file, map_location=map_location, weights_only=True)
