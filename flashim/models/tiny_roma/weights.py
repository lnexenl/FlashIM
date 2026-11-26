"""Tiny-RoMa checkpoint location and loading (``tiny_roma_v1_outdoor.pth`` from Parskatt/RoMa).

The release checkpoint already contains all 112 tensors: the fine-tuned XFeat backbone (``xfeat.0.*``, 76 tensors),
``coarse_matcher.*`` (18 tensors) and ``fine_matcher.*`` (18 tensors), so ``xfeat.pt`` is not needed.
"""

from __future__ import annotations

from pathlib import Path

import torch

from flashim.models.weights_util import Checkpoint, find_weights

CHECKPOINT = Checkpoint(
    "tiny_roma_v1_outdoor.pth",
    "https://github.com/Parskatt/storage/releases/download/roma/tiny_roma_v1_outdoor.pth",
    "3e2ac3481f754243eb3f5c8ea0358667561f1ce139c3edd03d7d21927ffba30c",
)

LICENSES = {
    "code": "MIT (Parskatt/RoMa); XFeat backbone Apache-2.0 (verlab/accelerated_features); FlashIM re-implementation",
    "weights": "MIT (Parskatt/RoMa release; fine-tuned XFeat backbone weights included)",
}


def load_state_dict(
    path: str | Path | None = None,
    map_location: str | torch.device = "cpu",
) -> dict[str, torch.Tensor]:
    """``tiny_roma_v1_outdoor.pth`` from ``path`` / ``$FLASHIM_TINY_ROMA_CHECKPOINT`` / torch hub cache,
    downloaded and SHA-256-verified on first use."""
    file = find_weights(CHECKPOINT, path=path, env="FLASHIM_TINY_ROMA_CHECKPOINT")
    return torch.load(file, map_location=map_location, weights_only=True)
