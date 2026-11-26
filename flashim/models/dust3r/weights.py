"""DUSt3R checkpoint location and loading."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from flashim.models.weights_util import Checkpoint, find_weights

CHECKPOINT = Checkpoint(
    "DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth",
    "https://download.europe.naverlabs.com/ComputerVision/DUSt3R/DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth",
    "9f194a32e34fc124ed10f64115b8e848b7da87c2d1ed131d34b4806721c19dbd",
)

LICENSES = {
    "code": "CC BY-NC-SA 4.0 (DUSt3R / CroCo, Naver Corporation; non-commercial use only); FlashIM re-implementation",
    "weights": "CC BY-NC-SA 4.0 (non-commercial use only; subject to training dataset licenses, see upstream NOTICE)",
}


def load_state_dict(
    path: str | Path | None = None,
    map_location: str | torch.device = "cpu",
) -> dict[str, torch.Tensor]:
    """Load the ``DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth`` state dict from ``path``,
    ``$FLASHIM_DUST3R_CHECKPOINT``, or the torch hub checkpoint folder (downloaded on first use)."""
    file = find_weights(CHECKPOINT, path=path, env="FLASHIM_DUST3R_CHECKPOINT")
    with torch.serialization.safe_globals([argparse.Namespace]):
        ckpt = torch.load(file, map_location=map_location, weights_only=True)
    return ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
