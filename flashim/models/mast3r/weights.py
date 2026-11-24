"""MASt3R checkpoint location and loading."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from flashim.models.weights_util import Checkpoint, find_weights

CHECKPOINT = Checkpoint(
    "MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth",
    "https://download.europe.naverlabs.com/ComputerVision/MASt3R/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth",
    "e28f91b488554653e2b46ddae9c78c1143e0bcb2e27d3e26cdb0b717f1568eb2",
)

LICENSES = {
    "code": "CC BY-NC-SA 4.0 (MASt3R / DUSt3R / CroCo, Naver Corporation; non-commercial use only); FlashIM re-implementation",
    "weights": "CC BY-NC-SA 4.0 (non-commercial use only; subject to training dataset licenses, see upstream CHECKPOINTS_NOTICE)",
}


def load_state_dict(
    path: str | Path | None = None,
    map_location: str | torch.device = "cpu",
) -> dict[str, torch.Tensor]:
    """Load the ``MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth`` state dict from ``path``,
    ``$FLASHIM_MAST3R_CHECKPOINT``, or the torch hub checkpoint folder (downloaded on first use)."""
    file = find_weights(CHECKPOINT, path=path, env="FLASHIM_MAST3R_CHECKPOINT")
    with torch.serialization.safe_globals([argparse.Namespace]):
        ckpt = torch.load(file, map_location=map_location, weights_only=True)
    return ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
