"""RoMa (v1) checkpoints (Parskatt/storage ``roma`` release) + DINOv2 ViT-L/14 (shared with DeDoDe)."""

from __future__ import annotations

from pathlib import Path

import torch

from flashim.models.dedode.weights import CHECKPOINTS as _DEDODE_CHECKPOINTS
from flashim.models.weights_util import Checkpoint, ModelInfo, find_weights

RELEASES = "https://github.com/Parskatt/storage/releases/download/roma"
MINIMA_RELEASES = "https://github.com/LSXI7/storage/releases/download/MINIMA"

CHECKPOINTS = {
    "outdoor": Checkpoint(
        "roma_outdoor.pth",
        f"{RELEASES}/roma_outdoor.pth",
        "c7a45c80d41ad788a63c641d1b686d7cb3f297f40097c6f4e75039889e5cc8ba",
    ),
    "indoor": Checkpoint(
        "roma_indoor.pth",
        f"{RELEASES}/roma_indoor.pth",
        "4d3dca889ae1ef245123dc62aab914475c7bbf41f2c8002606450fb6cf2d91e6",
    ),
    "minima": Checkpoint(
        "minima_roma.pth",
        f"{MINIMA_RELEASES}/minima_roma.pth",
        "17f3923bd780e8f1450792706e0643f70c8864ad11bf5fd4f2b54714bac23538",
    ),
}
DINOV2 = _DEDODE_CHECKPOINTS["dinov2"]

LICENSES = {
    "code": "MIT (Parskatt/RoMa); DINOv2 Apache-2.0; FlashIM re-implementation",
    "weights": "RoMa outdoor / indoor (MIT), MINIMA-RoMa (Apache-2.0, LSXI7/MINIMA), DINOv2 ViT-L/14 (Apache-2.0, frozen)",
}
MODEL_INFO = ModelInfo(("Apache-2.0", "MIT"), ("Apache-2.0", "MIT"))
MODEL_INFO_OVERRIDES = {
    "minima": ModelInfo(("Apache-2.0", "MIT"), ("Apache-2.0",)),
}


def load_state_dict(
    variant: str = "outdoor", path: str | Path | None = None, map_location: str | torch.device = "cpu"
) -> dict[str, torch.Tensor]:
    """``roma_{outdoor,indoor}.pth`` or ``minima_roma.pth`` from ``checkpoint`` / ``$FLASHIM_ROMA_CHECKPOINT`` or hub."""
    if variant not in CHECKPOINTS:
        raise ValueError(f"Unknown RoMa variant {variant!r}; choose from {sorted(CHECKPOINTS)}")
    file = find_weights(CHECKPOINTS[variant], path=path, env="FLASHIM_ROMA_CHECKPOINT")
    return torch.load(file, map_location=map_location, weights_only=True)


def load_dinov2(path: str | Path | None = None, map_location: str | torch.device = "cpu") -> dict[str, torch.Tensor]:
    """Frozen DINOv2 ViT-L/14 weights (shared with DeDoDe-G / LoMa)."""
    file = find_weights(DINOV2, path=path, env="FLASHIM_DEDODE_DINOV2")
    return torch.load(file, map_location=map_location, weights_only=True)
