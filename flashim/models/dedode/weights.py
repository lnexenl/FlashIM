"""DeDoDe checkpoints (Parskatt/DeDoDe releases): detector-L v2, descriptor B / G, and DINOv2 ViT-L/14 for G."""

from __future__ import annotations

from pathlib import Path

import torch

from flashim.models.weights_util import find_weights

RELEASES = "https://github.com/Parskatt/DeDoDe/releases/download"
FILES = {
    "detector-L": ("dedode_detector_L_v2.pth", f"{RELEASES}/v2/dedode_detector_L_v2.pth"),
    "descriptor-B": ("dedode_descriptor_B.pth", f"{RELEASES}/dedode_pretrained_models/dedode_descriptor_B.pth"),
    "descriptor-G": ("dedode_descriptor_G.pth", f"{RELEASES}/dedode_pretrained_models/dedode_descriptor_G.pth"),
    "dinov2": ("dinov2_vitl14_pretrain.pth", "https://dl.fbaipublicfiles.com/dinov2/dinov2_vitl14/dinov2_vitl14_pretrain.pth"),
}
LICENSES = {
    "code": "MIT (Parskatt/DeDoDe); DINOv2 Apache-2.0; FlashIM re-implementation",
    "weights": "DeDoDe v2 detector / descriptors, DINOv2 ViT-L/14 (frozen, DeDoDe-G)",
}


def load(name: str, path: str | Path | None = None, map_location="cpu") -> dict[str, torch.Tensor]:
    file, url = FILES[name]
    return torch.load(
        find_weights(file, path=path, env=f"FLASHIM_DEDODE_{name.replace('-', '_').upper()}", help=f"Download {url}"),
        map_location=map_location, weights_only=True,
    )
