"""DeDoDe checkpoints (Parskatt/DeDoDe releases): detector-L v2, descriptor B / G, and DINOv2 ViT-L/14 for G."""

from __future__ import annotations

from pathlib import Path

import torch

from flashim.models.weights_util import Checkpoint, find_weights

RELEASES = "https://github.com/Parskatt/DeDoDe/releases/download"
CHECKPOINTS = {
    "detector-L": Checkpoint(
        "dedode_detector_L_v2.pth", f"{RELEASES}/v2/dedode_detector_L_v2.pth",
        "4113809dd9e0367af013a45fc2255a6b243ff241cd06520d17a65d9e231bdc17",
    ),
    "descriptor-B": Checkpoint(
        "dedode_descriptor_B.pth", f"{RELEASES}/dedode_pretrained_models/dedode_descriptor_B.pth",
        "8eccfc270ec990ced60cd54a434411a47c4c504de13586f596d042e005b3022b",
    ),
    "descriptor-G": Checkpoint(
        "dedode_descriptor_G.pth", f"{RELEASES}/dedode_pretrained_models/dedode_descriptor_G.pth",
        "ef6e3f2911bb3c179960db15545a2137d0746054bb5bad75559524ccab1fee41",
    ),
    "dinov2": Checkpoint(
        "dinov2_vitl14_pretrain.pth",
        "https://dl.fbaipublicfiles.com/dinov2/dinov2_vitl14/dinov2_vitl14_pretrain.pth",
        "d5383ea8f4877b2472eb973e0fd72d557c7da5d3611bd527ceeb1d7162cbf428",
    ),
}
LICENSES = {
    "code": "MIT (Parskatt/DeDoDe); DINOv2 Apache-2.0; FlashIM re-implementation",
    "weights": "DeDoDe v2 detector / descriptors, DINOv2 ViT-L/14 (frozen, DeDoDe-G)",
}


def load(name: str, path: str | Path | None = None, map_location="cpu") -> dict[str, torch.Tensor]:
    file = find_weights(CHECKPOINTS[name], path=path, env=f"FLASHIM_DEDODE_{name.replace('-', '_').upper()}")
    return torch.load(file, map_location=map_location, weights_only=True)
