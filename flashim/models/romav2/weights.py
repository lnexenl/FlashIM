"""RoMaV2 checkpoint location and loading."""

from __future__ import annotations

import torch

CHECKPOINT_URL = "https://github.com/Parskatt/RoMaV2/releases/download/v2.0.1/romav2.0.1.pt"
CHECKPOINT_SHA256_PREFIX = None  # the release does not publish a hash; torch.hub caches by filename

LICENSES = {
    "code": "MIT (RoMaV2); FlashIM re-implementation",
    "weights": "RoMaV2 checkpoint includes DINOv3 ViT-L/16 weights under the DINOv3 License",
}


def load_state_dict(map_location: str | torch.device = "cpu") -> dict[str, torch.Tensor]:
    """Download (or reuse the torch.hub cache of) ``romav2.0.1.pt``."""
    return torch.hub.load_state_dict_from_url(CHECKPOINT_URL, map_location=map_location, weights_only=True)
