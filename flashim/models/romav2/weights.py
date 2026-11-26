"""RoMaV2 checkpoint location and loading."""

from __future__ import annotations

import torch

from flashim.models.weights_util import Checkpoint, ModelInfo, find_weights

CHECKPOINT = Checkpoint(
    "romav2.0.1.pt",
    "https://github.com/Parskatt/RoMaV2/releases/download/v2.0.1/romav2.0.1.pt",
    "1557dec0d21b62366465f7ff4d5fdf228cc695d0582e196ad2b80e05230828b7",
)

LICENSES = {
    "code": "MIT (RoMaV2); DINOv3 / DPT parts under the DINOv3 License Agreement",
    "weights": "romav2.0.1.pt includes DINOv3 ViT-L/16 under the DINOv3 License. The v2.0.1 README "
    "specifies MIT for code. Its scope for the other tensors in this mixed checkpoint is unresolved "
    "(https://github.com/Parskatt/RoMaV2/issues/52).",
}
MODEL_INFO = ModelInfo(("LicenseRef-DINOv3", "MIT"), ("LicenseRef-DINOv3", "NOASSERTION"))


def load_state_dict(map_location: str | torch.device = "cpu") -> dict[str, torch.Tensor]:
    """``romav2.0.1.pt`` from ``$FLASHIM_ROMAV2_CHECKPOINT``, else the torch hub checkpoint folder (where
    ``torch.hub`` cached it before), downloaded on first use."""
    file = find_weights(CHECKPOINT, env="FLASHIM_ROMAV2_CHECKPOINT")
    return torch.load(file, map_location=map_location, weights_only=True)
