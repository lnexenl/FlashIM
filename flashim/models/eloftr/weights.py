"""EfficientLoFTR weights (vismatch safetensors mirror of upstream ``eloftr_outdoor.ckpt``, verified bit-identical across
all 447 tensors; upstream's Lightning ``.ckpt`` contains pickled ``ModelCheckpoint`` objects that fail
``torch.load(..., weights_only=True)``)."""

from __future__ import annotations

from pathlib import Path

import torch
from safetensors.torch import load_file

from flashim.models.weights_util import Checkpoint, ModelInfo, find_weights

CHECKPOINT = Checkpoint(
    "eloftr_outdoors.safetensors",
    "https://huggingface.co/vismatch/eloftr/resolve/d4b1a3c12ee4f5a55a7d4cf8a7e66c8456a7f93c/eloftr_outdoors.safetensors",
    "40a6941d13de760bb51de1360d075c2b28264c5415e9b7d3bbfd67d8c7cbdc9f",
)
LICENSES = {
    "code": (
        "Apache-2.0 (EfficientLoFTR, zju3dv/EfficientLoFTR @ ffd4a46440; "
        "upstream re-licensed to ZJU3DV Project Registration License v1.0 on 2026-09-15, commit 07e9c1401e); "
        "FlashIM re-implementation"
    ),
    "weights": (
        "Apache-2.0 (vismatch/eloftr mirror of eloftr_outdoor.ckpt; "
        "upstream zju3dv/EfficientLoFTR re-licensed to ZJU3DV PRL v1.0 on 2026-09-15, so newer upstream use is under PRL)"
    ),
}
MODEL_INFO = ModelInfo(("Apache-2.0",), ("Apache-2.0",))


def load_state_dict(path: str | Path | None = None, map_location: str = "cpu") -> dict[str, torch.Tensor]:
    file = find_weights(CHECKPOINT, path=path, env="FLASHIM_ELOFTR_CHECKPOINT")
    if file.suffix == ".safetensors":
        raw = load_file(str(file), device=str(map_location))
    else:
        ckpt = torch.load(file, map_location=map_location, weights_only=True)
        raw = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt
    return {k.removeprefix("matcher."): v for k, v in raw.items()}
