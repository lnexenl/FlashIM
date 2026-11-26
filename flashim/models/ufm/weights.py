"""UFM checkpoint location and loading."""

from __future__ import annotations

from pathlib import Path

import torch
from safetensors.torch import load_file

from flashim.models.weights_util import Checkpoint, find_weights

CHECKPOINT = Checkpoint(
    "ufm_base.safetensors",
    "https://huggingface.co/infinity1096/UFM-Base/resolve/main/model.safetensors",
    "b11d1c27bb39edd59b95df365f1754d1500af14fdf40940f4826436a4a55fce6",
)

LICENSES = {
    "code": "BSD-3-Clause (UniFlowMatch/UFM and castacks/UniCeption; DPT blocks from CroCo/DUSt3R CC BY-NC-SA 4.0; "
    "DINOv2 Apache-2.0); FlashIM re-implementation",
    "weights": "CC BY-NC-SA 4.0 (non-commercial use only; inherits underlying training dataset licenses)",
}


def load_state_dict(
    checkpoint: str | Path | None = None,
    map_location: str | torch.device = "cpu",
) -> dict[str, torch.Tensor]:
    """``ufm_base.safetensors`` from ``checkpoint``, ``$FLASHIM_UFM_CHECKPOINT``, else the torch hub
    checkpoint folder, downloaded on first use."""
    file = find_weights(CHECKPOINT, path=checkpoint, env="FLASHIM_UFM_CHECKPOINT")
    return load_file(str(file), device=str(map_location))
