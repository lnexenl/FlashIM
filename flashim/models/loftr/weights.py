"""LoFTR checkpoints (the kornia mirror of the official ``outdoor_ds`` / ``indoor_ds_new`` weights; the ``state_dict`` of a
Lightning checkpoint with a ``matcher.`` prefix)."""

from __future__ import annotations

from pathlib import Path

import torch

from flashim.models.weights_util import Checkpoint, find_weights

_MIRROR = "https://cmp.felk.cvut.cz/~mishkdmy/models"
CHECKPOINTS = {
    "outdoor": Checkpoint(
        "loftr_outdoor.ckpt", f"{_MIRROR}/loftr_outdoor.ckpt",
        "21f5bec5968178e8bc8b7633441836fe5de4f47d861dd2cd7dc38e271b0479ec",
    ),
    "indoor": Checkpoint(
        "loftr_indoor.ckpt", f"{_MIRROR}/loftr_indoor_ds_new.ckpt",
        "be9ff88b323ec27889114719f668ae41aff7034b56a4c4acbd46b8b180b87ed3",
    ),
}
LICENSES = {
    "code": "Apache-2.0 (LoFTR, zju3dv/LoFTR; kornia); FlashIM re-implementation",
    "weights": "LoFTR outdoor_ds / indoor_ds_new",
}


def load_state_dict(variant: str, path: str | Path | None = None, map_location="cpu") -> dict[str, torch.Tensor]:
    file = find_weights(CHECKPOINTS[variant], path=path, env="FLASHIM_LOFTR_CHECKPOINT")
    ckpt = torch.load(file, map_location=map_location, weights_only=True)
    return {k.removeprefix("matcher."): v for k, v in ckpt["state_dict"].items()}
