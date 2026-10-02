"""LoFTR checkpoints (the kornia mirror of the official ``outdoor_ds`` / ``indoor_ds_new`` weights; the ``state_dict`` of a
Lightning checkpoint with a ``matcher.`` prefix)."""

from __future__ import annotations

from pathlib import Path

import torch

from flashim.models.weights_util import find_weights

FILES = {
    "outdoor": ("loftr_outdoor.ckpt", "http://cmp.felk.cvut.cz/~mishkdmy/models/loftr_outdoor.ckpt"),
    "indoor": ("loftr_indoor.ckpt", "http://cmp.felk.cvut.cz/~mishkdmy/models/loftr_indoor_ds_new.ckpt"),
}
LICENSES = {"code": "Apache-2.0 (LoFTR, zju3dv/LoFTR; kornia); FlashIM re-implementation", "weights": "LoFTR outdoor_ds / indoor_ds_new"}


def load_state_dict(variant: str, path: str | Path | None = None, map_location="cpu") -> dict[str, torch.Tensor]:
    file, url = FILES[variant]
    ckpt = torch.load(
        find_weights(file, path=path, env="FLASHIM_LOFTR_CHECKPOINT", help=f"Download {url} (saved as {file})"),
        map_location=map_location, weights_only=False,
    )
    return {k.removeprefix("matcher."): v for k, v in ckpt["state_dict"].items()}
