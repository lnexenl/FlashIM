"""XoFTR checkpoint location and loading (``weights_xoftr_640.ckpt`` from the upstream release folder; Lightning
checkpoint whose ``state_dict`` carries a ``matcher.`` prefix)."""

from __future__ import annotations

from pathlib import Path

import torch

from flashim.models.weights_util import Checkpoint, ModelInfo, find_weights

CHECKPOINT_ENV = "FLASHIM_XOFTR_CHECKPOINT"
# ``weights_xoftr_640.ckpt`` in the Google Drive folder linked from https://github.com/OnderT/XoFTR (its only
# release); ``confirm=t`` skips Drive's large-file warning page, the SHA-256 pins the content.
CHECKPOINT = Checkpoint(
    "weights_xoftr_640.ckpt",
    "https://drive.usercontent.google.com/download?id=1oRkEGsLpPIxlulc6a7c2q5H1XTNbfkVj"
    "&export=download&confirm=t",
    "6e7f24e553f76746ef49061bf44c7d11252e56057470d1421bcd4e6506af3a67",
)

LICENSES = {
    "code": "Apache-2.0 (XoFTR, OnderT/XoFTR, derived from zju3dv/LoFTR); FlashIM re-implementation",
    "weights": "XoFTR weights_xoftr_640.ckpt (OnderT/XoFTR, Apache-2.0)",
}
MODEL_INFO = ModelInfo(("Apache-2.0",), ("Apache-2.0",))


def find_checkpoint(path: str | Path | None = None) -> Path:
    return find_weights(CHECKPOINT, path=path, env=CHECKPOINT_ENV)


def load_state_dict(
    path: str | Path | None = None, map_location: str | torch.device = "cpu"
) -> dict[str, torch.Tensor]:
    ckpt = torch.load(find_checkpoint(path), map_location=map_location, weights_only=True)
    state = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt
    return {k.removeprefix("matcher."): v for k, v in state.items()}
