"""RIPE checkpoint (``ripe_weights.pth``; the fraunhofer.de host is not always reachable, the same file ships as
``weights_ripe.pth`` in the ``JohannesK14/RIPE`` Hugging Face space, which is where it is downloaded from)."""

from __future__ import annotations

from pathlib import Path

import torch

from flashim.models.weights_util import (
    FRAUNHOFER_ACADEMIC,
    Checkpoint,
    ModelInfo,
    RestrictedWeights,
    find_weights,
)

RESTRICTION = RestrictedWeights("ripe", FRAUNHOFER_ACADEMIC)

CHECKPOINT = Checkpoint(
    "ripe_weights.pth",
    "https://huggingface.co/spaces/JohannesK14/RIPE/resolve/main/weights_ripe.pth",
    "1931b338dc21f9a84d2b030c77fe2334073f5622bb47464d881333a7dfd42d43",
    noncommercial=RESTRICTION,
)
LICENSES = {
    "code": "RIPE (fraunhoferhhi/RIPE) research license; FlashIM re-implementation",
    "weights": "ripe_weights.pth (Fraunhofer Academic Use License v2.0, non-commercial use only)",
}
MODEL_INFO = ModelInfo((FRAUNHOFER_ACADEMIC,), (FRAUNHOFER_ACADEMIC,))


def load_state_dict(path: str | Path | None = None, map_location="cpu") -> dict[str, torch.Tensor]:
    file = find_weights(CHECKPOINT, path=path, env="FLASHIM_RIPE_CHECKPOINT")
    return torch.load(file, map_location=map_location, weights_only=True)
