"""RIPE checkpoint (``ripe_weights.pth``; the fraunhofer.de host is not always reachable, the same file ships as
``weights_ripe.pth`` in the ``JohannesK14/RIPE`` Hugging Face space)."""

from __future__ import annotations

from pathlib import Path

import torch

from flashim.models.weights_util import find_weights

HELP = ("Download https://cvg.hhi.fraunhofer.de/RIPE/ripe_weights.pth or "
        "https://huggingface.co/spaces/JohannesK14/RIPE/resolve/main/weights_ripe.pth (saved as ripe_weights.pth)")
LICENSES = {"code": "RIPE (fraunhoferhhi/RIPE) research license; FlashIM re-implementation", "weights": "ripe_weights.pth (RIPE, ICCV 2025)"}


def load_state_dict(path: str | Path | None = None, map_location="cpu") -> dict[str, torch.Tensor]:
    return torch.load(find_weights("ripe_weights.pth", path=path, env="FLASHIM_RIPE_CHECKPOINT", help=HELP), map_location=map_location, weights_only=True)
