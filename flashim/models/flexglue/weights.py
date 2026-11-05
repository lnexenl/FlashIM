"""FlexGlue checkpoint location and conversion."""

from __future__ import annotations

from pathlib import Path

import torch
from torch import Tensor

from flashim.models.weights_util import Checkpoint, find_weights

CHECKPOINT_ENV = "FLASHIM_FLEXGLUE_CHECKPOINT"
# cvg/FlexGlue's v0.1 release (the matcher and its DINOv3-ConvNeXt descriptor in one file).
CHECKPOINT = Checkpoint(
    "flexglue.pth",
    "https://github.com/cvg/FlexGlue/releases/download/v0.1/flexglue.pth",
    "d7371f743b8cdc4adf723c2aa05b181587fe4bc177fced139697975dff790068",
)

LICENSES = {
    "code": "MIT (cvg/FlexGlue); FlashIM re-implementation",
    "weights": "cvg/FlexGlue release flexglue.pth: the DINOv3 ConvNeXt-B backbone (descriptor.backbone.*) under the "
    "DINOv3 License; the matcher and descriptor heads MIT",
}

# Upstream's per-layer correspondence head names its covisibility logit layer ``disambiguate``.
_RENAMED = ((".disambiguate.", ".covisibility."),)


def convert_state_dict(state: dict[str, Tensor]) -> dict[str, Tensor]:
    """Upstream layout -> FlashIM layout.

    ``matcher.*`` loses its prefix (``posenc.Wr`` becomes ``posenc_wr``); ``extractor.*`` becomes the ``descriptor``
    submodule (``extractor.backbone`` -> ``descriptor.backbone``); the head's covisibility logit layer ``disambiguate``
    is renamed ``covisibility``.
    """
    out = {}
    for key, value in state.items():
        if key.startswith("matcher."):
            key = key[len("matcher."):]
            if key.startswith("posenc.Wr."):
                key = "posenc_wr." + key.split(".", 2)[2]
        elif key.startswith("extractor."):
            key = "descriptor." + key[len("extractor."):]
        for old, new in _RENAMED:
            key = key.replace(old, new)
        out[key] = value
    return out


def truncate(state: dict[str, Tensor], num_layers: int) -> dict[str, Tensor]:
    """Drop the transformer layers past ``num_layers`` and their heads (upstream's truncated ``n_layers``)."""
    def keep(key: str) -> bool:
        parts = key.split(".")
        limit = num_layers + (parts[0] == "heads")  # heads[k] reads the tokens after k layers
        return parts[0] not in ("transformers", "heads") or int(parts[1]) < limit

    return {k: v for k, v in state.items() if keep(k)}


def find_checkpoint(path: str | Path | None = None) -> Path:
    return find_weights(CHECKPOINT, path=path, env=CHECKPOINT_ENV)


def load_state_dict(path: str | Path | None = None, map_location: str | torch.device = "cpu") -> dict[str, Tensor]:
    state = torch.load(find_checkpoint(path), map_location=map_location, weights_only=True)
    return convert_state_dict(state)
