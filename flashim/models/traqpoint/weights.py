"""TraqPoint checkpoint location and loading."""

from __future__ import annotations

import re
from pathlib import Path

import torch

from flashim.models.weights_util import Checkpoint, find_weights

CHECKPOINT_ENV = "FLASHIM_TRAQPOINT_CHECKPOINT"
# Pinned Hugging Face commit of https://huggingface.co/lihao20888/traqpoint (linked from
# https://github.com/xiaomi-research/traqpoint).
CHECKPOINT = Checkpoint(
    "traqpoint_best.pth",
    "https://huggingface.co/lihao20888/traqpoint/resolve/ce86f7baf8e7bb7e2386f75f40d9351cb5b9ae47/traqpoint_best.pth",
    "48ac1a3c07f13b2be27babbde9ee0489208287d92139576667e4d4b443134b4b",
)

LICENSES = {
    "code": "Apache-2.0 (TraqPoint, xiaomi-research/traqpoint); FlashIM re-implementation",
    "weights": "TraqPoint release traqpoint_best.pth (Apache-2.0); DINOv3 ConvNeXt-B backbone "
    "(descriptor.backbone.0.dino.*) fine-tuned from Meta's dinov3_convnext_base under the DINOv3 License",
}

# Upstream module paths -> FlashIM paths (flatten wrapper prefixes `descriptor.` / `backbone.0.dino.`).
_RENAMES = (
    (re.compile(r"^descriptor\.backbone\.0\.dino\."), "backbone."),
    (re.compile(r"^descriptor\."), ""),
)

# `dino.norms.3` is an exact alias of `dino.norm` in DINOv3 ConvNeXt.
_DROPPED = frozenset({"descriptor.backbone.0.dino.norms.3.weight", "descriptor.backbone.0.dino.norms.3.bias"})


def convert_state_dict(state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    out = {}
    for key, value in state.items():
        if key in _DROPPED:
            continue
        for pattern, replacement in _RENAMES:
            new = pattern.sub(replacement, key)
            if new != key:
                key = new
                break
        out[key] = value
    return out


def find_checkpoint(path: str | Path | None = None) -> Path:
    return find_weights(CHECKPOINT, path=path, env=CHECKPOINT_ENV)


def load_state_dict(path: str | Path | None = None, map_location: str | torch.device = "cpu") -> dict[str, torch.Tensor]:
    state = torch.load(find_checkpoint(path), map_location=map_location, weights_only=True)
    for wrapper in ("model", "state_dict"):
        if wrapper in state and isinstance(state[wrapper], dict):
            state = state[wrapper]
    return convert_state_dict(state)
