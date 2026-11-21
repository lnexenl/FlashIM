"""SuperGlue checkpoint locations, SHA-256 pins and weight layout conversion."""

from __future__ import annotations

import re
from pathlib import Path

import torch

from flashim.models.weights_util import Checkpoint, find_weights

VARIANTS = ("outdoor", "indoor")
DEFAULT_VARIANT = "outdoor"

_COMMIT = "ddcf11f42e7e0732a0c4607648f9448ea8d73590"
_BASE_URL = f"https://raw.githubusercontent.com/magicleap/SuperGluePretrainedNetwork/{_COMMIT}/models/weights"

CHECKPOINTS = {
    "outdoor": Checkpoint(
        "superglue_outdoor.pth",
        f"{_BASE_URL}/superglue_outdoor.pth",
        "2f5f5e9bb3febf07b69df633c4c3ff7a17f8af26a023aae2b9303d22339195bd",
    ),
    "indoor": Checkpoint(
        "superglue_indoor.pth",
        f"{_BASE_URL}/superglue_indoor.pth",
        "0e710469be25ebe1e2ccf68edcae8b2945b0617c8e7e68412251d9d47f5052b1",
    ),
}

LICENSES = {
    "code": "FlashIM re-implementation of SuperGlue (Sarlin et al., CVPR 2020); no upstream code copied "
    "(upstream SuperGluePretrainedNetwork carries a Magic Leap no-redistribution notice)",
    "weights": "superglue_outdoor.pth / superglue_indoor.pth (Magic Leap, Inc.; Academic or non-profit "
    "organization non-commercial research use only)",
}

_PROJ_RE = re.compile(r"^gnn\.layers\.\d+\.attn\.proj\.[012]\.(weight|bias)$")
_MERGE_W_RE = re.compile(r"^gnn\.layers\.\d+\.attn\.merge\.weight$")


def convert_state_dict(state: dict[str, torch.Tensor], num_heads: int = 4) -> dict[str, torch.Tensor]:
    """Upstream ``Conv1d(kernel_size=1)`` layout -> FlashIM ``nn.Linear`` + head-major SDPA layout.

    - Every 1x1 ``Conv1d`` weight ``[out, in, 1]`` becomes ``[out, in]`` for ``nn.Linear`` (module paths unchanged).
    - Upstream ``MultiHeadedAttention`` packs ``[head_dim, num_heads]`` along the channel axis, which gives stride
      ``num_heads`` along ``head_dim`` in ``[B, H, N, D]``. Reordering ``proj.{0,1,2}`` output rows and ``merge``
      input columns to ``[num_heads, head_dim]`` keeps the layer function identical while leaving ``head_dim``
      contiguous (stride 1) for Flash / cuDNN SDPA.
    """
    out: dict[str, torch.Tensor] = {}
    for key, value in state.items():
        if value.ndim == 3 and value.shape[-1] == 1 and key.endswith(".weight"):
            value = value.squeeze(-1)
        if _PROJ_RE.match(key):
            head_dim = value.shape[0] // num_heads
            shaped = value.reshape(head_dim, num_heads, *value.shape[1:])
            value = shaped.transpose(0, 1).reshape(value.shape).contiguous()
        elif _MERGE_W_RE.match(key):
            head_dim = value.shape[1] // num_heads
            shaped = value.reshape(value.shape[0], head_dim, num_heads)
            value = shaped.transpose(1, 2).reshape(value.shape).contiguous()
        out[key] = value
    return out


def find_checkpoint(variant: str = DEFAULT_VARIANT, path: str | Path | None = None) -> Path:
    if variant not in CHECKPOINTS:
        raise ValueError(f"Unknown SuperGlue variant {variant!r}; choose from {VARIANTS}")
    env = f"FLASHIM_SUPERGLUE_{variant.upper()}_CHECKPOINT"
    return find_weights(CHECKPOINTS[variant], path=path, env=env)


def load_state_dict(
    variant: str = DEFAULT_VARIANT,
    num_heads: int = 4,
    path: str | Path | None = None,
    map_location: str | torch.device = "cpu",
) -> dict[str, torch.Tensor]:
    state = torch.load(find_checkpoint(variant, path), map_location=map_location, weights_only=True)
    return convert_state_dict(state, num_heads=num_heads)
