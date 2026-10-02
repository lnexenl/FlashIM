"""LightGlue checkpoint location and conversion."""

from __future__ import annotations

import os
import re
from pathlib import Path

import torch

CHECKPOINT_ENV = "FLASHIM_LIGHTGLUE_CHECKPOINT"
ALIKED_CHECKPOINT_ENV = "FLASHIM_ALIKED_LIGHTGLUE_CHECKPOINT"
CHECKPOINT_HELP = (
    "Download rdd_lg.pth from the Google Drive folder linked in https://github.com/xtcpete/rdd (README), then pass "
    f"checkpoint=..., set {CHECKPOINT_ENV}, or place it at <torch hub dir>/checkpoints/rdd_lg.pth."
)
ALIKED_CHECKPOINT_HELP = (
    "Download raco_aliked_lightglue.pth from "
    "https://github.com/cvg/LightGlue/releases/download/v0.1_arxiv/raco_aliked_lightglue.pth, then pass "
    f"checkpoint=..., set {ALIKED_CHECKPOINT_ENV}, or place it at "
    "<torch hub dir>/checkpoints/raco_aliked_lightglue.pth."
)

LICENSES = {
    "code": "Apache-2.0 (LightGlue, cvg/LightGlue; RDD, xtcpete/rdd); FlashIM re-implementation",
    "weights": "RDD release checkpoint rdd_lg.pth (LightGlue trained on RDD descriptors), or cvg/LightGlue's "
    "raco_aliked_lightglue.pth (LightGlue trained on ALIKED descriptors, 128 -> 256 input projection)",
}

_UNUSED = re.compile(r"^confidence_thresholds|^log_assignment\.\d+\.r$")
_OLD_LAYOUT = re.compile(r"^(self_attn|cross_attn)\.(\d+)\.")  # cvg releases before ``transformers.N.*``


def convert_state_dict(state: dict[str, torch.Tensor], num_heads: int, num_layers: int) -> dict[str, torch.Tensor]:
    """Upstream layout -> FlashIM layout.

    - The ``confidence_thresholds`` buffer (and the stray ``log_assignment.N.r`` of the cvg releases) is dropped
      (recomputed); token confidences and per-layer assignments are kept for early stopping. Pre-``transformers.N``
      key layouts (``self_attn.N.*``, ``cross_attn.N.*``) are renamed.
    - ``Wqkv`` output channels are reordered from upstream's interleaved ``[head, dim, (q, k, v)]`` to
      ``[(q, k, v), head, dim]`` so q, k and v are contiguous slices.
    """
    out = {}
    for key, value in state.items():
        if _UNUSED.match(key):
            continue
        key = _OLD_LAYOUT.sub(r"transformers.\2.\1.", key)
        if ".Wqkv." in key:
            rows = value.shape[0]
            head_dim = rows // (3 * num_heads)
            shaped = value.reshape(num_heads, head_dim, 3, *value.shape[1:])
            value = shaped.permute(2, 0, 1, *range(3, shaped.ndim)).reshape(value.shape).contiguous()
        out[key] = value
    return out


def find_checkpoint(path: str | Path | None = None) -> Path:
    candidates = [path, os.environ.get(CHECKPOINT_ENV), Path(torch.hub.get_dir()) / "checkpoints" / "rdd_lg.pth"]
    candidates.append(Path.home() / "work" / "RDD" / "weights" / "rdd_lg.pth")  # sibling checkout of RDD
    for candidate in candidates:
        if candidate and Path(candidate).is_file():
            return Path(candidate)
    raise FileNotFoundError(f"LightGlue checkpoint not found. {CHECKPOINT_HELP}")


def find_aliked_checkpoint(path: str | Path | None = None) -> Path:
    candidates = [
        path,
        os.environ.get(ALIKED_CHECKPOINT_ENV),
        Path(torch.hub.get_dir()) / "checkpoints" / "raco_aliked_lightglue.pth",
    ]
    for candidate in candidates:
        if candidate and Path(candidate).is_file():
            return Path(candidate)
    raise FileNotFoundError(f"ALIKED LightGlue checkpoint not found. {ALIKED_CHECKPOINT_HELP}")


def load_state_dict(
    num_heads: int, num_layers: int, path: str | Path | None = None, map_location: str | torch.device = "cpu"
) -> dict[str, torch.Tensor]:
    state = torch.load(find_checkpoint(path), map_location=map_location, weights_only=True)
    return convert_state_dict(state, num_heads, num_layers)


def load_aliked_state_dict(
    num_heads: int, num_layers: int, path: str | Path | None = None, map_location: str | torch.device = "cpu"
) -> dict[str, torch.Tensor]:
    state = torch.load(find_aliked_checkpoint(path), map_location=map_location, weights_only=True)
    return convert_state_dict(state, num_heads, num_layers)


CVG_FILES = {"superpoint": "superpoint_lightglue.pth", "disk": "disk_lightglue.pth", "sift": "sift_lightglue.pth"}


def load_cvg_state_dict(
    variant: str, num_heads: int, num_layers: int, path: str | Path | None = None, map_location="cpu"
) -> dict[str, torch.Tensor]:
    """cvg/LightGlue's per-extractor release (``<extractor>_lightglue.pth``)."""
    from flashim.models.weights_util import find_weights

    file = find_weights(
        CVG_FILES[variant], path=path, env=f"FLASHIM_{variant.upper()}_LIGHTGLUE_CHECKPOINT",
        help=f"Download https://github.com/cvg/LightGlue/releases/download/v0.1_arxiv/{CVG_FILES[variant]}",
    )
    return convert_state_dict(torch.load(file, map_location=map_location, weights_only=True), num_heads, num_layers)
