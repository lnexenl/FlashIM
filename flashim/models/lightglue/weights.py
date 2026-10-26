"""LightGlue checkpoint location and conversion."""

from __future__ import annotations

import re
from pathlib import Path

import torch

from flashim.models.weights_util import Checkpoint, find_weights

CHECKPOINT_ENV = "FLASHIM_LIGHTGLUE_CHECKPOINT"
# ``weights/rdd_lg.pth`` of the Google Drive folder linked from https://github.com/xtcpete/rdd (its only
# release); ``confirm=t`` skips Drive's page for files too large to virus-scan, the SHA-256 pins the content.
CHECKPOINT = Checkpoint(
    "rdd_lg.pth",
    "https://drive.usercontent.google.com/download?id=19U_vjcThjo-Ykoh4mEVg7F3agaOyvH-S"
    "&export=download&confirm=t",
    "3c26a6207a735eb7941b7b5ef454fb9b205d6d76e00cd65fbe56e710319f05a9",
)

LICENSES = {
    "code": "Apache-2.0 (LightGlue, cvg/LightGlue; RDD, xtcpete/rdd); FlashIM re-implementation",
    "weights": "RDD release checkpoint rdd_lg.pth (LightGlue trained on RDD descriptors), or cvg/LightGlue's "
    "per-extractor releases (<variant>_lightglue.pth, see CVG_FILES)",
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
    return find_weights(CHECKPOINT, path=path, env=CHECKPOINT_ENV)


def load_state_dict(
    num_heads: int, num_layers: int, path: str | Path | None = None, map_location: str | torch.device = "cpu"
) -> dict[str, torch.Tensor]:
    state = torch.load(find_checkpoint(path), map_location=map_location, weights_only=True)
    return convert_state_dict(state, num_heads, num_layers)


# cvg/LightGlue trains one LightGlue per (detector, descriptor) pair: ``aliked`` is ALIKED's own keypoints and
# descriptors, ``raco-aliked`` RaCo's keypoints with ALIKED descriptors (same architecture, different weights).
_CVG_SHA256 = {
    "superpoint": "6ff7040d0a497fc6639337946d7538dae07428c18f77a067a0b5a960e7cc551a",
    "disk": "b5b21d47ea24f2c5e501aec9c91b9716e4c8c3429a4dc1e615c133c4c9378335",
    "sift": "5b52b8d9982d43532dc042606b346bb9594c9f5a4bd6f64362c63866287b4ac0",
    "aliked": "d975e965b105311a6143194852297dff4f02aea5cc2e10cecfed966ca0e22503",
    "raco-aliked": "a804c3fe2851cc3c765cda6a9f412845893c8d8e2a3048f848d7f6ba172e6312",
}
_CVG_RELEASE = "https://github.com/cvg/LightGlue/releases/download/v0.1_arxiv"
CVG_FILES = {variant: f"{variant.replace('-', '_')}_lightglue.pth" for variant in _CVG_SHA256}
CVG_CHECKPOINTS = {
    variant: Checkpoint(file, f"{_CVG_RELEASE}/{file}", _CVG_SHA256[variant])
    for variant, file in CVG_FILES.items()
}


def load_cvg_state_dict(
    variant: str, num_heads: int, num_layers: int, path: str | Path | None = None, map_location="cpu"
) -> dict[str, torch.Tensor]:
    """cvg/LightGlue's per-extractor release (``<extractor>_lightglue.pth``)."""
    env = f"FLASHIM_{variant.upper().replace('-', '_')}_LIGHTGLUE_CHECKPOINT"
    file = find_weights(CVG_CHECKPOINTS[variant], path=path, env=env)
    return convert_state_dict(torch.load(file, map_location=map_location, weights_only=True), num_heads, num_layers)
