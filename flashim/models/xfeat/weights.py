"""XFeat checkpoints from verlab/accelerated_features: ``xfeat.pt`` (net) and ``xfeat-lighterglue.pt`` (net + matcher)."""

from __future__ import annotations

from pathlib import Path

import torch

from flashim.models.weights_util import Checkpoint, find_weights

_REPO = "https://github.com/verlab/accelerated_features/raw/main/weights"
CHECKPOINTS = {
    name: Checkpoint(name, f"{_REPO}/{name}", sha)
    for name, sha in (
        ("xfeat.pt", "0f5187fd7bedd26c7fe6acc9685444493a165a35ecc087b33c2db3627f3ea10b"),
        ("xfeat-lighterglue.pt", "766102df37f11189efe5b0811d1f47c72b22629b79bfabfcfff9d2a2f84654b8"),
    )
}
LICENSES = {"code": "Apache-2.0 (verlab/accelerated_features); FlashIM re-implementation", "weights": "XFeat release"}


def _find(name: str, path):
    return find_weights(CHECKPOINTS[name], path=path, env="FLASHIM_XFEAT_CHECKPOINT")


def load_state_dict(path: str | Path | None = None, map_location="cpu") -> dict[str, torch.Tensor]:
    return torch.load(_find("xfeat.pt", path), map_location=map_location, weights_only=True)


def load_lighterglue_state_dict(num_heads: int, num_layers: int, path: str | Path | None = None, map_location="cpu"):
    """The ``matcher.*`` entries of ``xfeat-lighterglue.pt`` in FlashIM's LightGlue layout."""
    from flashim.models.lightglue.weights import convert_state_dict

    state = torch.load(_find("xfeat-lighterglue.pt", path), map_location=map_location, weights_only=True)
    state = {k.removeprefix("matcher."): v for k, v in state.items() if k.startswith("matcher.")}
    return convert_state_dict(state, num_heads, num_layers)
