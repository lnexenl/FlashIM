"""XFeat checkpoints from verlab/accelerated_features: ``xfeat.pt`` (net) and ``xfeat-lighterglue.pt`` (net + matcher)."""

from __future__ import annotations

from pathlib import Path

import torch

from flashim.models.weights_util import find_weights

HELP = "Download xfeat.pt / xfeat-lighterglue.pt from the `weights/` folder of https://github.com/verlab/accelerated_features"
LICENSES = {"code": "Apache-2.0 (verlab/accelerated_features); FlashIM re-implementation", "weights": "XFeat release"}


def _find(name: str, path):
    return find_weights(
        name, path=path, env="FLASHIM_XFEAT_CHECKPOINT", help=HELP,
        extra=[Path.home() / "work" / "oracles" / "accelerated_features" / "weights" / name],
    )


def load_state_dict(path: str | Path | None = None, map_location="cpu") -> dict[str, torch.Tensor]:
    return torch.load(_find("xfeat.pt", path), map_location=map_location, weights_only=True)


def load_lighterglue_state_dict(num_heads: int, num_layers: int, path: str | Path | None = None, map_location="cpu"):
    """The ``matcher.*`` entries of ``xfeat-lighterglue.pt`` in FlashIM's LightGlue layout."""
    from flashim.models.lightglue.weights import convert_state_dict

    state = torch.load(_find("xfeat-lighterglue.pt", path), map_location=map_location, weights_only=True)
    state = {k.removeprefix("matcher."): v for k, v in state.items() if k.startswith("matcher.")}
    return convert_state_dict(state, num_heads, num_layers)
