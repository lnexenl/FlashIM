"""SuperPoint checkpoints:

- ``v1`` (default): ``superpoint_v1.pth`` of cvg/LightGlue's release (Magic Leap weights, no BatchNorm).
- ``sp_v6``: ``sp_v6.tgz`` of rpautrat/SuperPoint (VGG + BatchNorm TF1 SavedModel used by OmniGlue),
  converted on first load by :mod:`flashim.models.omniglue.tf_bundle` and cached as ``sp_v6.safetensors``.
"""

from __future__ import annotations

from pathlib import Path

import torch

from flashim.models.weights_util import Checkpoint, find_weights

CHECKPOINT_ENV = "FLASHIM_SUPERPOINT_CHECKPOINT"
CHECKPOINT_SP_V6_ENV = "FLASHIM_SUPERPOINT_SP_V6_CHECKPOINT"

CHECKPOINT = Checkpoint(
    "superpoint_v1.pth",
    "https://github.com/cvg/LightGlue/releases/download/v0.1_arxiv/superpoint_v1.pth",
    "52b6708629640ca883673b5d5c097c4ddad37d8048b33f09c8ca0d69db12c40e",
)
CHECKPOINTS = {
    "v1": CHECKPOINT,
    "sp_v6": Checkpoint(
        "sp_v6.tgz",
        "https://raw.githubusercontent.com/rpautrat/SuperPoint/master/pretrained_models/sp_v6.tgz",
        "e8f0cc63e8842aff9e76caab7db470a36a1cbce6eef187d4ecd4a6f1134e0846",
    ),
}

LICENSES = {
    "code": "FlashIM re-implementation of SuperPoint (Magic Leap / rpautrat/SuperPoint MIT); no upstream code copied (LightGlue's superpoint.py carries a Magic Leap no-redistribution notice)",
    "weights": "superpoint_v1.pth (Magic Leap; non-commercial research license); sp_v6.tgz (rpautrat/SuperPoint, MIT license)",
}

# TF variable prefix -> (PyTorch Conv2d name, PyTorch BatchNorm2d name)
_SP_V6_CONV_MAP = (
    ("superpoint/vgg/conv1_1", "conv1a", "bn1a"),
    ("superpoint/vgg/conv1_2", "conv1b", "bn1b"),
    ("superpoint/vgg/conv2_1", "conv2a", "bn2a"),
    ("superpoint/vgg/conv2_2", "conv2b", "bn2b"),
    ("superpoint/vgg/conv3_1", "conv3a", "bn3a"),
    ("superpoint/vgg/conv3_2", "conv3b", "bn3b"),
    ("superpoint/vgg/conv4_1", "conv4a", "bn4a"),
    ("superpoint/vgg/conv4_2", "conv4b", "bn4b"),
    ("superpoint/detector/conv1", "convPa", "bnPa"),
    ("superpoint/detector/conv2", "convPb", "bnPb"),
    ("superpoint/descriptor/conv1", "convDa", "bnDa"),
    ("superpoint/descriptor/conv2", "convDb", "bnDb"),
)


def _convert_sp_v6(raw: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    state: dict[str, torch.Tensor] = {}
    for tf_prefix, conv_name, bn_name in _SP_V6_CONV_MAP:
        # TF Conv2D kernel [kH, kW, Cin, Cout] -> PyTorch [Cout, Cin, kH, kW]
        state[f"{conv_name}.weight"] = raw[f"{tf_prefix}/conv/kernel"].permute(3, 2, 0, 1).contiguous()
        state[f"{conv_name}.bias"] = raw[f"{tf_prefix}/conv/bias"].contiguous()
        state[f"{bn_name}.weight"] = raw[f"{tf_prefix}/bn/gamma"].contiguous()
        state[f"{bn_name}.bias"] = raw[f"{tf_prefix}/bn/beta"].contiguous()
        state[f"{bn_name}.running_mean"] = raw[f"{tf_prefix}/bn/moving_mean"].contiguous()
        state[f"{bn_name}.running_var"] = raw[f"{tf_prefix}/bn/moving_variance"].contiguous()
    return state


def load_state_dict(
    path: str | Path | None = None, map_location="cpu", *, variant: str = "v1"
) -> dict[str, torch.Tensor]:
    if variant == "v1":
        file = find_weights(CHECKPOINT, path=path, env=CHECKPOINT_ENV)
        return torch.load(file, map_location=map_location, weights_only=True)
    if variant == "sp_v6":
        from flashim.models.omniglue.tf_bundle import load_cached_safetensors

        file = find_weights(CHECKPOINTS["sp_v6"], path=path, env=CHECKPOINT_SP_V6_ENV)
        return load_cached_safetensors(file, "sp_v6.safetensors", _convert_sp_v6, map_location=map_location)
    raise ValueError(f"Unknown SuperPoint variant {variant!r}; choose from {sorted(CHECKPOINTS)}")
