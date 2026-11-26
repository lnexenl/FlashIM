"""OmniGlue checkpoints (Google Research ``og_export.zip`` and Meta ``dinov2_vitb14_pretrain.pth``).

``og_export.zip`` is downloaded directly from Google's release URL, verified against its SHA-256,
parsed in pure Python via :mod:`flashim.models.omniglue.tf_bundle`, and cached as ``og_export.safetensors``
next to the archive (zero runtime TensorFlow dependency).
"""

from __future__ import annotations

from pathlib import Path

import torch

from flashim.models.omniglue.tf_bundle import load_cached_safetensors
from flashim.models.weights_util import Checkpoint, ModelInfo, find_weights

CHECKPOINT_ENV = "FLASHIM_OMNIGLUE_CHECKPOINT"
DINOV2_ENV = "FLASHIM_OMNIGLUE_DINOV2_CHECKPOINT"

CHECKPOINTS = {
    "omniglue": Checkpoint(
        "og_export.zip",
        "https://storage.googleapis.com/omniglue/og_export.zip",
        "3a4f95421117e61309a563d30baff904b8db11aa8a71a0be87086b43a862b7f5",
    ),
    "dinov2": Checkpoint(
        "dinov2_vitb14_pretrain.pth",
        "https://dl.fbaipublicfiles.com/dinov2/dinov2_vitb14/dinov2_vitb14_pretrain.pth",
        "0b8b82f85de91b424aded121c7e1dcc2b7bc6d0adeea651bf73a13307fad8c73",
    ),
}

LICENSES = {
    "code": "Apache-2.0 (Google LLC, OmniGlue; Meta, DINOv2); MIT (rpautrat/SuperPoint sp_v6); FlashIM re-implementation",
    "weights": "og_export.zip (Apache-2.0, Google LLC); dinov2_vitb14_pretrain.pth (Apache-2.0, Meta); sp_v6.tgz (MIT, rpautrat/SuperPoint)",
}
MODEL_INFO = ModelInfo(("Apache-2.0", "MIT"), ("Apache-2.0", "MIT"))
MODEL_INFO_OVERRIDES = {
    "omniglue-matcher": ModelInfo(("Apache-2.0",), ("Apache-2.0",)),
}


def convert_omniglue_bundle(raw: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Map raw ``og_export`` TensorBundle entries (``model/variables/<i>/.ATTRIBUTES/VARIABLE_VALUE``)
    to FlashIM's :class:`~flashim.models.omniglue.model.OmniGlueMatcher` state dict.

    Mapping verified bit-identical against ``tf.saved_model.load(og_export).signatures['serving_default'].variables``
    by ``scripts/convert_omniglue_weights.py`` for all 317 tensors:

    - ``0..25``: keypoint encoder (4x ``dense/kernel, bias, bn/gamma, beta, moving_mean, moving_variance`` + ``dense_4``)
    - ``26..277``: 18 attentional aggregation layers' trainable weights (14 tensors/layer: Q, K, V, Out, MLP0, BN gamma/beta, MLP1)
    - ``278..313``: 18 attentional aggregation layers' non-trainable BN stats (2 tensors/layer: moving_mean, moving_variance)
    - ``314..316``: ``final_proj`` (``dense_113``) kernel/bias and Sinkhorn ``bin_score`` (``Variable:0``)
    """
    def var(idx: int) -> torch.Tensor:
        return raw[f"model/variables/{idx}/.ATTRIBUTES/VARIABLE_VALUE"]

    state: dict[str, torch.Tensor] = {}
    # Keypoint encoder: dense .. dense_3 (+ batch_normalization .. batch_normalization_3), then dense_4
    for k in range(4):
        base = 6 * k
        state[f"kenc.layers.{k}.linear.weight"] = var(base + 0).t().contiguous()
        state[f"kenc.layers.{k}.linear.bias"] = var(base + 1).contiguous()
        state[f"kenc.layers.{k}.bn.weight"] = var(base + 2).contiguous()
        state[f"kenc.layers.{k}.bn.bias"] = var(base + 3).contiguous()
        state[f"kenc.layers.{k}.bn.running_mean"] = var(base + 4).contiguous()
        state[f"kenc.layers.{k}.bn.running_var"] = var(base + 5).contiguous()
    state["kenc.out.weight"] = var(24).t().contiguous()
    state["kenc.out.bias"] = var(25).contiguous()

    # 18 attentional aggregation layers (alternating self-attention and DINO-masked cross-attention)
    for layer_idx in range(18):
        base = 26 + 14 * layer_idx
        prefix = f"gnn.layers.{layer_idx}"
        for proj_i, proj_name in enumerate(("q_proj", "k_proj", "v_proj", "out_proj")):
            state[f"{prefix}.attn.{proj_name}.weight"] = var(base + 2 * proj_i).t().contiguous()
            state[f"{prefix}.attn.{proj_name}.bias"] = var(base + 2 * proj_i + 1).contiguous()
        state[f"{prefix}.mlp.0.weight"] = var(base + 8).t().contiguous()
        state[f"{prefix}.mlp.0.bias"] = var(base + 9).contiguous()
        state[f"{prefix}.mlp.1.weight"] = var(base + 10).contiguous()
        state[f"{prefix}.mlp.1.bias"] = var(base + 11).contiguous()
        state[f"{prefix}.mlp.1.running_mean"] = var(278 + 2 * layer_idx).contiguous()
        state[f"{prefix}.mlp.1.running_var"] = var(278 + 2 * layer_idx + 1).contiguous()
        state[f"{prefix}.mlp.3.weight"] = var(base + 12).t().contiguous()
        state[f"{prefix}.mlp.3.bias"] = var(base + 13).contiguous()

    state["final_proj.weight"] = var(314).t().contiguous()
    state["final_proj.bias"] = var(315).contiguous()
    state["bin_score"] = var(316).contiguous()
    return state


def load_matcher_state_dict(path: str | Path | None = None, map_location="cpu") -> dict[str, torch.Tensor]:
    file = find_weights(CHECKPOINTS["omniglue"], path=path, env=CHECKPOINT_ENV)
    return load_cached_safetensors(file, "og_export.safetensors", convert_omniglue_bundle, map_location=map_location)


def load_dinov2_state_dict(path: str | Path | None = None, map_location="cpu") -> dict[str, torch.Tensor]:
    file = find_weights(CHECKPOINTS["dinov2"], path=path, env=DINOV2_ENV)
    return torch.load(file, map_location=map_location, weights_only=True)
