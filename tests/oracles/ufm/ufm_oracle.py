"""Builds the vendored upstream UFM (``uniflowmatch/`` + ``uniception/``) from a local safetensors
checkpoint, without network access or ``jaxtyping``.

Not upstream code (FlashIM's, Apache-2.0 like the rest of the repository). Put this directory on
``sys.path`` and::

    from ufm_oracle import load_oracle
    oracle = load_oracle("~/.cache/torch/hub/checkpoints/ufm_base.safetensors", device="cuda")

Patches applied at build time (never inside the vendored files):
- stubs ``jaxtyping.Float`` if ``jaxtyping`` is not installed (UniCeption uses it only for type annotations);
- patches ``torch.hub.load`` during construction to build Meta's DINOv2 ViT-L/14 from FlashIM's already
  vendored ``tests/oracles/dedode/DeDoDe/transformer/dinov2.py`` instead of downloading from GitHub;
- loads the ``infinity1096/UFM-Base`` config and checkpoint with ``strict=True``.
"""

from __future__ import annotations

import sys
import types
from pathlib import Path
from unittest import mock

import torch
from safetensors.torch import load_model

# Exact config.json from https://huggingface.co/infinity1096/UFM-Base
UFM_BASE_CONFIG = {
    "encoder_str": "dinov2",
    "encoder_kwargs": {
        "data_norm_type": "dinov2",
        "name": "dinov2_encoder",
        "patch_size": 14,
        "size": "large",
        "with_registers": False,
    },
    "info_sharing_and_head_structure": "dual+single",
    "info_sharing_str": "global_attention",
    "info_sharing_kwargs": {
        "attn_drop": 0.0,
        "custom_positional_encoding": None,
        "depth": 12,
        "dim": 768,
        "drop_path": 0.0,
        "gradient_checkpointing": False,
        "indices": [5, 8],
        "init_values": None,
        "input_embed_dim": 1024,
        "max_num_views": 2,
        "mlp_ratio": 4.0,
        "name": "global_attention",
        "norm_intermediate": True,
        "num_heads": 12,
        "pretrained_checkpoint_path": None,
        "proj_drop": 0.0,
        "qk_norm": False,
        "qkv_bias": True,
        "size": "base",
        "use_rand_idx_pe_for_non_reference_views": False,
    },
    "head_type": "dpt",
    "feature_head_kwargs": {
        "dpt_feature": {
            "feature_dim": 256,
            "hooks": [0, 1, 2, 3],
            "input_feature_dims": [1024, 768, 768, 768],
            "layer_dims": [96, 192, 384, 768],
            "output_width_ratio": 1,
            "patch_size": 14,
            "pretrained_checkpoint_path": None,
            "use_bn": False,
        },
        "dpt_processor": {
            "hidden_dims": [128, 128],
            "input_feature_dim": 256,
            "output_dim": 2,
            "pretrained_checkpoint_path": None,
        },
    },
    "adaptors_kwargs": {
        "flow": {
            "class": "FlowAdaptor",
            "kwargs": {
                "base_shape": [224, 224],
                "flow_mean": [0.0, 0.0],
                "flow_std": [25, 25],
                "name": "flow",
                "output_normalized_coordinate": False,
                "scale_strategy": "scale_both",
            },
        }
    },
    "detach_uncertainty_head": False,
    "uncertainty_head_type": "dpt",
    "uncertainty_head_kwargs": {
        "dpt_feature": {
            "feature_dim": 256,
            "hooks": [0, 1, 2, 3],
            "input_feature_dims": [1024, 768, 768, 768],
            "layer_dims": [96, 192, 384, 768],
            "output_width_ratio": 1,
            "patch_size": 14,
            "pretrained_checkpoint_path": None,
            "use_bn": False,
        },
        "dpt_processor": {
            "hidden_dims": [128, 128],
            "input_feature_dim": 256,
            "output_dim": 1,
            "pretrained_checkpoint_path": None,
        },
    },
    "uncertainty_adaptors_kwargs": {
        "non_occluded_mask": {
            "class": "MaskAdaptor",
            "kwargs": {"name": "non_occluded_mask"},
        }
    },
    "inference_resolution": [560, 420],
}


def _ensure_jaxtyping() -> None:
    if "jaxtyping" in sys.modules:
        return
    try:
        import jaxtyping  # noqa: F401
        return
    except ImportError:
        pass
    stub = types.ModuleType("jaxtyping")

    class _Subscriptable(type):
        def __getitem__(cls, item):
            return object

    class Float(metaclass=_Subscriptable):
        pass

    stub.Float = Float
    sys.modules["jaxtyping"] = stub


def _build_dinov2_vitl14():
    dedode_root = Path(__file__).resolve().parent.parent / "dedode"
    added = str(dedode_root) not in sys.path
    if added:
        sys.path.insert(0, str(dedode_root))
    try:
        from DeDoDe.transformer.dinov2 import vit_large
        return vit_large(patch_size=14, img_size=518, init_values=1.0, block_chunks=0)
    finally:
        if added:
            sys.path.remove(str(dedode_root))


def load_oracle(checkpoint: str | Path, *, device: str | torch.device = "cuda"):
    """Load upstream ``UniFlowMatchConfidence`` offline from ``checkpoint``."""
    _ensure_jaxtyping()
    from uniflowmatch.models.ufm import UniFlowMatchConfidence

    def _mock_hub_load(repo, model_name, *args, **kwargs):
        if repo == "facebookresearch/dinov2" and model_name == "dinov2_vitl14":
            return _build_dinov2_vitl14()
        raise RuntimeError(f"the UFM oracle does not download ({repo}, {model_name})")

    with mock.patch.object(torch.hub, "load", side_effect=_mock_hub_load):
        model = UniFlowMatchConfidence(**UFM_BASE_CONFIG)
    missing, unexpected = load_model(model, str(Path(checkpoint).expanduser()), strict=True)
    assert not missing and not unexpected, (missing, unexpected)
    return model.to(device).eval()
