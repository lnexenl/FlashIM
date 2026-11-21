"""RoMa (v1)'s :class:`~flashim.build.ModelSpec` (``roma`` = outdoor, ``roma-indoor`` = indoor).

Variants follow the same pattern as ``loma`` / ``loma-<variant>`` and ``loftr``: ``roma`` defaults to ``outdoor``
(and also accepts ``overrides={"variant": "indoor"}``), while ``roma-indoor`` is registered directly for indoor
matching. Presets are ``fp32``, ``bf16`` (default), ``int8``, and ``fp8``; measured fidelity and speed live in
:mod:`flashim.models.roma.quant`.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from typing import Any

import torch
from torch import nn

from flashim.build import BuildContext, EngineSpec, ModelSpec, Pass
from flashim.build import dtype as resolve_dtype
from flashim.config import RunConfig
from flashim.models.roma import quant
from flashim.models.roma.config import RoMaConfig
from flashim.models.roma.model import RoMa
from flashim.quant.passes.separable_conv_block import fold_block_batchnorm
from flashim.quant.passes.transformer_block import (
    QuantizedAttention,
    enable_fused_epilogues,
    fold_layer_scale,
)

COMPILE_REGIONS = (
    "encoder.cnn",
    "encoder.dinov2",
    "decoder.embedding_decoder",
    *(f"decoder.conv_refiner.{s}.{part}" for s in ("16", "8", "4", "2", "1") for part in ("block1", "hidden_blocks")),
)


def _loader(default_variant: str):
    def load(config: RunConfig) -> RoMa:
        variant = config.variant or default_variant
        return RoMa.from_pretrained(
            RoMaConfig(**config.arch),
            variant=variant,
            checkpoint=config.checkpoint,
            device=config.device,
        )

    return load


def _fold_batchnorm(model: RoMa, ctx: BuildContext) -> int:
    """Fold eval BatchNorm into preceding convs across VGG19-BN, projections, and all ConvRefiners."""
    folded = 0
    for conv_name, bn_name in model.batchnorm_pairs():
        conv = model.get_submodule(conv_name)
        bn = model.get_submodule(bn_name)
        if isinstance(conv, nn.Conv2d) and isinstance(bn, nn.BatchNorm2d):
            parent_name, _, conv_attr = conv_name.rpartition(".")
            parent = model.get_submodule(parent_name) if parent_name else model
            setattr(parent, conv_attr, nn.utils.fusion.fuse_conv_bn_eval(conv, bn))
            bn_parent_name, _, bn_attr = bn_name.rpartition(".")
            bn_parent = model.get_submodule(bn_parent_name) if bn_parent_name else model
            setattr(bn_parent, bn_attr, nn.Identity())
            folded += 1
    # Also ensure any SeparableConvBlock.norm is marked folded if not already.
    fold_block_batchnorm(model.decoder.conv_refiner)
    return folded


def _channels_last(model: RoMa, ctx: BuildContext) -> list[str]:
    """NHWC weights and inputs for VGG19-BN and the 5 ConvRefiners."""
    applied = []
    model.encoder.channels_last = True
    for m in model.encoder.cnn.modules():
        if isinstance(m, nn.Conv2d):
            m.weight.data = m.weight.data.to(memory_format=torch.channels_last)
    applied.append("encoder.cnn")
    for scale, refiner in model.decoder.conv_refiner.items():
        refiner.channels_last = True
        for m in (*refiner.block1.modules(), *refiner.hidden_blocks.modules()):
            if isinstance(m, nn.Conv2d):
                m.weight.data = m.weight.data.to(memory_format=torch.channels_last)
        applied.append(f"decoder.conv_refiner.{scale}")
    return applied


def _fold_layer_scale(model: RoMa, ctx: BuildContext) -> list[str] | str:
    """Fold DINOv2 LayerScale into ``attn.proj`` / ``mlp.fc2`` before quantization."""
    if not ctx.config.precision.quant:
        return "skipped: not quantized"
    return fold_layer_scale(model)


def _fused_dynamic_vit_blocks(model: RoMa, ctx: BuildContext) -> list[str] | str:
    """int8: fuse DINOv2's DynamicInt8Linear blocks (LayerNorm + per-row quantize, GELU / residual epilogues)."""
    if ctx.config.precision.quant != "int8":
        return "skipped: not int8"
    from flashim.quant.passes.dynamic_transformer_block import fuse_dynamic_blocks

    return fuse_dynamic_blocks(model) or "skipped: no dynamic INT8 transformer blocks"


def _fused_block_epilogues(model: RoMa, ctx: BuildContext) -> list[str] | str:
    """fp8: fuse biases / residual adds into DINOv2's MXFP8 GEMM epilogues and quantize kernels."""
    return enable_fused_epilogues(model) if ctx.config.precision.quant == "fp8" else "skipped: not fp8"


def _set_precision(model: RoMa, config: RunConfig) -> None:
    dt = resolve_dtype(config.precision.compute_dtype)
    if dt == torch.float32:
        return
    # Cast encoders, coarse TransformerDecoder, and refiner conv stacks to compute_dtype;
    # keep geometry/coordinate layers (GP, projections, disp_emb, out_conv) in FP32.
    model.encoder.cnn.to(dt)
    model.encoder.dinov2.to(dt)
    model.decoder.embedding_decoder.blocks.to(dt)
    model.decoder.embedding_decoder.to_out.to(dt)
    model.decoder.embedding_decoder.compute_dtype = dt
    for refiner in model.decoder.conv_refiner.values():
        refiner.block1.to(dt)
        refiner.hidden_blocks.to(dt)
        refiner.compute_dtype = dt


def _set_attention(model: nn.Module, impl: str) -> None:
    for module in model.modules():
        if isinstance(module, QuantizedAttention):
            module.sdpa_impl = impl


def _apply_inference(model: RoMa, inference: Mapping[str, Any]) -> None:
    model.fast_sampling = inference.get("fast_sampling", os.environ.get("FLASHIM_FAST_SAMPLING", "1") != "0")


PASSES = {
    "fold_batchnorm": Pass(
        _fold_batchnorm, stage="pre_quant", exact=False, description="fold eval BatchNorm into VGG, proj, and refiners"
    ),
    "channels_last": Pass(
        _channels_last, stage="pre_quant", exact=False, description="NHWC weights and inputs for VGG and refiners"
    ),
    "fold_layer_scale": Pass(
        _fold_layer_scale, stage="pre_quant", exact=False, description="fold DINOv2 LayerScale into proj / fc2"
    ),
    "fused_dynamic_vit_blocks": Pass(
        _fused_dynamic_vit_blocks, description="int8: fused LayerNorm + row quantize and epilogues for DINOv2 blocks"
    ),
    "fused_block_epilogues": Pass(
        _fused_block_epilogues, exact=False, description="fp8: fuse DINOv2 biases and residual adds into GEMM epilogues"
    ),
}


def _spec(name: str, default_variant: str, description: str) -> ModelSpec:
    presets: dict[str, dict[str, Any]] = {
        "fp32": {},
        "bf16": {
            "precision": {"compute_dtype": "bf16"},
            "optimize": {"passes": ["fold_batchnorm", "channels_last"]},
        },
        "int8": {
            "extends": "bf16",
            "precision": {"quant": "int8"},
            "optimize": {"passes+": ["fold_layer_scale", "fused_dynamic_vit_blocks"]},
        },
        "fp8": {
            "extends": "bf16",
            "precision": {"quant": "fp8"},
            "optimize": {"passes+": ["fold_layer_scale", "fused_block_epilogues"]},
        },
    }
    if default_variant != "outdoor":
        presets = {k: {"variant": default_variant, **v} for k, v in presets.items()}
    calibrations = {
        "fp8": quant.CALIBRATION_DIR / f"roma_{default_variant}_fp8_mega.safetensors",
        "int8": quant.CALIBRATION_DIR / f"roma_{default_variant}_int8_mega.safetensors",
        "outdoor/fp8": quant.CALIBRATION_DIR / "roma_outdoor_fp8_mega.safetensors",
        "outdoor/int8": quant.CALIBRATION_DIR / "roma_outdoor_int8_mega.safetensors",
        "indoor/fp8": quant.CALIBRATION_DIR / "roma_indoor_fp8_mega.safetensors",
        "indoor/int8": quant.CALIBRATION_DIR / "roma_indoor_int8_mega.safetensors",
    }
    return ModelSpec(
        name=name,
        role="dense",
        description=description,
        load=_loader(default_variant),
        presets=presets,
        default_preset="bf16",
        quant_configs=quant.QUANT_CONFIGS,
        calibrations=calibrations,
        protected=quant.PROTECTED,
        passes=PASSES,
        engines={"eager": EngineSpec(), "compile": EngineSpec(COMPILE_REGIONS)},
        set_precision=_set_precision,
        set_attention=_set_attention,
        apply_inference=_apply_inference,
    )


SPEC = _spec("roma", "outdoor", "RoMa (v1) dense matcher (outdoor default; VGG19-BN + DINOv2 ViT-L/14 + GP + refiners)")
SPEC_INDOOR = _spec("roma-indoor", "indoor", "RoMa (v1) dense matcher (indoor weights)")
