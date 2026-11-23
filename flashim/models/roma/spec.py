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
from flashim.models.dedode.spec import fused_first_conv_epilogue, fused_pool_epilogue
from flashim.models.roma import quant
from flashim.models.roma.config import RoMaConfig
from flashim.models.roma.model import RoMa
from flashim.quant.passes.separable_conv_block import (
    fold_block_batchnorm,
    fuse_bf16_blocks,
    fuse_int8_producers,
)
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


def _compile_regions(model: nn.Module) -> tuple[str, ...]:
    """Skip compile regions under ``int8`` (every heavy block is already fused Triton/TileLang with <0.5 ms launch
    gap, and Dynamo 2.11 cannot trace ``@triton.autotune`` inside ``StaticInt8Conv2d``)."""
    from flashim.quant.methods.conv2d import StaticInt8Conv2d

    if any(isinstance(m, StaticInt8Conv2d) for m in model.encoder.cnn.modules()):
        return ()
    return COMPILE_REGIONS


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


def _pad_refiners(model: RoMa, ctx: BuildContext) -> dict[str, int]:
    """Zero-pad ConvRefiner channels (1377/1137/569/144/24 -> 1408/1152/576/192/32); mathematically exact.

    For ``fp8``, enables ``fuse_quantized`` on scales 16/8/4/2 so ``quantize_model`` builds dynamic
    :class:`MXFP8SeparableConvBlock` modules (TileLang TMA depthwise producer -> cuBLAS ``mxfp8_scaled_mm``).
    """
    padded = {}
    enable_mx_fuse = ctx.config.precision.quant == "fp8"
    for scale, target in quant.REFINER_PADDED_CHANNELS.items():
        refiner = model.decoder.conv_refiner[scale]
        if refiner.pad_channels(target):
            padded[scale] = target
        if enable_mx_fuse and scale != "1":
            for block in (refiner.block1, *refiner.hidden_blocks):
                block.fuse_quantized = True
    return padded


def _channels_last(model: RoMa, ctx: BuildContext) -> list[str]:
    """NHWC weights and inputs for VGG19-BN (skipped when VGG is INT8-quantized, which reads NCHW) and refiners."""
    applied = []
    q = ctx.config.precision.quant
    vgg_quantized = q is not None and ctx.spec.quant_configs[q].scheme_for("encoder.cnn.layers.3") is not None
    if not vgg_quantized:
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


def _fused_refiner_io(model: RoMa, ctx: BuildContext) -> list[str]:
    """Fuse refiner input assembly (``ops.roma_refiner_input``), FP32 3-ch output head (``ops.roma_refiner_head``),
    channels-last projection GEMMs, and contiguous last-dim coarse softmax (avoids 4.1 ms SpatialSoftMax)."""
    model.decoder.fused_io = True
    model.decoder.fast_softmax = True
    applied = []
    for scale, refiner in model.decoder.conv_refiner.items():
        refiner.enable_fused_io()
        applied.append(f"decoder.conv_refiner.{scale}")
    return applied


def _fused_bf16_refiners(model: RoMa, ctx: BuildContext) -> list[str]:
    """Replace all 5 ConvRefiners' separable conv blocks with :class:`BF16SeparableConvBlock`."""
    return fuse_bf16_blocks(model, ("decoder.conv_refiner.",))


def _fused_bf16_s1(model: RoMa, ctx: BuildContext) -> list[str]:
    """Replace scale-1 ConvRefiner blocks (padded to 32 ch) with fused 32-ch :class:`BF16SeparableConvBlock`."""
    return fuse_bf16_blocks(model, ("decoder.conv_refiner.1.",))


def _int8_refiner_producers(model: RoMa, ctx: BuildContext) -> list[str] | str:
    """int8: fuse depthwise 5x5 + ReLU + static INT8 quantize (TileLang TMA / Triton) into scales 16/8/4/2."""
    if ctx.config.precision.quant != "int8":
        return "skipped: not int8"
    return fuse_int8_producers(model, ("decoder.conv_refiner.",))


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
    # Cast encoders, coarse TransformerDecoder, fine refiner projections (scales 8/4/2/1), and refiner conv stacks
    # to compute_dtype; keep scale-16 projection (feeds GP Cholesky), GP, disp_emb, and out_conv in FP32.
    model.encoder.cnn.to(dt)
    model.encoder.dinov2.to(dt)
    model.decoder.embedding_decoder.blocks.to(dt)
    model.decoder.embedding_decoder.to_out.to(dt)
    model.decoder.embedding_decoder.compute_dtype = dt
    for scale in ("8", "4", "2", "1"):
        model.decoder.proj[scale].to(dt)
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
    "pad_refiners": Pass(
        _pad_refiners, stage="pre_quant", description="zero-pad refiner channels to 1408/1152/576/192/32"
    ),
    "channels_last": Pass(
        _channels_last, stage="pre_quant", exact=False, description="NHWC weights and inputs for VGG and refiners"
    ),
    "fold_layer_scale": Pass(
        _fold_layer_scale, stage="pre_quant", exact=False, description="fold DINOv2 LayerScale into proj / fc2"
    ),
    "fused_bf16_refiners": Pass(
        _fused_bf16_refiners, exact=False, description="bf16: fused depthwise+ReLU (+ pointwise at scale 1) refiners"
    ),
    "fused_refiner_io": Pass(
        _fused_refiner_io, exact=False, description="fused refiner input/head Triton ops + last-dim coarse softmax"
    ),
    "fused_bf16_s1": Pass(
        _fused_bf16_s1, exact=False, description="fused 32-channel BF16 scale-1 refiner blocks"
    ),
    "int8_refiner_producers": Pass(
        _int8_refiner_producers, exact=False, description="int8: TileLang/Triton depthwise+ReLU->INT8 refiner producers"
    ),
    "fused_pool_epilogue": Pass(
        fused_pool_epilogue, description="int8: max pool + next conv quantize in VGG INT8 conv epilogue"
    ),
    "fused_first_conv_epilogue": Pass(
        fused_first_conv_epilogue, description="int8: first VGG conv bias + ReLU + quantize as one pass"
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
            "optimize": {
                "passes": ["fold_batchnorm", "pad_refiners", "channels_last", "fused_bf16_refiners", "fused_refiner_io"]
            },
        },
        "int8": {
            "precision": {"compute_dtype": "bf16", "quant": "int8"},
            "optimize": {
                "passes": [
                    "fold_batchnorm",
                    "pad_refiners",
                    "channels_last",
                    "fold_layer_scale",
                    "fused_dynamic_vit_blocks",
                    "int8_refiner_producers",
                    "fused_pool_epilogue",
                    "fused_first_conv_epilogue",
                    "fused_bf16_s1",
                    "fused_refiner_io",
                ]
            },
        },
        "fp8": {
            "precision": {"compute_dtype": "bf16", "quant": "fp8"},
            "optimize": {
                "passes": [
                    "fold_batchnorm",
                    "pad_refiners",
                    "channels_last",
                    "fold_layer_scale",
                    "fused_block_epilogues",
                    "fused_bf16_s1",
                    "fused_refiner_io",
                ]
            },
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
        "minima/fp8": quant.CALIBRATION_DIR / "roma_minima_fp8_mega.safetensors",
        "minima/int8": quant.CALIBRATION_DIR / "roma_minima_int8_mega.safetensors",
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
        engines={"eager": EngineSpec(), "compile": EngineSpec(_compile_regions)},
        set_precision=_set_precision,
        set_attention=_set_attention,
        apply_inference=_apply_inference,
    )


SPEC = _spec("roma", "outdoor", "RoMa (v1) dense matcher (outdoor default; VGG19-BN + DINOv2 ViT-L/14 + GP + refiners)")
SPEC_INDOOR = _spec("roma-indoor", "indoor", "RoMa (v1) dense matcher (indoor weights)")
SPEC_MINIMA = _spec(
    "minima-roma",
    "minima",
    "MINIMA-RoMa modality-invariant dense matcher (RoMa v1 architecture, MINIMA weights)",
)
