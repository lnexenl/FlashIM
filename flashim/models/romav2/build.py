"""Build a ready-to-run RoMaV2 from a preset, a setting and an engine."""

from __future__ import annotations

import logging
import os
from pathlib import Path

import torch
from torch import nn

from flashim.engines.compile import compile_regions
from flashim.models.romav2.config import RoMaV2Config
from flashim.models.romav2.model import RoMaV2
from flashim.models.romav2.quant import PRESETS, PROTECTED
from flashim.platforms import Feature, platform_for
from flashim.quant.apply import quantize_model
from flashim.quant.calibration import CalibrationTable, Int8CalibrationTable
from flashim.quant.passes.transformer_block import QuantizedAttention
from flashim.quant.passes.separable_conv_block import (
    MXFP8SeparableConvBlock,
    fold_block_batchnorm,
    fuse_bf16_blocks,
)

logger = logging.getLogger(__name__)

ENGINES = ("eager", "compile", "compile-full")
# RoMaV2's validated regions: compiled numerics are covered by its P8 AUC results.
COMPILE_REGIONS = ("f", "matcher.mv_vit")
# compile-full: also VGG, the DPT head and every refiner's conv stack (+ VGG BN fold).
COMPILE_FULL_REGIONS = (
    *COMPILE_REGIONS,
    "refiner_features",
    "matcher.head",
    *(f"refiners.{stride}.{part}" for stride in (4, 2, 1) for part in ("block1", "hidden_blocks")),
)


def _refiners_channels_last(model: RoMaV2) -> None:
    """Refiner conv weights to channels-last, and refiners build their inputs channels-last."""
    for refiner in model.refiners.values():
        refiner.channels_last = True
        for child in refiner.modules():
            if isinstance(child, nn.Conv2d):
                child.weight.data = child.weight.data.to(memory_format=torch.channels_last)


def load_romav2(
    preset: str = "fp8",
    *,
    setting: str = "precise",
    engine: str = "eager",
    device: torch.device | str | None = None,
    calibration: str | Path | CalibrationTable | None = None,
    strict: bool = False,
    config: RoMaV2Config | None = None,
    compile_mode: str | None = "max-autotune-no-cudagraphs",
    fast_sampling: bool | None = None,
    attention: str = "sdpa",
) -> RoMaV2:
    """Load RoMaV2 weights, apply a quantization preset and an execution engine.

    ``engine``:
      - ``"eager"``;
      - ``"compile"``: ``torch.compile`` of the DINOv3 and matcher-ViT regions (RoMaV2's
        validated configuration);
      - ``"compile-full"``: additionally folds VGG BatchNorm and compiles VGG, the DPT
        head and the refiner conv stacks (~20% faster; changes rounding vs ``compile``).
    On platforms without native MXFP8 GEMMs, FP8 presets run with emulation kernels
    (``strict=True`` raises instead).

    ``attention``: ``"sdpa"`` (torch SDPA, default), ``"sage"`` or ``"sage-fp8"`` (SageAttention INT8-QK attention
    in every quantized transformer block; needs the ``sageattention`` package; not RoMaV2-exact).

    ``fast_sampling``: fused-KDE ``sample()`` (default on; ``FLASHIM_FAST_SAMPLING=0`` restores RoMaV2's).
    """
    if preset not in PRESETS:
        raise ValueError(f"Unknown preset {preset!r}; choose from {sorted(PRESETS)}")
    if engine not in ENGINES:
        raise ValueError(f"Unknown engine {engine!r}; choose from {ENGINES}")
    cfg = config or RoMaV2Config(setting=setting)
    model = RoMaV2.from_pretrained(cfg, device=device)
    platform = platform_for(model.device)
    spec = PRESETS[preset]
    model.preset = preset
    if spec.config is not None:
        int8 = any(scheme.weights.type == "int" for scheme in spec.config.config_groups.values())
        if not int8 and not platform.supports(Feature.MXFP8_SCALED_MM):
            message = f"Preset {preset!r}: no native MXFP8 GEMM on {platform.describe()}; using emulation"
            if strict:
                raise RuntimeError(message)
            logger.warning(message)
        table = calibration
        if not isinstance(table, (CalibrationTable, Int8CalibrationTable)):
            table = (Int8CalibrationTable if int8 else CalibrationTable).load(table or spec.calibration)
        if spec.fold_refiners:
            _refiners_channels_last(model)
            fold_block_batchnorm(model.refiners)
        model.quant_report = quantize_model(model, spec.config, table, protected=PROTECTED)
        model.calibration_metadata = dict(table.metadata)
    if attention != "sdpa":
        if spec.config is None:
            raise ValueError(f"attention={attention!r} needs a quantized preset (fused transformer blocks)")
        for module in model.modules():
            if isinstance(module, QuantizedAttention):
                module.sdpa_impl = attention
        model.attention = attention
    if spec.fused_bf16_refiners:
        prefixes = tuple(f"refiners.{stride}." for stride in spec.fused_bf16_refiners)
        fused = fuse_bf16_blocks(model, prefixes)
        if model.quant_report is not None:
            model.quant_report.fused_conv_blocks.extend(fused)
    if spec.fused_refiner_io:
        for refiner in model.refiners.values():
            refiner.fused_input = True
            refiner.enable_fused_heads()
    if spec.fused_mx_blocks:
        for module in model.modules():
            if isinstance(module, MXFP8SeparableConvBlock):
                module.fuse_pointwise = True
    if spec.batched_directions:
        if not spec.fused_refiner_io:
            raise ValueError("batched_directions requires fused_refiner_io")
        model.batched_directions = True
    if engine == "compile":
        compile_regions(model, COMPILE_REGIONS, mode=compile_mode)
    elif engine == "compile-full":
        model.refiner_features.fold_batchnorm(channels_last=True)
        compile_regions(model, COMPILE_FULL_REGIONS, mode=compile_mode)
    if fast_sampling is None:
        fast_sampling = os.environ.get("FLASHIM_FAST_SAMPLING", "1") != "0"
    model.fast_sampling = fast_sampling
    model.engine = engine
    return model
