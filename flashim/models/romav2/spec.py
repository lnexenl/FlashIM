"""RoMaV2's :class:`~flashim.build.ModelSpec`.

Presets are generated from :data:`flashim.models.romav2.quant.PRESETS` (whose docstring holds the accuracy evidence),
so the quantization config and the refiner flags of a preset keep one definition. RoMaV2 always runs mixed BF16
internally (DINOv3 weights BF16, autocast), so every preset's ``compute_dtype`` is ``bf16``.
"""

from __future__ import annotations

import dataclasses
import os

import torch
from torch import nn

from flashim.build import BuildContext, EngineSpec, ModelSpec, Pass
from flashim.config import RunConfig
from flashim.models.romav2 import quant
from flashim.models.romav2.config import RoMaV2Config
from flashim.models.romav2.model import RoMaV2

# RoMaV2's validated regions: compiled numerics are covered by its P8 AUC results.
COMPILE_REGIONS = ("f", "matcher.mv_vit")
# compile-full: also VGG, the DPT head and every refiner's conv stack (+ VGG BN fold).
COMPILE_FULL_REGIONS = (
    *COMPILE_REGIONS,
    "refiner_features",
    "matcher.head",
    *(f"refiners.{stride}.{part}" for stride in (4, 2, 1) for part in ("block1", "hidden_blocks")),
)


def _load(config: RunConfig) -> RoMaV2:
    return RoMaV2.from_pretrained(RoMaV2Config(**config.arch), device=config.device)


def _fold_refiners(model: RoMaV2, ctx: BuildContext) -> list[str]:
    """Refiner conv weights channels-last (and refiners build their inputs channels-last), BatchNorm folded."""
    from flashim.quant.passes.separable_conv_block import fold_block_batchnorm

    for refiner in model.refiners.values():
        refiner.channels_last = True
        for child in refiner.modules():
            if isinstance(child, nn.Conv2d):
                child.weight.data = child.weight.data.to(memory_format=torch.channels_last)
    return fold_block_batchnorm(model.refiners)


def _fold_fine_features(model: RoMaV2, ctx: BuildContext) -> int:
    """VGG19-BN fine features: BatchNorm folded into the convs, channels-last weights (what ``compile-full`` sets up
    for its compiled VGG). Not exact: the fold changes rounding."""
    return model.refiner_features.fold_batchnorm(channels_last=True)


def _fused_bf16(stride: int):
    def apply(model: RoMaV2, ctx: BuildContext) -> list[str]:
        from flashim.quant.passes.separable_conv_block import fuse_bf16_blocks

        fused = fuse_bf16_blocks(model, (f"refiners.{stride}.",))
        if model.quant_report is not None:
            model.quant_report.fused_conv_blocks.extend(fused)
        return fused

    return apply


def _fused_refiner_io(model: RoMaV2, ctx: BuildContext) -> None:
    for refiner in model.refiners.values():
        refiner.fused_input = True
        refiner.enable_fused_heads()


def _fused_mx_blocks(model: RoMaV2, ctx: BuildContext) -> int:
    from flashim.quant.passes.separable_conv_block import MXFP8SeparableConvBlock

    blocks = [m for m in model.modules() if isinstance(m, MXFP8SeparableConvBlock)]
    for block in blocks:
        block.fuse_pointwise = True
    return len(blocks)


def _fused_rope(model: RoMaV2, ctx: BuildContext) -> int:
    """DINOv3's RoPE through ``ops.rope_qk`` dispatch (one fused kernel for q and k) instead of the pinned upstream
    math. Not exact: the kernel's FMA contraction moves 24% of the rotated elements by one BF16 ulp. Measured on a
    Jetson AGX Orin, 16 x 2505 x 64 BF16: 1.54 -> 0.46 ms per call, 42 calls per mega1500 pair."""
    pinned = [module for module in model.f.modules() if getattr(module, "rope_impl", None) == "reference"]
    for module in pinned:
        module.rope_impl = None
    return len(pinned)


def _batched_directions(model: RoMaV2, ctx: BuildContext) -> None:
    model.batched_directions = True


def _set_attention(model: nn.Module, impl: str) -> None:
    from flashim.quant.passes.transformer_block import QuantizedAttention

    for module in model.modules():
        if isinstance(module, QuantizedAttention):
            module.sdpa_impl = impl


def _compile_full_setup(model: RoMaV2, ctx: BuildContext) -> None:
    model.refiner_features.fold_batchnorm(channels_last=True)


def _apply_inference(model: RoMaV2, inference: dict) -> None:
    model.fast_sampling = inference.get("fast_sampling", os.environ.get("FLASHIM_FAST_SAMPLING", "1") != "0")


def _validate(config: RunConfig) -> None:
    if config.optimize.attention != "sdpa" and config.precision.quant is None:
        raise ValueError(f"attention={config.optimize.attention!r} needs a quantized preset (fused transformer blocks)")
    passes = config.optimize.passes
    if "batched_directions" in passes and "fused_refiner_io" not in passes:
        raise ValueError("batched_directions requires fused_refiner_io")


def _overlay(preset: quant.Preset) -> dict:
    passes = ["fold_refiners"] if preset.fold_refiners else []
    passes += ["fold_fine_features"] if preset.fold_fine_features else []
    passes += [f"fused_bf16_s{stride}" for stride in preset.fused_bf16_refiners]
    passes += [name for name, on in (
        ("fused_refiner_io", preset.fused_refiner_io),
        ("fused_mx_blocks", preset.fused_mx_blocks),
        ("batched_directions", preset.batched_directions),
        ("fused_rope", preset.fused_rope),
    ) if on]
    return {
        "precision": {"compute_dtype": "bf16", "quant": preset.name if preset.config is not None else None},
        "optimize": {"passes": passes},
    }


_ALL_PRESETS = {**quant.PRESETS, **quant.ORACLE_PRESETS}
_QUANTIZED = {name: p for name, p in _ALL_PRESETS.items() if p.config is not None}
# Every FP8 preset falls back to ``int8`` (the oracle presets' refiner blocks are BF16 there, close enough).
_FP8_FALLBACKS = {name: "int8" for name in _QUANTIZED if name.startswith("fp8")}

SPEC = ModelSpec(
    name="romav2",
    role="dense",
    description="RoMa v2 dense matcher (DINOv3 ViT-L/16 + MV-ViT + DPT + conv refiners)",
    load=_load,
    presets={name: _overlay(p) for name, p in _ALL_PRESETS.items()},
    default_preset="fp8",
    quant_configs={name: p.config for name, p in _QUANTIZED.items()},
    calibrations={name: p.calibration for name, p in _QUANTIZED.items()},
    fp8_fallbacks=_FP8_FALLBACKS,
    protected=quant.PROTECTED,
    passes={
        "fold_refiners": Pass(_fold_refiners, stage="pre_quant", description="channels-last refiners, BN folded"),
        "fold_fine_features": Pass(
            _fold_fine_features, stage="pre_quant", exact=False, description="channels-last VGG fine features, BN folded"
        ),
        **{
            f"fused_bf16_s{s}": Pass(_fused_bf16(s), exact=False, description=f"fused BF16 stride-{s} refiner blocks")
            for s in (4, 2, 1)
        },
        "fused_refiner_io": Pass(_fused_refiner_io, exact=False, description="fused refiner input + output heads"),
        "fused_mx_blocks": Pass(_fused_mx_blocks, exact=False, description="dw + quantize + pw MXFP8 in one kernel"),
        "batched_directions": Pass(_batched_directions, description="both match directions as one batch"),
        "fused_rope": Pass(_fused_rope, exact=False, description="DINOv3 RoPE as one fused kernel"),
    },
    engines={
        "eager": EngineSpec(),
        "compile": EngineSpec(COMPILE_REGIONS),
        "compile-full": EngineSpec(COMPILE_FULL_REGIONS, setup=_compile_full_setup),
    },
    set_attention=_set_attention,
    apply_inference=_apply_inference,
    validate=_validate,
)


def legacy_overrides(
    *,
    setting: str = "precise",
    engine: str = "eager",
    device=None,
    calibration=None,
    strict: bool = False,
    config: RoMaV2Config | None = None,
    compile_mode: str | None = "max-autotune-no-cudagraphs",
    fast_sampling: bool | None = None,
    attention: str = "sdpa",
) -> dict:
    """``load_romav2`` keyword arguments -> a config overlay."""
    from flashim.quant.calibration import CalibrationTable, Int8CalibrationTable

    overlay: dict = {
        "arch": dataclasses.asdict(config) if config is not None else {"setting": setting},
        "engine": {"kind": engine, "mode": compile_mode},
        "optimize": {"strict": strict, "attention": attention},
    }
    if device is not None:
        overlay["device"] = str(device)
    if isinstance(calibration, (CalibrationTable, Int8CalibrationTable)):
        overlay["optimize"]["options"] = {"calibration_table": calibration}
    elif calibration is not None:
        overlay["precision"] = {"calibration": str(calibration)}
    if fast_sampling is not None:
        overlay["inference"] = {"fast_sampling": fast_sampling}
    return overlay
