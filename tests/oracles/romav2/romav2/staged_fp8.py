# Vendored from https://github.com/Parskatt/RoMaV2 (romav2/staged_fp8.py @ 95c9968) as the reference implementation of FlashIM's parity tests.
# Copyright (c) 2025 Johan Edstedt. SPDX-License-Identifier: MIT (see LICENSE in this directory).
# Changes: none besides this header.

"""Opt-in, precision-safe staged MXFP8 conversions for RoMaV2 experiments.

The variants in this module run *after* :class:`romav2.fast_fp8.FastFP8Model`
has made its private model copy.  They deliberately have a narrow surface:
refiner stages and the DPT matcher head are the only mutable regions, while
coordinate-producing heads remain eager.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from torch import nn

from romav2.fast_fp8_refiner import FastFP8PointwiseConv, select_dpt_convolution_path
from romav2.full_fp8_runtime import FullFP8Calibration
from romav2.inference import (
    PortablePointwiseConv2d,
    optimize_refiner_for_inference,
    quantize_refiner_pointwise_for_inference,
)
from romav2.refiner_fused_mxfp8 import (
    FusedRefinerMXFP8Block,
    fuse_refiner_mxfp8_for_inference,
)

__all__ = [
    "REFINER_VARIANTS",
    "StagedFP8Report",
    "apply_refiner_variant",
    "apply_staged_variant",
    "refiner_variant_steps",
]


StagedFP8Variant = Literal[
    "refiner_s4_fused",
    "dpt_pointwise",
    "refiner_s2_pointwise",
    "refiner_s2_fused",
    "refiner_s1_pointwise",
    "refiner_s1_fused",
    "refiner_bf16_optimized",
]

REFINER_VARIANTS = (
    "none",
    "s4_fused",
    "s2_pointwise",
    "s2_fused",
    "s4_fused_s2_pointwise",
    "s4_fused_s2_fused",
    "s4_fused_s2_fused_s1_fused",
    "s4_fused_s2_s1_pointwise",
    "bf16_optimized",
)

_DPT_HEAD_PREFIX = "matcher.head."
_DPT_FINAL_OUTPUT_CONV = "matcher.head.scratch.output_conv2.2"


@dataclass(frozen=True)
class StagedFP8Report:
    """JSON-serializable facts about one staged conversion.

    ``selected`` contains the exact fully-qualified module paths changed by
    this invocation. ``skipped`` is populated only when already-converted
    paths make an otherwise valid variant a no-op.
    """

    variant: StagedFP8Variant
    selected: tuple[str, ...]
    layer_count: int
    backend: str | None
    skipped: tuple[str, ...] | None = None


def _is_refiner_block(module: nn.Module) -> bool:
    return all(
        hasattr(module, attribute)
        for attribute in ("conv_depthwise", "norm", "relu", "conv_pointwise")
    )


def _refiner_block_names(model: nn.Module) -> tuple[str, ...]:
    return tuple(
        name
        for name, module in model.named_modules()
        if name and _is_refiner_block(module)
    )


def _require_refiner_stage(model: nn.Module, stride: str) -> nn.Module:
    try:
        stage = model.get_submodule(f"refiners.{stride}")
    except AttributeError as error:
        raise ValueError(f"Model has no refiners.{stride} stage") from error
    if not any(_is_refiner_block(module) for module in stage.modules()):
        raise ValueError(f"refiners.{stride} contains no refiner blocks")
    return stage


def _is_eligible_dpt_pointwise(name: str, module: nn.Module) -> bool:
    """Return whether ``module`` is an explicitly safe DPT pointwise target.

    The final DPT convolution produces warp and confidence coordinates, so it
    is protected even though it is geometrically a 1x1 convolution.  Calling
    this helper with a name outside ``matcher.head`` always returns ``False``;
    this prevents a generic 1x1 convolution, including any refiner head, from
    being admitted accidentally.
    """

    return (
        name.startswith(_DPT_HEAD_PREFIX)
        and name != _DPT_FINAL_OUTPUT_CONV
        and select_dpt_convolution_path(module) == "fp8_pointwise"
    )


def _replace_module(model: nn.Module, name: str, replacement: nn.Module) -> None:
    parent_name, separator, child_name = name.rpartition(".")
    if not separator:
        raise ValueError(f"Cannot replace root module {name!r}")
    model.get_submodule(parent_name)._modules[child_name] = replacement


def _apply_refiner_s4_fused(model: nn.Module) -> StagedFP8Report:
    _require_refiner_stage(model, "4")
    block_names = _refiner_block_names(model)
    stage_prefix = "refiners.4."
    stage_block_names = tuple(
        name for name in block_names if name.startswith(stage_prefix)
    )
    if not stage_block_names and not any(
        name.startswith(stage_prefix)
        for name, module in model.named_modules()
        if isinstance(module, FusedRefinerMXFP8Block)
    ):
        raise ValueError("refiners.4 contains no refiner blocks")

    # This folds BatchNorm and installs channels-last weights for the copied
    # model only.  The subsequent explicit skip list is what prevents the
    # primitive from selecting any other C=512 structural block.
    optimize_refiner_for_inference(model, channels_last=True)
    already_fused = {
        name
        for name, module in model.named_modules()
        if isinstance(module, FusedRefinerMXFP8Block)
    }
    skip_names = tuple(
        name for name in block_names if not name.startswith(stage_prefix)
    )
    fuse_refiner_mxfp8_for_inference(model, skip_names=skip_names)
    selected = tuple(
        name
        for name, module in model.named_modules()
        if name.startswith(stage_prefix)
        and isinstance(module, FusedRefinerMXFP8Block)
        and name not in already_fused
    )
    skipped = tuple(name for name in already_fused if name.startswith(stage_prefix))
    return StagedFP8Report(
        variant="refiner_s4_fused",
        selected=selected,
        layer_count=len(selected),
        backend="mxfp8",
        skipped=skipped or None,
    )


def _apply_refiner_pointwise(
    model: nn.Module,
    stride: str,
    *,
    min_channels: int,
    variant: StagedFP8Variant,
) -> StagedFP8Report:
    stage = _require_refiner_stage(model, stride)
    prefix = f"refiners.{stride}"
    before = {
        f"{prefix}.{name}.conv_pointwise" if name else f"{prefix}.conv_pointwise"
        for name, module in stage.named_modules()
        if _is_refiner_block(module)
        and isinstance(module.conv_pointwise, PortablePointwiseConv2d)
    }
    report = quantize_refiner_pointwise_for_inference(
        stage, backend="mxfp8", min_channels=min_channels
    )
    selected = tuple(
        f"{prefix}.{name}"
        for name in report_names_for_refiner_pointwise(stage)
        if f"{prefix}.{name}" not in before
    )
    return StagedFP8Report(
        variant=variant,
        selected=selected,
        layer_count=len(selected),
        backend=report.backend,
        skipped=tuple(sorted(before)) or None,
    )


def _apply_refiner_s2_pointwise(model: nn.Module) -> StagedFP8Report:
    return _apply_refiner_pointwise(
        model, "2", min_channels=128, variant="refiner_s2_pointwise"
    )


def _apply_refiner_s1_pointwise(model: nn.Module) -> StagedFP8Report:
    return _apply_refiner_pointwise(
        model, "1", min_channels=32, variant="refiner_s1_pointwise"
    )


def _apply_refiner_stage_fused(
    model: nn.Module, stride: str, channels: int, variant: StagedFP8Variant
) -> StagedFP8Report:
    """Fuse one refiner stage with BF16 depthwise math and MXFP8 pointwise GEMM."""

    _require_refiner_stage(model, stride)
    block_names = _refiner_block_names(model)
    stage_prefix = f"refiners.{stride}."
    if not any(name.startswith(stage_prefix) for name in block_names):
        raise ValueError(f"refiners.{stride} contains no refiner blocks")
    optimize_refiner_for_inference(model, channels_last=True)
    already_fused = {
        name
        for name, module in model.named_modules()
        if name.startswith(stage_prefix)
        and isinstance(module, FusedRefinerMXFP8Block)
        and module.channels == channels
    }
    skip_names = tuple(
        name for name in block_names if not name.startswith(stage_prefix)
    )
    fuse_refiner_mxfp8_for_inference(model, skip_names=skip_names, channels=channels)
    selected = tuple(
        name
        for name, module in model.named_modules()
        if name.startswith(stage_prefix)
        and isinstance(module, FusedRefinerMXFP8Block)
        and module.channels == channels
        and name not in already_fused
    )
    return StagedFP8Report(
        variant=variant,
        selected=selected,
        layer_count=len(selected),
        backend="mxfp8",
        skipped=tuple(sorted(already_fused)) or None,
    )


def _apply_refiner_s2_fused(model: nn.Module) -> StagedFP8Report:
    return _apply_refiner_stage_fused(model, "2", 128, "refiner_s2_fused")


def _apply_refiner_s1_fused(model: nn.Module) -> StagedFP8Report:
    return _apply_refiner_stage_fused(model, "1", 32, "refiner_s1_fused")


def report_names_for_refiner_pointwise(stage: nn.Module) -> tuple[str, ...]:
    """Return paths selected by the stage-2 portable refiner primitive."""

    names: list[str] = []
    for name, module in stage.named_modules():
        if not _is_refiner_block(module):
            continue
        pointwise = module.conv_pointwise
        if isinstance(pointwise, PortablePointwiseConv2d):
            names.append(f"{name}.conv_pointwise" if name else "conv_pointwise")
    return tuple(names)


def _apply_dpt_pointwise(
    model: nn.Module, calibration: FullFP8Calibration
) -> StagedFP8Report:
    selected = tuple(
        (name, module)
        for name, module in model.named_modules()
        if _is_eligible_dpt_pointwise(name, module)
    )
    if not selected:
        already_converted = tuple(
            name
            for name, module in model.named_modules()
            if name.startswith(_DPT_HEAD_PREFIX)
            and isinstance(module, FastFP8PointwiseConv)
        )
        return StagedFP8Report(
            variant="dpt_pointwise",
            selected=(),
            layer_count=0,
            backend="mxfp8",
            skipped=already_converted or None,
        )

    try:
        convolution_calibration = calibration.convolution
    except AttributeError as error:
        raise TypeError("calibration must expose a convolution mapping") from error
    missing = tuple(name for name, _ in selected if name not in convolution_calibration)
    if missing:
        raise RuntimeError(
            "Staged FP8 calibration is missing DPT pointwise convolutions: "
            + ", ".join(missing)
        )

    # Construct every replacement before updating the module tree.  A bad
    # scale/device/backend therefore cannot leave a partially converted DPT.
    replacements = tuple(
        (
            name,
            FastFP8PointwiseConv.from_conv(module, convolution_calibration[name]),
        )
        for name, module in selected
    )
    for name, replacement in replacements:
        _replace_module(model, name, replacement)

    names = tuple(name for name, _ in replacements)
    return StagedFP8Report(
        variant="dpt_pointwise",
        selected=names,
        layer_count=len(names),
        backend="mxfp8",
    )


def _apply_refiner_bf16_optimized(model: nn.Module) -> StagedFP8Report:
    """Fold refiner Conv-BatchNorm pairs and use channels-last conv weights.

    This is the non-quantized control for ``refiner_s4_fused``. It does not
    replace refiner blocks, projections, or coordinate heads.
    """

    selected = _refiner_block_names(model)
    if not selected:
        raise ValueError("model contains no refiner blocks")
    optimize_refiner_for_inference(model, channels_last=True)
    return StagedFP8Report(
        variant="refiner_bf16_optimized",
        selected=selected,
        layer_count=len(selected),
        backend="bf16",
    )


def apply_staged_variant(
    model: nn.Module, calibration: FullFP8Calibration, variant: StagedFP8Variant
) -> StagedFP8Report:
    """Apply one opt-in staged FP8 experiment to a copied ``FastFP8Model``.

    ``model`` must be the ``.model`` held by an already-created
    :class:`romav2.fast_fp8.FastFP8Model` whose ``refiners`` subtree was kept
    in BF16.  No original model, coordinate head, warp/confidence path, or DPT
    final output convolution is modified.
    """

    if variant == "refiner_s4_fused":
        return _apply_refiner_s4_fused(model)
    if variant == "refiner_s2_pointwise":
        return _apply_refiner_s2_pointwise(model)
    if variant == "refiner_s2_fused":
        return _apply_refiner_s2_fused(model)
    if variant == "refiner_s1_pointwise":
        return _apply_refiner_s1_pointwise(model)
    if variant == "refiner_s1_fused":
        return _apply_refiner_s1_fused(model)
    if variant == "dpt_pointwise":
        return _apply_dpt_pointwise(model, calibration)
    if variant == "refiner_bf16_optimized":
        return _apply_refiner_bf16_optimized(model)
    raise ValueError(
        "unknown staged FP8 variant "
        f"{variant!r}; expected one of refiner_s4_fused, dpt_pointwise, "
        "refiner_s2_pointwise, refiner_s2_fused, refiner_s1_pointwise, "
        "refiner_s1_fused, refiner_bf16_optimized"
    )


def refiner_variant_steps(variant: str) -> tuple[StagedFP8Variant, ...]:
    """Map a CLI refiner token onto staged steps.

    ``s4_fused_s2_pointwise`` is stage-4 fusion then stage-2 pointwise.
    ``s4_fused_s2_s1_pointwise`` adds stage-1 pointwise after that. Neither
    combo token is a :data:`StagedFP8Variant`. Stage-1 blocks are C=32, which
    is one MXFP8 block; projections and coordinate heads stay BF16.
    ``bf16_optimized`` folds BatchNorm and enables channels-last refiner
    convolutions without quantizing them.
    """

    if variant == "none":
        return ()
    if variant == "s4_fused":
        return ("refiner_s4_fused",)
    if variant == "s2_pointwise":
        return ("refiner_s2_pointwise",)
    if variant == "s2_fused":
        return ("refiner_s2_fused",)
    if variant == "s4_fused_s2_pointwise":
        return ("refiner_s4_fused", "refiner_s2_pointwise")
    if variant == "s4_fused_s2_fused":
        return ("refiner_s4_fused", "refiner_s2_fused")
    if variant == "s4_fused_s2_fused_s1_fused":
        return ("refiner_s4_fused", "refiner_s2_fused", "refiner_s1_fused")
    if variant == "s4_fused_s2_s1_pointwise":
        return ("refiner_s4_fused", "refiner_s2_pointwise", "refiner_s1_pointwise")
    if variant == "bf16_optimized":
        return ("refiner_bf16_optimized",)
    allowed = ", ".join(REFINER_VARIANTS)
    raise ValueError(f"unknown refiner variant {variant!r}; expected one of {allowed}")


def apply_refiner_variant(
    model: nn.Module, calibration: FullFP8Calibration, variant: str
) -> tuple[StagedFP8Report, ...]:
    """Apply the staged FP8 steps for one CLI refiner variant.

    ``none`` returns an empty tuple and does not call
    :func:`apply_staged_variant`. Every other token calls that function once
    per step, in :func:`refiner_variant_steps` order.
    """

    return tuple(
        apply_staged_variant(model, calibration, step)
        for step in refiner_variant_steps(variant)
    )
