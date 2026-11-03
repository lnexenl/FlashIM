"""Spec factory for the sparse extractors: the four precision presets (fp32 / bf16 / int8 / fp8) every one of them
ships, built from a model's list of quantizable convolution / linear globs.

- ``fp32``: checkpoint precision (matches the upstream implementation). BatchNorm folding (exact) is on when the model
  has any.
- ``bf16``: trunk and heads in BF16; keypoint selection, softmax / sampling math and the model's ``fp32_modules``
  stay FP32.
- ``int8`` / ``fp8``: ``bf16`` + the ``quantized`` convolutions / linears as static INT8 (per-channel weights, one
  calibrated per-tensor activation scale) or static MXFP8 (E4M3 + E8M0 block-32 scales), as im2col + GEMM for
  convolutions (:mod:`flashim.quant.methods.conv2d`). ``int8_ignore`` removes layers from the INT8 preset only;
  ``int8_dynamic`` quantizes linears there with per-token dynamic activation scales instead of the static one (e.g. a
  ViT whose outlier tokens one per-tensor scale cannot represent; no calibration entries).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import torch
from torch import nn

from flashim.build import BuildContext, EngineSpec, ModelSpec, Pass, dtype
from flashim.config import RunConfig
from flashim.quant.scheme import (
    INT8_ACTIVATIONS,
    INT8_TOKEN_ACTIVATIONS,
    INT8_WEIGHTS,
    MXFP8,
    QuantConfig,
    QuantScheme,
)


def _set_precision(model: nn.Module, config: RunConfig) -> None:
    if config.precision.compute_dtype != "fp32":
        model.set_compute_dtype(dtype(config.precision.compute_dtype))


def _channels_last(model: nn.Module, ctx: BuildContext) -> str:
    """NHWC weights, so cuDNN runs every convolution without NCHW <-> NHWC conversions and depthwise convolutions leave
    PyTorch's native NCHW kernel (59.7 of 154 ms in DeDoDe's BF16 extract). Skipped for ``fp32`` (the reference
    numerics: cuDNN picks other algorithms in NHWC, which moved 1.3% of SuperPoint's keypoints past the oracle parity
    tolerance) and for the quantized presets, whose convolutions are im2col + GEMM modules (already NHWC activations)."""
    if ctx.config.precision.compute_dtype == "fp32":
        return "skipped: fp32"
    if ctx.config.precision.quant:
        return "skipped: quantized convolutions"
    model.to(memory_format=torch.channels_last)
    return "model"


def _fused_bf16_blocks(prefixes: Sequence[str]):
    def apply(model: nn.Module, ctx: BuildContext) -> list[str] | str:
        """Separable conv blocks (depthwise 5x5 -> ReLU -> 1x1, BatchNorm folded) under ``prefixes`` as one BF16
        kernel each, for the channel counts the fused op has a kernel for. Runs before quantization, so in the int8 /
        fp8 presets these blocks are BF16 as well (their 1x1 convs leave the quantized set; the calibration entries
        go unused): measured per block on sm_120, the BF16 fused kernel is 0.234 ms at 32 ch / 896x1200 against
        0.380 ms (INT8 producer + GEMM) and 0.468 ms (MXFP8), and 0.325 ms at 128 ch / 448x600 against 0.297 ms
        for the fused INT8 block, which has no 64-channel kernel. ``fp32`` keeps the reference numerics."""
        from flashim.ops.conv import DEPTHWISE5X5_RELU_POINTWISE_CHANNELS
        from flashim.quant.passes.separable_conv_block import fuse_bf16_blocks

        if ctx.config.precision.compute_dtype == "fp32":
            return "skipped: fp32"
        return fuse_bf16_blocks(model, tuple(prefixes), channels=DEPTHWISE5X5_RELU_POINTWISE_CHANNELS)

    return apply


def extractor_spec(
    *,
    name: str,
    description: str,
    load,
    calibration_dir: Path,
    quantized: Sequence[str],
    protected: Sequence[str] = (),
    int8_ignore: Sequence[str] = (),
    int8_dynamic: Sequence[str] = (),
    compile_regions: Sequence[str] = (),
    has_batchnorm: bool = True,
    extra_passes: Mapping[str, Pass] | None = None,
    extra_presets: Mapping[str, Mapping[str, Any]] | None = None,
    default_preset: str = "bf16",
    role: str = "extractor",
    structural_passes: Sequence[str] = (),
    channels_last: bool = False,
    fused_bf16_blocks: Sequence[str] = (),
) -> ModelSpec:
    def fold(model: nn.Module, ctx: BuildContext) -> list[str]:
        model.fused_batchnorm = model.fold_batchnorm()
        return model.fused_batchnorm

    passes = {"fold_batchnorm": Pass(fold, stage="pre_quant", description="exact conv+BN fold")} if has_batchnorm else {}
    passes.update(extra_passes or {})
    if channels_last:  # opt in per model with a measurement: it is slower on DISK (28 -> 39 ms) and ALIKE (6.3 -> 7.8 ms)
        passes["channels_last"] = Pass(_channels_last, description="NHWC convolutions (bf16)", exact=False)
        structural_passes = (*structural_passes, "channels_last")
    if fused_bf16_blocks:  # module-name prefixes of ``flashim.layers.SeparableConvBlock`` decoders
        passes["fused_bf16_blocks"] = Pass(
            _fused_bf16_blocks(fused_bf16_blocks), stage="pre_quant", exact=False,
            description="fused BF16 separable conv blocks (bf16 / int8 / fp8)",
        )
        structural_passes = (*structural_passes, "fused_bf16_blocks")
    always = ([*(["fold_batchnorm"] if has_batchnorm else []), *structural_passes])
    fp32 = {"optimize": {"passes": always}} if always else {}
    quant_configs = {
        "int8": QuantConfig(
            {  # first match wins: the dynamic group before the static one
                **({"dynamic": QuantScheme(tuple(int8_dynamic), INT8_WEIGHTS, INT8_TOKEN_ACTIVATIONS)} if int8_dynamic else {}),
                "conv": QuantScheme(tuple(quantized), INT8_WEIGHTS, INT8_ACTIVATIONS),
            },
            ignore=tuple(int8_ignore), name="int8",
        ),
        "fp8": QuantConfig({"conv": QuantScheme(tuple(quantized), MXFP8, MXFP8)}, ignore=(), name="fp8"),
    }
    presets = {
        "fp32": fp32,
        "bf16": {"extends": "fp32", "precision": {"compute_dtype": "bf16"}},
        "int8": {"extends": "bf16", "precision": {"quant": "int8"}},
        "fp8": {"extends": "bf16", "precision": {"quant": "fp8"}},
        **(extra_presets or {}),
    }
    return ModelSpec(
        name=name,
        role=role,
        description=description,
        load=load,
        presets=presets,
        default_preset=default_preset,
        quant_configs=quant_configs,
        calibrations={q: Path(calibration_dir) / f"{name}_{q}.safetensors" for q in quant_configs},
        protected=tuple(protected),
        passes=passes,
        engines={"eager": EngineSpec(), "compile": EngineSpec(tuple(compile_regions), recompile_limit=64)},
        set_precision=_set_precision,
    )
