"""XoFTR's :class:`~flashim.build.ModelSpec` (presets ``fp32`` / ``bf16`` / ``int8`` / ``fp8``)."""

from __future__ import annotations

from pathlib import Path

import torch
from torch import nn

from flashim.build import BuildContext, Pass
from flashim.config import RunConfig
from flashim.models.loftr.model import LoFTREncoderLayer
from flashim.models.sparse_spec import extractor_spec
from flashim.models.xoftr.model import XoFTR
from flashim.models.xoftr.quant import PROTECTED, QUANTIZED


def _load(config: RunConfig) -> XoFTR:
    return XoFTR.from_pretrained(checkpoint=config.checkpoint, device=config.device, **config.arch)


def _channels_last(model: nn.Module, ctx: BuildContext) -> str:
    """NHWC backbone weights for cuDNN convolutions; skipped for quantized presets (im2col + GEMM on NCHW)."""
    if ctx.config.precision.quant:
        return "skipped: quantized backbone"
    model.backbone.to(memory_format=torch.channels_last)
    return "backbone"


def _bf16_linear_attention(model: nn.Module, ctx: BuildContext) -> int | str:
    """BF16 operands for the coarse linear-attention GEMMs (skipped for ``fp32`` reference numerics)."""
    if ctx.config.precision.compute_dtype == "fp32":
        return "skipped: fp32"
    layers = [m for m in model.modules() if isinstance(m, LoFTREncoderLayer)]
    for layer in layers:
        layer.attention_gemm_dtype = torch.bfloat16
    return len(layers)


def _fold_similarity_temperature(model: nn.Module, ctx: BuildContext) -> str:
    """Fold ``1 / temperature`` into the left operand of coarse similarity (skipped for ``fp32``)."""
    if ctx.config.precision.compute_dtype == "fp32":
        return "skipped: fp32"
    model.fold_temperature = True
    return "coarse matching"


SPEC = extractor_spec(
    name="xoftr",
    description="XoFTR cross-modal (visible-thermal) detector-free matcher; match(a, b) only",
    load=_load,
    calibration_dir=Path(__file__).parent / "calibration",
    quantized=QUANTIZED,
    protected=PROTECTED,
    compile_regions=("backbone",),
    role="dense",
    extra_passes={
        "channels_last": Pass(_channels_last, description="NHWC backbone convolutions (bf16 / fp32)", exact=False),
        "bf16_linear_attention": Pass(
            _bf16_linear_attention, description="linear-attention GEMMs on BF16 tensor cores (not fp32)", exact=False
        ),
        "fold_similarity_temperature": Pass(
            _fold_similarity_temperature, description="coarse-similarity temperature folded into an operand", exact=False
        ),
    },
    structural_passes=("channels_last", "bf16_linear_attention", "fold_similarity_temperature"),
)
