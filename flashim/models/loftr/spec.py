"""LoFTR's :class:`~flashim.build.ModelSpec` (presets ``fp32`` / ``bf16`` / ``int8`` / ``fp8``); variant ``outdoor``
(MegaDepth, default) or ``indoor`` (ScanNet)."""

from __future__ import annotations

from pathlib import Path

import torch
from torch import nn

from flashim.build import BuildContext, Pass
from flashim.config import RunConfig
from flashim.models.loftr.model import LoFTR, LoFTREncoderLayer
from flashim.models.sparse_spec import extractor_spec

# Quantized: the backbone's 3x3 / 1x1 convs after the 1-channel 7x7 stem and every transformer linear (q/k/v/merge/MLP,
# coarse and fine). Protected: the stem and the coarse -> fine feature projections (``fine_preprocess``), which feed the
# fine offsets directly.
_LINEARS = ("q_proj", "k_proj", "v_proj", "merge", "mlp.0", "mlp.2")
QUANTIZED = (
    "backbone.layer[123].[01].conv[12]", "backbone.layer[123].[01].downsample.0", "backbone.layer[123]_outconv",
    "backbone.layer[12]_outconv2.[03]",
    *(f"{stage}.layers.*.{leaf}" for stage in ("loftr_coarse", "loftr_fine") for leaf in _LINEARS),
)
PROTECTED = ("backbone.conv1", "fine_preprocess.*")


def _load(config: RunConfig) -> LoFTR:
    return LoFTR.from_pretrained(variant=config.variant or "outdoor", checkpoint=config.checkpoint, device=config.device, **config.arch)


def _channels_last(model: nn.Module, ctx: BuildContext) -> str:
    """NHWC backbone weights for the cuDNN convolutions: measured 24.0 -> 21.2 ms per 896x1200 image in BF16 and
    43.0 -> 39.7 ms in FP32 on sm_120 (the NCHW <-> NHWC conversions around every conv go away); MegaDepth-1500 AUC@5
    0.6657 vs 0.6638 without it (4 seeds, std 0.0005). Skipped for the
    quantized presets, whose backbone convs are im2col + GEMM modules on NCHW buffers."""
    if ctx.config.precision.quant:
        return "skipped: quantized backbone"
    model.backbone.to(memory_format=torch.channels_last)
    return "backbone"


def _bf16_linear_attention(model: nn.Module, ctx: BuildContext) -> int | str:
    """BF16 operands for the linear-attention GEMMs of every transformer layer (see ``linear_attention``): measured
    75.4 -> 71.2 ms per 1200 px pair; MegaDepth-1500 AUC@5 0.6650 vs 0.6657 without it (4 seeds, std 0.0005). Skipped
    for ``fp32`` (the reference numerics)."""
    if ctx.config.precision.compute_dtype == "fp32":
        return "skipped: fp32"
    layers = [m for m in model.modules() if isinstance(m, LoFTREncoderLayer)]
    for layer in layers:
        layer.attention_gemm_dtype = torch.bfloat16
    return len(layers)


SPEC = extractor_spec(
    name="loftr",
    description="LoFTR detector-free matcher (variant outdoor | indoor); match(a, b) only",
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
    },
    structural_passes=("channels_last", "bf16_linear_attention"),
)
