"""EfficientLoFTR's :class:`~flashim.build.ModelSpec` (presets ``fp32`` / ``bf16`` / ``int8`` / ``fp8``); variant
``full`` (dual-softmax coarse matching, default) or ``opt`` (skip-softmax, faster)."""

from __future__ import annotations

from dataclasses import replace

import torch
from torch import nn

from flashim.build import BuildContext, Pass
from flashim.config import RunConfig
from flashim.models.eloftr import quant
from flashim.models.eloftr.config import SETTINGS
from flashim.models.eloftr.model import AggregatedEncoderLayer, ELoFTR
from flashim.models.sparse_spec import extractor_spec


def _load(config: RunConfig) -> ELoFTR:
    return ELoFTR.from_pretrained(
        variant=config.variant or "full",
        checkpoint=config.checkpoint,
        device=config.device,
        **config.arch,
    )


def _validate(config: RunConfig) -> None:
    if config.variant is not None and config.variant not in SETTINGS:
        raise ValueError(f"eloftr: unknown variant {config.variant!r}; choose from {sorted(SETTINGS)}")


def _reparameterize(model: nn.Module, ctx: BuildContext) -> list[str]:
    """Fuse RepVGG 3x3 + 1x1 + identity branches and BatchNorms into one 3x3 conv; quantized presets keep the
    identity BN scale out of ``layer2`` / ``layer3`` conv weights and add it in the quantized conv epilogue."""
    if ctx.config.precision.quant in ("int8", "fp8"):
        return model.reparameterize(keep_identity_stages=("layer2", "layer3"))
    return model.reparameterize()


def _pack_kv(model: nn.Module, ctx: BuildContext) -> list[str] | str:
    """Pack ``k_proj`` + ``v_proj`` into ``kv_proj`` on every coarse transformer layer. Skipped for ``fp32``."""
    if ctx.config.precision.compute_dtype == "fp32":
        return "skipped: fp32"
    return model.pack_kv()


def _fuse_coarse_mlp(model: nn.Module, ctx: BuildContext) -> list[str] | str:
    """Fold ``merge`` into ``mlp_m`` and split ``mlp[0]`` into ``mlp_x`` + ``mlp_m`` for fused NHWC coarse blocks.
    Skipped for ``fp32`` (reference numerics)."""
    if ctx.config.precision.compute_dtype == "fp32":
        return "skipped: fp32"
    return model.fuse_coarse_mlp()


def _backbone_channels_last(model: nn.Module, ctx: BuildContext) -> str:
    """NHWC weights + input on unquantized ``backbone`` convs (all of ``backbone`` in ``bf16``, ``layer0`` + ``layer1``
    in ``int8`` / ``fp8``) and ``fine_preprocess`` convs (enabling fused NHWC upsample+add). Skipped for ``fp32``."""
    if ctx.config.precision.compute_dtype == "fp32":
        return "skipped: fp32"
    for module in model.backbone.modules():
        if isinstance(module, nn.Conv2d):
            module.to(memory_format=torch.channels_last)
    for module in model.fine_preprocess.modules():
        if isinstance(module, nn.Conv2d):
            module.to(memory_format=torch.channels_last)
    model.backbone_channels_last = True
    return "backbone + fine_preprocess"


def _tc_coarse_similarity(model: nn.Module, ctx: BuildContext) -> str:
    """Run coarse similarity GEMM on BF16 tensor cores with FP32 output (``ops.matmul_nt_fp32`` for ``full``,
    ``torch.mm`` + 1D confidence scaling for ``opt``). Skipped for ``fp32`` (reference numerics)."""
    if ctx.config.precision.compute_dtype == "fp32":
        return "skipped: fp32"
    model.use_tc_similarity = True
    return "coarse matching"


def _set_attention(model: nn.Module, impl: str) -> None:
    target = "reference" if impl == "sdpa" else impl
    for module in model.modules():
        if isinstance(module, AggregatedEncoderLayer):
            module.sdpa_impl = target


_BASE = extractor_spec(
    name="eloftr",
    description="EfficientLoFTR detector-free matcher (variant full | opt); match(a, b) only",
    load=_load,
    calibration_dir=quant.CALIBRATION_DIR,
    quantized=quant.QUANTIZED,
    protected=quant.PROTECTED,
    int8_dynamic=quant.INT8_DYNAMIC,
    # Compiling backbone/fine_preprocess regressed BF16 (16.49 vs 12.63 ms) and hit a Dynamo @triton.autotune bug.
    compile_regions=("loftr_coarse",),
    role="dense",
    extra_passes={
        "reparameterize": Pass(
            _reparameterize,
            stage="pre_quant",
            description="fuse RepVGG branches into one conv (split-identity residual for quantized presets)",
        ),
        "pack_kv": Pass(
            _pack_kv,
            stage="pre_quant",
            description="pack coarse k_proj + v_proj into kv_proj",
            exact=False,
        ),
        "fuse_coarse_mlp": Pass(
            _fuse_coarse_mlp,
            stage="pre_quant",
            description="fold coarse merge into mlp_m and split mlp.0 for fused NHWC block",
            exact=False,
        ),
        "backbone_channels_last": Pass(
            _backbone_channels_last,
            description="NHWC convolutions on unquantized backbone + fine_preprocess",
            exact=False,
        ),
        "tc_coarse_similarity": Pass(
            _tc_coarse_similarity,
            description="BF16 tensor-core coarse similarity with fused temperature",
            exact=False,
        ),
    },
    structural_passes=(
        "reparameterize",
        "pack_kv",
        "fuse_coarse_mlp",
        "backbone_channels_last",
        "tc_coarse_similarity",
    ),
)

_QUANT_CONFIGS = {
    **_BASE.quant_configs,
    "fp8": replace(_BASE.quant_configs["fp8"], ignore=quant.FP8_IGNORE),
}

SPEC = replace(_BASE, quant_configs=_QUANT_CONFIGS, set_attention=_set_attention, validate=_validate)
