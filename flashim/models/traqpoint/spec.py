"""TraqPoint's :class:`~flashim.build.ModelSpec`; quantization presets and their rationale are in ``quant.py``."""

from __future__ import annotations

import dataclasses

import torch
from torch import nn

from flashim.build import BuildContext, EngineSpec, ModelSpec, Pass, dtype
from flashim.config import RunConfig
from flashim.models.traqpoint import quant
from flashim.models.traqpoint.backbone import ConvNeXtBlock
from flashim.models.traqpoint.config import TraqPointConfig
from flashim.models.traqpoint.model import TraqPoint

COMPILE_REGIONS = ("backbone", *(f"input_proj.{i}" for i in range(4)), "detector", "transformer")


def _load(config: RunConfig) -> TraqPoint:
    arch = dataclasses.replace(TraqPointConfig(), **config.arch) if config.arch else None
    return TraqPoint.from_pretrained(arch, checkpoint=config.checkpoint, device=config.device)


def _set_precision(model: TraqPoint, config: RunConfig) -> None:
    if config.precision.compute_dtype != "fp32":
        model.set_compute_dtype(dtype(config.precision.compute_dtype))
    detector = config.precision.overrides.get("detector", "fp32")
    if detector != "fp32":
        model.detector.set_dtype(dtype(detector))


def _fold_batchnorm(model: TraqPoint, ctx: BuildContext) -> list[str]:
    """Fold eval BatchNorm into its preceding Conv2d across ``detector`` and ``matchibility_head`` (exact)."""
    from flashim.models.rdd.model import BasicLayer
    from flashim.models.traqpoint.detector import ConvBlock, ResBlock

    if model.training:
        raise RuntimeError("fold_batchnorm requires model.eval()")
    folded = []
    for name, module in model.named_modules():
        if isinstance(module, (ConvBlock, ResBlock)) and isinstance(module.bn1, nn.BatchNorm2d):
            module.conv1 = nn.utils.fusion.fuse_conv_bn_eval(module.conv1, module.bn1)
            module.bn1 = nn.Identity()
            module.conv2 = nn.utils.fusion.fuse_conv_bn_eval(module.conv2, module.bn2)
            module.bn2 = nn.Identity()
            folded += [f"{name}.conv1", f"{name}.conv2"]
        elif isinstance(module, BasicLayer) and isinstance(module.layer[1], nn.BatchNorm2d):
            module.layer[0] = nn.utils.fusion.fuse_conv_bn_eval(module.layer[0], module.layer[1])
            module.layer[1] = nn.Identity()
            folded.append(f"{name}.layer.0")
    return folded


def _fold_convnext_gamma(model: TraqPoint, ctx: BuildContext) -> int | str:
    """Fold ConvNeXt ``gamma`` into ``pwconv2`` before BF16 cast / quantization (skipped in fp32)."""
    if ctx.config.precision.compute_dtype == "fp32":
        return "skipped: fp32"
    return sum(m.fold_gamma() for m in model.backbone.modules() if isinstance(m, ConvNeXtBlock))


def _channels_last(model: TraqPoint, ctx: BuildContext) -> str:
    """NHWC ConvNeXt + input_proj in bf16/int8/fp8: eliminates NCHW<->NHWC copies and runs 7x7 dwconv in cuDNN NHWC."""
    if ctx.config.precision.compute_dtype == "fp32":
        return "skipped: fp32"
    model.backbone.to(memory_format=torch.channels_last)
    model.backbone.channels_last = True
    model.input_proj.to(memory_format=torch.channels_last)
    return "backbone+input_proj"


def _fuse_query_projections(model: nn.Module, ctx: BuildContext) -> int:
    from flashim.layers import MSDeformAttn

    model.fused_query_projections = sum(m.fuse_query_projections() for m in model.modules() if isinstance(m, MSDeformAttn))
    return model.fused_query_projections


_BASE = {"optimize": {"passes": ["fold_batchnorm", "fold_convnext_gamma", "channels_last", "fuse_query_projections"]}}
_BF16 = {"extends": "fp32", "precision": {"compute_dtype": "bf16"}}

PRESETS = {
    "fp32": _BASE,
    "bf16": _BF16,
    "int8": {"extends": "bf16", "precision": {"quant": "int8"}},
    "fp8": {"extends": "bf16", "precision": {"quant": "fp8"}},
}

SPEC = ModelSpec(
    name="traqpoint",
    role="extractor",
    description="TraqPoint sparse detector/descriptor (DINOv3 ConvNeXt-B + 4-level deformable transformer + RL detector)",
    load=_load,
    presets=PRESETS,
    default_preset="bf16",
    quant_configs={name: preset.config for name, preset in quant.PRESETS.items() if preset.config is not None},
    calibrations={name: preset.calibration for name, preset in quant.PRESETS.items() if preset.config is not None},
    protected=quant.PROTECTED,
    passes={
        "fold_batchnorm": Pass(
            _fold_batchnorm, stage="pre_quant", description="exact conv+BN fold (detector + matchibility_head)"
        ),
        "fold_convnext_gamma": Pass(
            _fold_convnext_gamma, stage="pre_quant", exact=False,
            description="fold ConvNeXtBlock.gamma into pwconv2 before quantization",
        ),
        "channels_last": Pass(
            _channels_last, exact=False, description="NHWC ConvNeXt + input_proj (bf16 / int8 / fp8)"
        ),
        "fuse_query_projections": Pass(
            _fuse_query_projections, description="one GEMM for sampling offsets + attention weights (non-quantized)"
        ),
    },
    engines={"eager": EngineSpec(), "compile": EngineSpec(COMPILE_REGIONS, recompile_limit=64)},
    set_precision=_set_precision,
)
