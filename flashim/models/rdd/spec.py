"""RDD's :class:`~flashim.build.ModelSpec`; quantization presets and their rationale are in ``quant.py``."""

from __future__ import annotations

import dataclasses

from torch import nn

from flashim.build import BuildContext, EngineSpec, ModelSpec, Pass, dtype
from flashim.config import RunConfig
from flashim.models.rdd import quant
from flashim.models.rdd.config import RDDConfig
from flashim.models.rdd.model import RDD

# Inductor fuses the norms/activations/permutes around the (custom-op) GEMMs and the deformable kernel.
COMPILE_REGIONS = ("backbone", *(f"input_proj.{i}" for i in range(5)), "detector", "transformer")


def _load(config: RunConfig) -> RDD:
    arch = dataclasses.replace(RDDConfig(), **config.arch) if config.arch else None
    return RDD.from_pretrained(arch, checkpoint=config.checkpoint, device=config.device)


def _set_precision(model: RDD, config: RunConfig) -> None:
    if config.precision.compute_dtype != "fp32":
        model.set_compute_dtype(dtype(config.precision.compute_dtype))
    detector = config.precision.overrides.get("detector", "fp32")
    if detector != "fp32":  # the keypoint detector trunk; its score head stays FP32
        model.detector.set_dtype(dtype(detector))


def _fold_batchnorm(model: RDD, ctx: BuildContext) -> list[str]:
    """Fold eval BatchNorm into its preceding Conv2d across ``detector`` and ``matchibility_head`` (exact)."""
    from flashim.models.rdd.detector import ConvBlock, ResBlock
    from flashim.models.rdd.model import BasicLayer

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


def _fuse_query_projections(model: nn.Module, ctx: BuildContext) -> int:
    from flashim.layers import MSDeformAttn

    model.fused_query_projections = sum(m.fuse_query_projections() for m in model.modules() if isinstance(m, MSDeformAttn))
    return model.fused_query_projections


_BASE = {"optimize": {"passes": ["fold_batchnorm", "fuse_query_projections"]}}
_BF16 = {"extends": "fp32", "precision": {"compute_dtype": "bf16"}}
PRESETS = {
    "fp32": _BASE,
    "bf16": _BF16,
    "fp8": {"extends": "bf16", "precision": {"quant": "fp8"}},
}

SPEC = ModelSpec(
    name="rdd",
    role="extractor",
    description="RDD sparse detector/descriptor (ConvNeXt-tiny + multi-scale deformable transformer)",
    load=_load,
    presets=PRESETS,
    default_preset="fp8",
    quant_configs={name: preset.config for name, preset in quant.PRESETS.items() if preset.config is not None},
    calibrations={name: preset.calibration for name, preset in quant.PRESETS.items() if preset.config is not None},
    protected=quant.PROTECTED,
    passes={
        "fold_batchnorm": Pass(
            _fold_batchnorm, stage="pre_quant", description="exact conv+BN fold (detector + matchibility_head)"
        ),
        "fuse_query_projections": Pass(
            _fuse_query_projections, description="one GEMM for sampling offsets + attention weights (non-quantized)"
        ),
    },
    engines={"eager": EngineSpec(), "compile": EngineSpec(COMPILE_REGIONS, recompile_limit=64)},
    set_precision=_set_precision,
)

