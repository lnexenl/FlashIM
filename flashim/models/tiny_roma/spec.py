"""Tiny-RoMa's :class:`~flashim.build.ModelSpec` (presets ``fp32`` / ``bf16`` / ``int8`` / ``fp8``)."""

from __future__ import annotations

import torch
from torch import nn

from flashim.build import BuildContext, EngineSpec, ModelSpec, Pass, dtype
from flashim.config import RunConfig
from flashim.models.tiny_roma import quant
from flashim.models.tiny_roma.config import TinyRoMaConfig
from flashim.models.tiny_roma.model import TinyRoMa, apply_inference


def _compile_regions(model: TinyRoMa) -> tuple[str, ...]:
    """Compile the BF16/FP32 conv stacks. Quantized ``coarse_matcher`` layers call the same ``@triton.autotune``
    kernel four times, which triggers a PyTorch 2.11 Dynamo side-effect tracking bug when compiled as one graph
    (and is already fused in Triton), so quantized presets compile ``xfeat.0`` and ``fine_matcher``."""
    from flashim.quant.methods.conv2d import StaticInt8Conv2d, StaticMXFP8Conv2d

    if any(isinstance(m, (StaticInt8Conv2d, StaticMXFP8Conv2d)) for m in model.coarse_matcher.modules()):
        return ("xfeat.0", "fine_matcher")
    return ("xfeat.0", "coarse_matcher", "fine_matcher")


def _load(config: RunConfig) -> TinyRoMa:
    return TinyRoMa.from_pretrained(
        TinyRoMaConfig(**config.arch), checkpoint=config.checkpoint, device=config.device
    )


def _fold_batchnorm(model: TinyRoMa, ctx: BuildContext) -> list[str]:
    model.fused_batchnorm = model.fold_batchnorm()
    return model.fused_batchnorm


def _smooth_coord_inputs(model: TinyRoMa, ctx: BuildContext) -> dict[str, float]:
    """SmoothQuant scale on the 2 normalized warp coordinate channels entering ``coarse_matcher.0``:
    scales ``coarse_warp`` by 32x and divides ``coarse_matcher.0.layer.0.weight[:, -2:]`` by 32x (exact
    equivalence in float; eliminates the 80x weight outlier ratio so INT8 keeps BF16 fidelity)."""
    scale = quant.COARSE_COORD_SCALE
    conv = model.coarse_matcher[0].layer[0]
    conv.weight.data[:, -2:].div_(scale)
    model.coarse_coord_scale = scale
    return {"coarse_coord_scale": scale}


def _channels_last(model: TinyRoMa, ctx: BuildContext) -> int | str:
    """NHWC weights for every unquantized BF16 ``nn.Conv2d`` (backbone, ``fine_matcher``, and ``coarse_matcher``
    in ``bf16``): avoids cuDNN NCHW <-> NHWC reformat kernels and makes ``tiny_roma_corr_pos_embed``'s feature
    permutations zero-copy. Skipped for ``fp32`` (keeps oracle reference numerics)."""
    if ctx.config.precision.compute_dtype == "fp32":
        return "skipped: fp32"
    count = 0
    for mod in model.modules():
        if isinstance(mod, nn.Conv2d) and mod.weight.dtype != torch.float32:
            mod.weight.data = mod.weight.data.to(memory_format=torch.channels_last)
            count += 1
    model.xfeat[0].channels_last = True
    return count


def _set_precision(model: TinyRoMa, config: RunConfig) -> None:
    if config.precision.compute_dtype != "fp32":
        model.set_compute_dtype(dtype(config.precision.compute_dtype))


SPEC = ModelSpec(
    name="tiny-roma",
    role="dense",
    description="Tiny-RoMa dense matcher (XFeat backbone + coarse/fine conv refiners, tiny_roma_v1_outdoor)",
    load=_load,
    presets={
        "fp32": {"optimize": {"passes": ["fold_batchnorm"]}},
        "bf16": {"extends": "fp32", "precision": {"compute_dtype": "bf16"}, "optimize": {"passes": ["fold_batchnorm", "channels_last"]}},
        "int8": {
            "extends": "bf16",
            "precision": {"quant": "int8"},
            "optimize": {"passes": ["fold_batchnorm", "smooth_coord_inputs", "channels_last"]},
        },
        "fp8": {
            "extends": "bf16",
            "precision": {"quant": "fp8"},
            "optimize": {"passes": ["fold_batchnorm", "channels_last"]},
        },
    },
    default_preset="bf16",
    quant_configs=quant.QUANT_CONFIGS,
    calibrations=quant.CALIBRATIONS,
    protected=quant.PROTECTED,
    passes={
        "fold_batchnorm": Pass(_fold_batchnorm, stage="pre_quant", description="exact conv+BN fold"),
        "smooth_coord_inputs": Pass(
            _smooth_coord_inputs, stage="pre_quant", description="SmoothQuant 32x coordinate scaling on coarse_matcher.0"
        ),
        "channels_last": Pass(_channels_last, stage="post_quant", exact=False, description="NHWC BF16 convolutions"),
    },
    engines={
        "eager": EngineSpec(),
        "compile": EngineSpec(_compile_regions, recompile_limit=64),
    },
    set_precision=_set_precision,
    apply_inference=apply_inference,
)
