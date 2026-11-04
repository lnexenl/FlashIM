"""FlexGlue's spec: the sparse-to-dense matcher with its fp32 / bf16 / int8 / fp8 presets."""

from __future__ import annotations

import dataclasses

from flashim.build import EngineSpec, ModelSpec, dtype
from flashim.config import RunConfig
from flashim.models.flexglue import quant
from flashim.models.flexglue.config import FlexGlueConfig
from flashim.models.flexglue.model import FlexGlue


def _load(config: RunConfig) -> FlexGlue:
    arch = dataclasses.replace(FlexGlueConfig(), **config.arch)
    arch.validate()
    return FlexGlue.from_pretrained(arch, checkpoint=config.checkpoint, device=config.device)


def _set_precision(model: FlexGlue, config: RunConfig) -> None:
    if config.precision.compute_dtype != "fp32":
        model.set_compute_dtype(dtype(config.precision.compute_dtype))


def _set_attention(model: FlexGlue, impl: str) -> None:
    for layer in model.transformers:
        layer.self_attn.sdpa_impl = layer.cross_attn.sdpa_impl = impl


def _validate(config: RunConfig) -> None:
    if config.optimize.attention != "sdpa" and config.precision.compute_dtype == "fp32":
        raise ValueError(f"attention={config.optimize.attention!r} needs a bf16 / int8 / fp8 preset (BF16 attention "
                         "inputs); in fp32 it would silently run torch SDPA")


def _regions(model: FlexGlue) -> list[str]:
    return [
        "descriptor",
        "input_proj",
        *(f"transformers.{i}" for i in range(len(model.transformers))),
        *(f"heads.{i}" for i in range(len(model.heads))),
    ]


_QUANTIZED = {name: p for name, p in quant.PRESETS.items() if p.config is not None}

SPEC = ModelSpec(
    name="flexglue",
    role="matcher",
    description="FlexGlue sparse-to-dense matcher (DINOv3-ConvNeXt descriptor, 9 transformer layers, per-layer "
    "correspondence heads with a 1.5-entmax readout; matches keypoints or samples the dense warp)",
    load=_load,
    presets={
        "fp32": {},
        "bf16": {"precision": {"compute_dtype": "bf16"}},
        **{name: {"extends": "bf16", "precision": {"quant": name}} for name in _QUANTIZED},
    },
    default_preset="fp8",
    quant_configs={name: p.config for name, p in _QUANTIZED.items()},
    calibrations={name: p.calibration for name, p in _QUANTIZED.items()},
    protected=quant.PROTECTED,
    engines={"eager": EngineSpec(), "compile": EngineSpec(_regions, recompile_limit=64)},
    set_precision=_set_precision,
    set_attention=_set_attention,
    validate=_validate,
)
