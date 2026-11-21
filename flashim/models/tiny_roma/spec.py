"""Tiny-RoMa's :class:`~flashim.build.ModelSpec` (presets ``fp32`` / ``bf16`` / ``int8`` / ``fp8``)."""

from __future__ import annotations

from flashim.build import BuildContext, EngineSpec, ModelSpec, Pass, dtype
from flashim.config import RunConfig
from flashim.models.tiny_roma import quant
from flashim.models.tiny_roma.config import TinyRoMaConfig
from flashim.models.tiny_roma.model import TinyRoMa, apply_inference

COMPILE_REGIONS = ("xfeat.0", "coarse_matcher", "fine_matcher")


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
        "bf16": {"extends": "fp32", "precision": {"compute_dtype": "bf16"}},
        "int8": {
            "extends": "bf16",
            "precision": {"quant": "int8"},
            "optimize": {"passes": ["fold_batchnorm", "smooth_coord_inputs"]},
        },
        "fp8": {"extends": "bf16", "precision": {"quant": "fp8"}},
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
    },
    engines={
        "eager": EngineSpec(),
        "compile": EngineSpec(COMPILE_REGIONS, recompile_limit=64),
    },
    set_precision=_set_precision,
    apply_inference=apply_inference,
)
