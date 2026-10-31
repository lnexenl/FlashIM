"""``loma-matcher``: LoMa's matcher spec (variant b128 | b | l | g | r; presets ``fp32`` / ``bf16`` / ``int8`` /
``fp8``).

The quantized presets are LightGlue's (same module tree): ``bf16`` + every transformer linear as static MXFP8 / INT8,
positions, ``input_proj`` and the assignment heads protected; one calibration table per variant and format
(``calibration/loma_<variant>_<fmt>_mega.safetensors``, ``scripts/calibrate_lightglue.py --pipeline loma-<variant>``).
``int8`` also runs the ``rotate_residual_stream`` pass (Hadamard rotations folded into the weights,
:func:`~flashim.models.lightglue.quant.rotate_residual_stream`) and its tables smooth ``ffn.3``'s input
(``ROTATED_INT8_SMOOTHING``).
``bf16`` keeps positions, rotary math and the assignment softmaxes in FP32 (upstream's autocast runs the positional
projection in BF16 too)."""

from __future__ import annotations

import dataclasses
from pathlib import Path

from flashim.build import EngineSpec, ModelSpec, Pass
from flashim.config import RunConfig
from flashim.models.lightglue import quant as lightglue_quant
from flashim.models.lightglue.spec import _regions, _set_attention, _set_precision
from flashim.models.loma.matcher import VARIANTS, LoMaMatcher

CALIBRATION_DIR = Path(__file__).parent / "calibration"
TRANSFORMER_LINEARS = lightglue_quant.TRANSFORMER_LINEARS
PROTECTED = lightglue_quant.PROTECTED
QUANT_CONFIGS = {name: lightglue_quant.PRESETS[name].config for name in ("int8", "fp8")}


def _load(config: RunConfig) -> LoMaMatcher:
    variant = config.variant or "b"
    if variant not in VARIANTS:
        raise ValueError(f"Unknown variant {variant!r}; choose from {sorted(VARIANTS)}")
    arch = dataclasses.replace(VARIANTS[variant], **config.arch)
    return LoMaMatcher.from_variant(variant, arch, checkpoint=config.checkpoint, device=config.device)


PASSES = {
    "rotate_residual_stream": Pass(
        lambda model, ctx: lightglue_quant.rotate_residual_stream(model), stage="pre_quant", exact=False,
        description="Hadamard rotations folded into the weights (int8: evens out the quantized inputs' channels)",
    ),
}

SPEC = ModelSpec(
    name="loma-matcher",
    role="matcher",
    description="LoMa matcher (LightGlue-style, DaD keypoints + LoMa descriptors): SparseFeatures x "
    "SparseFeatures -> matches (variant b128 | b | l | g | r)",
    load=_load,
    presets={
        "fp32": {},
        "bf16": {"precision": {"compute_dtype": "bf16"}},
        "fp8": {"extends": "bf16", "precision": {"quant": "fp8"}},
        "int8": {"extends": "bf16", "precision": {"quant": "int8"}, "optimize": {"passes": ["rotate_residual_stream"]}},
    },
    default_preset="bf16",
    quant_configs=QUANT_CONFIGS,
    calibrations={  # without a variant: B's
        **{name: CALIBRATION_DIR / f"loma_b_{name}_mega.safetensors" for name in QUANT_CONFIGS},
        **{f"{variant}/{name}": CALIBRATION_DIR / f"loma_{variant}_{name}_mega.safetensors"
           for variant in VARIANTS for name in QUANT_CONFIGS},
    },
    protected=PROTECTED,
    passes=PASSES,
    engines={"eager": EngineSpec(), "compile": EngineSpec(_regions, recompile_limit=64)},
    set_precision=_set_precision,
    set_attention=_set_attention,
)
