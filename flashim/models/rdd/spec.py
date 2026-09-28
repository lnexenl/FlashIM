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


def _fuse_query_projections(model: nn.Module, ctx: BuildContext) -> int:
    from flashim.layers import MSDeformAttn

    model.fused_query_projections = sum(m.fuse_query_projections() for m in model.modules() if isinstance(m, MSDeformAttn))
    return model.fused_query_projections


_BASE = {"optimize": {"passes": ["fuse_query_projections"]}}
_BF16 = {"extends": "fp32", "precision": {"compute_dtype": "bf16"}}
PRESETS = {
    "fp32": _BASE,
    "bf16": _BF16,
    **{name: {"extends": "bf16", "precision": {"quant": name}} for name in ("fp8", "fp8-full", "fp8-backbone")},
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
        "fuse_query_projections": Pass(
            _fuse_query_projections, description="one GEMM for sampling offsets + attention weights (non-quantized)"
        ),
    },
    engines={"eager": EngineSpec(), "compile": EngineSpec(COMPILE_REGIONS, recompile_limit=64)},
    set_precision=_set_precision,
)


def legacy_overrides(
    *,
    engine: str = "eager",
    device=None,
    checkpoint=None,
    calibration=None,
    strict: bool = False,
    config: RDDConfig | None = None,
    compile_mode: str | None = "max-autotune-no-cudagraphs",
    detector_dtype=None,
) -> dict:
    """``load_rdd`` keyword arguments -> a config overlay."""
    from flashim.quant.calibration import CalibrationTable

    overlay: dict = {"engine": {"kind": engine, "mode": compile_mode}, "optimize": {"strict": strict}}
    if device is not None:
        overlay["device"] = str(device)
    if checkpoint is not None:
        overlay["checkpoint"] = str(checkpoint)
    if config is not None:
        overlay["arch"] = dataclasses.asdict(config)
    if isinstance(calibration, CalibrationTable):
        overlay["optimize"]["options"] = {"calibration_table": calibration}
    elif calibration is not None:
        overlay["precision"] = {"calibration": str(calibration)}
    if detector_dtype is not None:
        name = {"float32": "fp32", "bfloat16": "bf16", "float16": "fp16"}[str(detector_dtype).removeprefix("torch.")]
        overlay.setdefault("precision", {})["overrides"] = {"detector": name}
    return overlay
