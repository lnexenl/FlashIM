"""Generic model builder: :class:`ModelSpec` + :class:`~flashim.config.RunConfig` -> ready-to-run model.

Every model goes through the same stages, in this order::

    spec.load(config)                      weights -> module on the device (arch / variant / checkpoint)
    passes with stage "pre_quant"          e.g. BatchNorm folding
    spec.set_precision                     compute dtype + model-specific precision overrides
    quantize_model(quant config)           static MXFP8 / INT8 linears + structural fusions
    passes with stage "post_quant"         model-specific fusions, in the order the config lists them
    spec.set_attention                     ops.sdpa impl for attention layers
    engine                                 per-kind setup + torch.compile of the spec's regions
    spec.apply_inference                   per-call defaults

A pass whose platform requirement is not met is skipped (logged; ``optimize.strict`` raises instead). Presets are the
four :data:`PRECISIONS`; a model without the one asked for, or an FP8 one on a GPU without FP8 (below sm_89: Ampere,
Jetson Orin), builds the nearest it has (:data:`NEAREST`, see :func:`resolve`). The returned model carries
``model.run_config`` (the resolved :class:`RunConfig`: what is built) and ``model.build_report`` (:class:`BuildReport`:
what actually happened, the replaced presets included).
"""

from __future__ import annotations

import importlib
import logging
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import torch
from torch import nn

from flashim.config import RunConfig, load_file, merge, parse_overrides, resolve_preset
from flashim.log import get_logger
from flashim.platforms import Feature, Platform, default_device, platform_for

logger = get_logger("build")

DTYPES = {"fp32": torch.float32, "bf16": torch.bfloat16, "fp16": torch.float16}

# The presets every model chooses from: checkpoint precision, BF16 compute, static INT8 / MXFP8 quantization.
PRECISIONS = ("fp32", "bf16", "int8", "fp8")
# The preset built in place of one a model does not have, or of ``fp8`` on a GPU without FP8: the first of these the
# model has. INT8 MMA is native from sm_75 on; measured on a Jetson AGX Orin (sm_87), where FP8 runs on the torch
# references: RoMaV2 fp8 3334 ms against int8 1122 ms (bf16 1981 ms), LightGlue 209 against 56 ms, RDD (no INT8) fp8
# 1151 ms against bf16 398 ms.
NEAREST = {"fp8": ("int8", "bf16", "fp32"), "int8": ("bf16", "fp32"), "bf16": ("fp32",), "fp32": ("bf16",)}


@dataclass(frozen=True)
class Pass:
    """A named model transformation. ``apply(model, ctx)`` may return a value recorded in the report."""

    apply: Callable[[nn.Module, BuildContext], Any]
    stage: Literal["pre_quant", "post_quant"] = "post_quant"
    requires: Callable[[Platform], bool] | None = None
    description: str = ""
    exact: bool = True  # numerics unchanged (up to rounding order)?


@dataclass(frozen=True)
class EngineSpec:
    """``regions``: submodule paths to ``torch.compile`` (or a function of the model); ``setup`` runs first."""

    regions: Sequence[str] | Callable[[nn.Module], Sequence[str]] = ()
    setup: Callable[[nn.Module, BuildContext], None] | None = None
    recompile_limit: int | None = None


EAGER = EngineSpec()


@dataclass(frozen=True)
class ModelSpec:
    name: str
    role: Literal["dense", "extractor", "matcher", "pipeline"]
    description: str
    load: Callable[[RunConfig], nn.Module] | None
    presets: Mapping[str, Mapping[str, Any]]  # a subset of PRECISIONS -> config overlay
    default_preset: str
    quant_configs: Mapping[str, Any] = field(default_factory=dict)  # name -> QuantConfig
    calibrations: Mapping[str, Path] = field(default_factory=dict)  # quant (or "<variant>/<quant>") -> default table
    protected: tuple[str, ...] = ()
    passes: Mapping[str, Pass] = field(default_factory=dict)
    engines: Mapping[str, EngineSpec] = field(default_factory=lambda: {"eager": EAGER})
    set_precision: Callable[[nn.Module, RunConfig], None] | None = None
    set_attention: Callable[[nn.Module, str], None] | None = None
    apply_inference: Callable[[nn.Module, Mapping[str, Any]], None] | None = None
    validate: Callable[[RunConfig], None] | None = None
    # Composite models: build the components (``config.components``, already resolved) and combine them.
    compose: Callable[[RunConfig, Callable[[RunConfig], nn.Module]], nn.Module] | None = None

    def __post_init__(self) -> None:
        unknown = sorted(set(self.presets) - set(PRECISIONS))
        if unknown or self.default_preset not in self.presets:
            raise ValueError(f"{self.name}: presets must be among {PRECISIONS} and include the default, got "
                             f"{sorted(self.presets)} (default {self.default_preset!r})")

    def check(self, config: RunConfig) -> None:
        unknown = [p for p in config.optimize.passes if p not in self.passes]
        if unknown:
            raise ValueError(f"{self.name}: unknown pass(es) {unknown}; available: {sorted(self.passes)}")
        if config.precision.quant is not None and config.precision.quant not in self.quant_configs:
            raise ValueError(
                f"{self.name}: unknown quant config {config.precision.quant!r}; available: {sorted(self.quant_configs)}"
            )
        if self.compose is None and config.engine.kind not in self.engines:  # composites: components' engines
            raise ValueError(f"{self.name}: unknown engine {config.engine.kind!r}; choose from {sorted(self.engines)}")
        if self.validate is not None:
            self.validate(config)


@dataclass
class BuildReport:
    config_digest: str
    platform: str
    passes: dict[str, Any] = field(default_factory=dict)  # applied pass -> its return value
    skipped: dict[str, str] = field(default_factory=dict)  # skipped pass -> reason
    quant: dict[str, Any] | None = None
    emulated_quant: bool = False
    fallbacks: dict[str, str] = field(default_factory=dict)  # model / component path -> "asked -> built (why)" preset
    compiled_regions: list[str] = field(default_factory=list)
    components: dict[str, BuildReport] = field(default_factory=dict)


@dataclass
class BuildContext:
    config: RunConfig
    spec: ModelSpec
    platform: Platform
    report: BuildReport

    def option(self, name: str, default: Any = None) -> Any:
        return self.config.optimize.options.get(name, default)


# -- registry ---------------------------------------------------------------------------------------------

_SPEC_PATHS: dict[str, str] = {}
_SPECS: dict[str, ModelSpec] = {}


def register_spec(name: str, target: str | ModelSpec) -> None:
    """Register a model spec, or ``"package.module:ATTR"`` to import it lazily."""
    if isinstance(target, ModelSpec):
        _SPECS[name] = target
    else:
        _SPEC_PATHS[name] = target


def get_spec(name: str) -> ModelSpec:
    spec = _SPECS.get(name)
    if spec is None:
        if name not in _SPEC_PATHS:
            raise ValueError(f"Unknown model {name!r}; available: {sorted({*_SPECS, *_SPEC_PATHS})}")
        module, attr = _SPEC_PATHS[name].split(":")
        spec = _SPECS[name] = getattr(importlib.import_module(module), attr)
    return spec


def spec_names() -> list[str]:
    return sorted({*_SPECS, *_SPEC_PATHS})


# -- resolution -------------------------------------------------------------------------------------------

_PROPAGATED = ("device", "engine")  # pipeline-level keys applied to components that do not set them


def _keeps_fp8(user: Mapping[str, Any]) -> str | None:
    """Why a user config rules the FP8 fallback out (``None``: it does not): ``optimize.strict`` (build what was asked
    or raise), or a calibration table passed with a preset, which is an FP8 table. One answer for the whole config,
    components included: a pipeline's preset and its matcher's calibration table are replaced together or not at all."""
    optimize, precision = user.get("optimize") or {}, user.get("precision") or {}
    if optimize.get("strict"):
        return "optimize.strict"
    if precision.get("calibration") or (optimize.get("options") or {}).get("calibration_table") is not None:
        return "a calibration table was passed"
    return next(filter(None, map(_keeps_fp8, (user.get("components") or {}).values())), None)


def _nearest(spec: ModelSpec, preset: str, skip: str | None = None) -> str:
    return next(p for p in (preset, *NEAREST[preset]) if p in spec.presets and p != skip)


def _choose_preset(
    spec: ModelSpec, preset: str, platform: Platform | None, keep: str | None, strict: bool, component: bool
) -> tuple[str, str | None]:
    """The preset built for ``preset`` and why it differs (``None``: it does not): the nearest one ``spec`` has
    (``optimize.strict`` raises instead; logged at info level for a pipeline's ``component``, where it is the rule),
    and on a GPU without FP8 the nearest non-FP8 one, unless ``keep``. The CPU keeps FP8: it has no INT8 kernels
    either, and FP8 emulation there is how the numerics are checked without a GPU."""
    if preset not in PRECISIONS:
        raise ValueError(f"{spec.name}: unknown preset {preset!r}; choose from {PRECISIONS}")
    built, reason, level = _nearest(spec, preset), None, logging.WARNING
    if built != preset:
        if strict:
            raise ValueError(f"{spec.name}: no {preset!r} preset (optimize.strict); available: {sorted(spec.presets)}")
        reason, level = f"{spec.name} has no {preset}", logging.INFO if component else logging.WARNING
    if built == "fp8" and platform is not None and platform.device_type != "cpu" and not platform.supports(
        Feature.FP8_E4M3
    ):
        if keep is not None:
            logger.warning("%s: no FP8 on %s, preset 'fp8' kept (%s)", spec.name, platform.describe(), keep)
        else:
            built, reason, level = _nearest(spec, "fp8", skip="fp8"), f"no FP8 on {platform.describe()}", logging.WARNING
    if reason is not None:
        logger.log(level, "%s: preset %r replaced by %r (%s)", spec.name, preset, built, reason)
    return built, reason


def resolve(
    name: str,
    preset: str | None = None,
    config: RunConfig | Mapping[str, Any] | str | Path | None = None,
    overrides: Iterable[str] | Mapping[str, Any] = (),
    *,
    platform: Platform | None = None,
    fallbacks: dict[str, str] | None = None,
) -> RunConfig:
    """Resolve ``name``'s ``preset`` (one of :data:`PRECISIONS`; default: the spec's) + ``config`` (a RunConfig, dict
    or file) + dotted ``overrides`` into a validated :class:`RunConfig`.

    The model, and every component of a pipeline (the pipeline's preset is each component's), builds the nearest
    preset it has (:data:`NEAREST`; ``optimize.strict`` raises instead). With ``platform`` (:func:`load` passes the
    target device's), ``fp8`` is also replaced on a GPU without FP8, unless the config sets ``optimize.strict`` or
    passes a calibration table anywhere (:func:`_keeps_fp8`). ``preset`` of the result names the preset that is built;
    ``fallbacks`` (a dict) collects ``path -> "asked -> built (why)"``. Without ``platform`` the result does not depend
    on the machine: build it with :func:`build_config` to run an FP8 preset emulated on such a GPU."""
    user = merge(_user_overlay(config), parse_overrides(overrides))
    keep = None if platform is None else _keeps_fp8(user)
    return _resolve(name, preset, user, platform, keep, False, "", fallbacks)


def _user_overlay(config: RunConfig | Mapping[str, Any] | str | Path | None) -> dict[str, Any]:
    if isinstance(config, RunConfig):
        return config.to_dict()
    if isinstance(config, (str, Path)):
        return load_file(config)
    return dict(config) if config is not None else {}


def _resolve(
    name: str, preset: str | None, user: dict[str, Any], platform: Platform | None, keep: str | None, strict: bool,
    path: str, fallbacks: dict[str, str] | None,
) -> RunConfig:
    """``user``: everything layered over the preset (for a component: the pipeline preset's entry + the user's);
    ``strict``: the pipeline's ``optimize.strict``, which its components inherit unless they set their own."""
    spec = get_spec(name)
    user.pop("model", None)
    preset = preset or user.pop("preset", None) or spec.default_preset
    user.pop("preset", None)
    strict = bool((user.get("optimize") or {}).get("strict", strict))
    built, reason = _choose_preset(spec, preset, platform, keep, strict, component=bool(path))
    if reason is not None and fallbacks is not None:
        fallbacks[path or spec.name] = f"{preset} -> {built} ({reason})"
    data = merge(resolve_preset(spec.presets, built), user)
    user_components = user.get("components") or {}
    for component, sub in (data.get("components") or {}).items():
        sub = dict(sub)
        for key in _PROPAGATED:
            if key in user and key not in (user_components.get(component) or {}):
                sub[key] = merge(sub.get(key) or {}, user[key]) if isinstance(user[key], Mapping) else user[key]
        sub_name = sub.pop("model")
        data["components"][component] = _resolve(
            sub_name, sub.pop("preset", None), sub, platform, keep, strict, f"{path}.{component}".lstrip("."), fallbacks
        ).to_dict()
    if strict:
        data = merge(data, {"optimize": {"strict": True}})
    data.update(model=spec.name, preset=built)
    resolved = RunConfig.from_dict(data)
    spec.check(resolved)
    return resolved


# -- building ---------------------------------------------------------------------------------------------


def _load_calibration(spec: ModelSpec, config: RunConfig, int8: bool):
    from flashim.quant.calibration import CalibrationTable, Int8CalibrationTable

    table = config.optimize.options.get("calibration_table")  # an already-loaded table (not serializable)
    if table is not None:
        return table
    quant = config.precision.quant
    path = config.precision.calibration or spec.calibrations.get(f"{config.variant}/{quant}") or spec.calibrations.get(quant)
    if path is None:
        raise ValueError(f"{spec.name}: quant config {config.precision.quant!r} needs a calibration table")
    return (Int8CalibrationTable if int8 else CalibrationTable).load(path)


def _run_passes(model: nn.Module, ctx: BuildContext, stage: str) -> None:
    for name in ctx.config.optimize.passes:
        spec_pass = ctx.spec.passes[name]
        if spec_pass.stage != stage:
            continue
        if spec_pass.requires is not None and not spec_pass.requires(ctx.platform):
            reason = f"not supported on {ctx.platform.describe()}"
            if ctx.config.optimize.strict:
                raise RuntimeError(f"{ctx.spec.name}: pass {name!r} {reason}")
            logger.warning("%s: skipping pass %r (%s)", ctx.spec.name, name, reason)
            ctx.report.skipped[name] = reason
            continue
        ctx.report.passes[name] = spec_pass.apply(model, ctx)


def _quantize(model: nn.Module, ctx: BuildContext) -> None:
    from flashim.quant.apply import quantize_model

    spec, config = ctx.spec, ctx.config
    quant = spec.quant_configs[config.precision.quant]
    int8 = any(scheme.weights.type == "int" for scheme in quant.config_groups.values())
    if not int8 and not ctx.platform.supports(Feature.MXFP8_SCALED_MM):
        message = f"{spec.name}: no native MXFP8 GEMM on {ctx.platform.describe()}; quantized linears are emulated"
        if config.optimize.strict:
            raise RuntimeError(message)
        logger.warning(message)
        ctx.report.emulated_quant = True
    table = _load_calibration(spec, config, int8)
    model.quant_report = quantize_model(model, quant, table, protected=spec.protected)
    model.calibration_metadata = dict(table.metadata)
    ctx.report.quant = model.quant_report.summary()


def _apply_engine(model: nn.Module, ctx: BuildContext) -> None:
    from flashim.engines.compile import compile_regions

    engine = ctx.config.engine
    engine_spec = ctx.spec.engines[engine.kind]
    if engine_spec.setup is not None:
        engine_spec.setup(model, ctx)
    regions = engine_spec.regions(model) if callable(engine_spec.regions) else engine_spec.regions
    if not regions:
        return
    if engine_spec.recompile_limit is not None:
        # Regions compile once per input shape (static shapes); benchmarks with many resolutions would otherwise
        # exceed Dynamo's default limit and silently run the remaining shapes eagerly.
        import torch._dynamo.config as dynamo_config

        dynamo_config.recompile_limit = max(dynamo_config.recompile_limit, engine_spec.recompile_limit)
    ctx.report.compiled_regions = compile_regions(model, regions, mode=engine.mode, dynamic=engine.dynamic)


def build_config(config: RunConfig) -> nn.Module:
    """Build a model from a resolved config (see :func:`resolve`; no preset is replaced here)."""
    spec = get_spec(config.model)
    spec.check(config)
    if config.optimize.ops:
        from flashim import ops

        logger.info("%s: FLASHIM_OPS overrides %r (process-wide)", spec.name, config.optimize.ops)
        ops.configure(config.optimize.ops)
    if spec.compose is not None:
        model = spec.compose(config, build_config)
        report = BuildReport(config.digest(), platform_for(config.device).describe())
        report.components = {
            name: child.build_report for name, child in model.named_children() if hasattr(child, "build_report")
        }
    else:
        model = spec.load(config)
        device = next((p.device for p in model.parameters()), torch.device(config.device or "cpu"))
        report = BuildReport(config.digest(), platform_for(device).describe())
        ctx = BuildContext(config, spec, platform_for(device), report)
        _run_passes(model, ctx, "pre_quant")
        if spec.set_precision is not None:
            spec.set_precision(model, config)
        if config.precision.quant is not None:
            _quantize(model, ctx)
        _run_passes(model, ctx, "post_quant")
        if config.optimize.attention != "sdpa":
            if spec.set_attention is None:
                raise ValueError(f"{spec.name}: attention={config.optimize.attention!r} is not supported")
            spec.set_attention(model, config.optimize.attention)
            model.attention = config.optimize.attention
        _apply_engine(model, ctx)
    if spec.apply_inference is not None:
        spec.apply_inference(model, config.inference)
    if hasattr(model, "match_defaults"):  # an ImageMatcher: per-call defaults from the inference section
        from flashim.models.base import MATCH_OPTIONS

        model.match_defaults = {
            **type(model).match_defaults, **{k: v for k, v in config.inference.items() if k in MATCH_OPTIONS}
        }
    model.run_config = config
    model.build_report = report
    for attr, value in (("preset", config.preset), ("engine", config.engine.kind)):
        if not isinstance(getattr(type(model), attr, None), property):  # pipelines derive these from components
            setattr(model, attr, value)
    return model


def load(
    name: str,
    preset: str | None = None,
    *,
    device: torch.device | str | None = None,
    engine: str | None = None,
    strict: bool | None = None,
    calibration: str | Path | Any | None = None,
    config: RunConfig | Mapping[str, Any] | str | Path | None = None,
    overrides: Iterable[str] | Mapping[str, Any] = (),
) -> nn.Module:
    """Load a registered model: ``preset`` (``fp32`` / ``bf16`` / ``int8`` / ``fp8``; default: the model's) + ``config``
    (a :class:`RunConfig`, dict or .toml / .json / .yaml file) + dotted ``overrides`` + these keyword arguments (later
    layers win), resolved for the target device's platform (:func:`resolve`) and built (:func:`build_config`), e.g.
    ``load("romav2", "fp8", engine="compile", overrides={"arch.setting": "mega1500"})``.

    ``device`` / ``engine`` / ``strict`` apply to every component of a pipeline; ``calibration`` (a table path or a
    loaded table) only to single models (a component's: ``components.<name>.precision.calibration``). Presets the model
    or the platform replaced are in ``model.build_report.fallbacks``."""
    overlay: dict[str, Any] = {}
    if device is not None:
        overlay["device"] = str(device)
    if engine is not None:
        overlay["engine"] = {"kind": engine}
    if strict is not None:
        overlay["optimize"] = {"strict": strict}
    if calibration is not None:
        if get_spec(name).compose is not None:
            raise ValueError(f"{name}: pass a component's calibration (components.<name>.precision.calibration)")
        if isinstance(calibration, (str, Path)):
            overlay["precision"] = {"calibration": str(calibration)}
        else:  # an already-loaded table
            overlay = merge(overlay, {"optimize": {"options": {"calibration_table": calibration}}})
    user = merge(merge(_user_overlay(config), parse_overrides(overrides)), overlay)
    fallbacks: dict[str, str] = {}
    platform = platform_for(user.get("device") or default_device())
    model = build_config(resolve(name, preset, user, platform=platform, fallbacks=fallbacks))
    model.build_report.fallbacks = fallbacks
    return model


def dtype(name: str) -> torch.dtype:
    return DTYPES[name]
