"""Run configuration: one typed, layered, serializable description of how a model is built and run.

Sections (by when they take effect)::

    RunConfig
      model, variant, checkpoint, device
      arch       model-specific architecture fields (override the checkpoint config; different graph)
      precision  compute dtype, quantization config name, calibration, per-module precision overrides
      optimize   named passes (model-specific fusions), attention impl, op overrides, strict, pass options
      engine     eager / compile / ... (model-specific kinds), compile mode, dynamic shapes
      inference  model-specific per-call defaults (resolution, top_k, thresholds, sampling); no rebuild
      components model-specific component configs (a pipeline's extractor / matcher, ...)

Resolution (later layers win)::

    dataclass defaults < model preset (``extends`` chain) < config file / dict < dotted overrides < kwargs

A preset is a partial config (a nested dict) registered by the model's spec. Lists are replaced by default; a key
ending in ``+`` appends (``{"optimize": {"passes+": ["x"]}}``, CLI ``optimize.passes+=x``). The resolved config is
frozen, round-trips through :meth:`RunConfig.to_dict` / :meth:`RunConfig.from_dict`, and has a stable
:meth:`RunConfig.digest` for naming results.
"""

from __future__ import annotations

import copy
import dataclasses
import hashlib
import json
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any

DTYPES = ("fp32", "bf16", "fp16")


@dataclass(frozen=True)
class Precision:
    compute_dtype: str = "fp32"
    quant: str | None = None  # a QuantConfig name from the model spec
    calibration: str | None = None  # path; None = the spec's default table for ``quant``
    overrides: dict[str, str] = field(default_factory=dict)  # model-specific module -> dtype (e.g. RDD detector)


@dataclass(frozen=True)
class Optimize:
    passes: tuple[str, ...] = ()
    attention: str = "sdpa"  # ops.sdpa impl for attention layers: "sdpa" (torch) or e.g. "sage", "sage-fp8"
    ops: str = ""  # FLASHIM_OPS-syntax overrides (process-wide for now)
    strict: bool = False  # raise instead of emulating / skipping when the platform lacks a feature
    options: dict[str, Any] = field(default_factory=dict)  # pass options (model-specific, documented by the pass)


@dataclass(frozen=True)
class Engine:
    kind: str = "eager"
    mode: str | None = "max-autotune-no-cudagraphs"
    dynamic: bool = False


@dataclass(frozen=True)
class RunConfig:
    model: str
    preset: str | None = None  # informational: the preset this config was resolved from
    variant: str | None = None
    checkpoint: str | None = None
    device: str | None = None
    arch: dict[str, Any] = field(default_factory=dict)
    precision: Precision = field(default_factory=Precision)
    optimize: Optimize = field(default_factory=Optimize)
    engine: Engine = field(default_factory=Engine)
    inference: dict[str, Any] = field(default_factory=dict)
    components: dict[str, RunConfig] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return _to_plain(self)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> RunConfig:
        return _from_plain(cls, data, "")

    def digest(self) -> str:
        """Stable short hash of the resolved config (for result names / dedup)."""
        canonical = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"), default=str)
        return hashlib.sha256(canonical.encode()).hexdigest()[:12]

    def replace(self, **changes) -> RunConfig:
        return dataclasses.replace(self, **changes)


# -- plain-data conversion ---------------------------------------------------------------------------------


def _to_plain(value: Any) -> Any:
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return {f.name: _to_plain(getattr(value, f.name)) for f in fields(value)}
    if isinstance(value, tuple):
        return [_to_plain(v) for v in value]
    if isinstance(value, dict):
        return {k: _to_plain(v) for k, v in value.items()}
    return value


_NESTED = {"precision": Precision, "optimize": Optimize, "engine": Engine}


def _from_plain(cls: type, data: Mapping[str, Any], where: str) -> Any:
    known = {f.name: f for f in fields(cls)}
    unknown = sorted(set(data) - set(known))
    if unknown:
        raise ValueError(f"Unknown config key(s) {[where + k for k in unknown]}; valid: {sorted(known)}")
    kwargs = {}
    for name, value in data.items():
        nested = _NESTED.get(name)
        if nested is not None and value is not None and not isinstance(value, nested):
            if not isinstance(value, Mapping):
                raise TypeError(f"{where}{name} must be a mapping, got {type(value).__name__}")
            value = _from_plain(nested, value, f"{where}{name}.")
        elif name == "passes":
            value = tuple(value)
        elif name == "components":
            value = {
                key: sub if isinstance(sub, RunConfig) else _from_plain(RunConfig, sub, f"{where}components.{key}.")
                for key, sub in value.items()
            }
        kwargs[name] = value
    config = cls(**kwargs)
    if isinstance(config, Precision) and config.compute_dtype not in DTYPES:
        raise ValueError(f"{where}compute_dtype must be one of {DTYPES}, got {config.compute_dtype!r}")
    return config


# -- layering ---------------------------------------------------------------------------------------------


def merge(base: Mapping[str, Any], overlay: Mapping[str, Any]) -> dict[str, Any]:
    """Deep-merge ``overlay`` into ``base`` (new dict). Mappings merge recursively, everything else is replaced;
    ``"key+"`` appends to the list at ``key``."""
    out = dict(base)
    for key, value in overlay.items():
        if key.endswith("+"):
            key = key[:-1]
            out[key] = [*(out.get(key) or ()), *(value if isinstance(value, (list, tuple)) else [value])]
        elif isinstance(value, Mapping) and isinstance(out.get(key), Mapping):
            out[key] = merge(out[key], value)
        else:
            out[key] = value
    return out


def _parse_value(text: str) -> Any:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        lowered = text.lower()
        if lowered in ("true", "false"):
            return lowered == "true"
        if lowered in ("none", "null"):
            return None
        return text


def parse_overrides(overrides: Iterable[str] | Mapping[str, Any]) -> dict[str, Any]:
    """``["engine.kind=compile", "optimize.passes+=x", "inference.top_k=2048"]`` (or ``{"engine.kind": ...}``)
    -> a nested overlay dict. Values parse as JSON when they can (numbers, lists, true/false/null)."""
    items = overrides.items() if isinstance(overrides, Mapping) else (_split(o) for o in overrides)
    overlay: dict[str, Any] = {}
    for dotted, value in items:
        if isinstance(value, str) and not isinstance(overrides, Mapping):
            value = _parse_value(value)
        *parents, leaf = dotted.split(".")
        node = overlay
        for part in parents:
            node = node.setdefault(part, {})
        node[leaf] = value
    return overlay


def _split(override: str) -> tuple[str, str]:
    if "+=" in override:
        key, value = override.split("+=", 1)
        return key.strip() + "+", value.strip()
    if "=" not in override:
        raise ValueError(f"Override {override!r} is not key=value")
    key, value = override.split("=", 1)
    return key.strip(), value.strip()


def load_file(path: str | Path) -> dict[str, Any]:
    """A config overlay from ``.toml``, ``.json`` or (with PyYAML installed) ``.yaml`` / ``.yml``."""
    path = Path(path)
    text = path.read_text()
    if path.suffix == ".toml":
        try:
            import tomllib
        except ModuleNotFoundError:  # Python 3.10 (JetPack 6): the backport
            import tomli as tomllib

        return tomllib.loads(text)
    if path.suffix == ".json":
        return json.loads(text)
    if path.suffix in (".yaml", ".yml"):
        import yaml  # optional dependency

        return yaml.safe_load(text) or {}
    raise ValueError(f"Unsupported config file type {path.suffix!r} (use .toml, .json or .yaml)")


def resolve_preset(presets: Mapping[str, Mapping[str, Any]], name: str) -> dict[str, Any]:
    """Flatten a preset's ``extends`` chain into one overlay."""
    if name not in presets:
        raise ValueError(f"Unknown preset {name!r}; choose from {sorted(presets)}")
    overlay = copy.deepcopy(dict(presets[name]))  # callers mutate the result; presets are shared spec data
    parent = overlay.pop("extends", None)
    return merge(resolve_preset(presets, parent), overlay) if parent else overlay
