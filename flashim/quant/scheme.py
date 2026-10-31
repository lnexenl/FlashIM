"""Quantization schemes, in compressed-tensors vocabulary.

A :class:`QuantConfig` maps module-name globs (``targets``) to a
:class:`QuantScheme`; ``ignore`` excludes names. Models declare *protected*
modules (coordinate heads etc.); a config that targets one is a hard error rather
than a silent skip.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import asdict, dataclass, field
from fnmatch import fnmatchcase
from typing import Literal


@dataclass(frozen=True)
class QuantArgs:
    num_bits: int = 8
    type: Literal["float", "int"] = "float"
    format: str = "e4m3"
    strategy: Literal["tensor", "channel", "group", "block", "token"] = "group"
    group_size: int | None = 32
    scale_format: Literal["e8m0", "fp32"] = "e8m0"
    # Static activations use calibrated scales; dynamic ones compute them per call.
    dynamic: bool = False
    observer: str = "amax"


@dataclass(frozen=True)
class QuantScheme:
    targets: tuple[str, ...]
    weights: QuantArgs = QuantArgs()
    input_activations: QuantArgs | None = QuantArgs()

    @property
    def format_name(self) -> str:
        act = self.input_activations
        if self.weights.type == "int" and self.weights.num_bits == 8:
            return "int8_dynamic" if act is not None and act.dynamic else "int8_static"
        kind = "dynamic" if act is not None and act.dynamic else "static"
        return f"mxfp8_{kind}" if self.weights.group_size == 32 and self.weights.scale_format == "e8m0" else "custom"


MXFP8 = QuantArgs()
MXFP8_DYNAMIC = QuantArgs(dynamic=True, observer="none")
# Symmetric INT8: per-output-channel weights, static per-tensor activations (percentile-calibrated).
INT8_WEIGHTS = QuantArgs(type="int", format="int8", strategy="channel", group_size=None, scale_format="fp32")
INT8_ACTIVATIONS = QuantArgs(
    type="int", format="int8", strategy="tensor", group_size=None, scale_format="fp32", observer="percentile"
)
# Dynamic per-token INT8 activations (linears only): one scale per row, computed at run time.
INT8_TOKEN_ACTIVATIONS = QuantArgs(
    type="int", format="int8", strategy="token", group_size=None, scale_format="fp32", dynamic=True, observer="none"
)


@dataclass(frozen=True)
class QuantConfig:
    config_groups: dict[str, QuantScheme]
    ignore: tuple[str, ...] = ()
    name: str = ""

    def scheme_for(self, module_name: str) -> tuple[str, QuantScheme] | None:
        if any(fnmatchcase(module_name, pattern) for pattern in self.ignore):
            return None
        for group, scheme in self.config_groups.items():
            if any(fnmatchcase(module_name, pattern) for pattern in scheme.targets):
                return group, scheme
        return None

    def check_protected(self, module_names: Iterable[str], protected: Iterable[str]) -> None:
        protected = tuple(protected)
        hits = sorted(
            name
            for name in module_names
            if self.scheme_for(name) is not None and any(fnmatchcase(name, p) for p in protected)
        )
        if hits:
            raise ValueError(f"Quantization config {self.name!r} targets protected modules: {hits}")

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "config_groups": {k: asdict(v) for k, v in self.config_groups.items()},
            "ignore": list(self.ignore),
        }


@dataclass
class PackingReport:
    """What a quantization pass converted, and with which kernels."""

    linears: dict[str, str] = field(default_factory=dict)  # module name -> kernel name
    fused_blocks: list[str] = field(default_factory=list)
    fused_conv_blocks: list[str] = field(default_factory=list)
    convs: list[str] = field(default_factory=list)  # convolutions packed as im2col + GEMM
    bytes_before: int = 0
    bytes_after: int = 0

    def kernel_counts(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for kernel in self.linears.values():
            counts[kernel] = counts.get(kernel, 0) + 1
        return counts

    def summary(self) -> dict:
        return {
            "linears": len(self.linears),
            "linear_kernels": self.kernel_counts(),
            "fused_transformer_blocks": len(self.fused_blocks),
            "fused_conv_blocks": len(self.fused_conv_blocks),
            "im2col_convs": len(self.convs),
            "weight_bytes_before": self.bytes_before,
            "weight_bytes_after": self.bytes_after,
        }
