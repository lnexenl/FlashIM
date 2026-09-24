from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

import torch


class Feature(str, Enum):
    """Capabilities that kernels and quantization methods can require."""

    TRITON = "triton"
    FP8_E4M3 = "fp8_e4m3"
    # Block-scaled MXFP8 GEMM through ``torch.nn.functional.scaled_mm``
    # (E4M3 values, E8M0 scale per 32 K-elements, SWIZZLE_32_4_4 layout).
    MXFP8_SCALED_MM = "mxfp8_scaled_mm"
    # Row-wise scaled FP8 GEMM through ``torch._scaled_mm``.
    FP8_ROWWISE_SCALED_MM = "fp8_rowwise_scaled_mm"
    CUDA_GRAPHS = "cuda_graphs"


@dataclass(frozen=True)
class Platform:
    """A resolved device plus the features it supports.

    ``capability`` is the (major, minor) compute capability for CUDA/ROCm devices
    and ``None`` elsewhere.
    """

    device_type: str
    device_index: int | None = None
    name: str = ""
    capability: tuple[int, int] | None = None
    features: frozenset[Feature] = field(default_factory=frozenset)

    @property
    def device(self) -> torch.device:
        return torch.device(self.device_type, self.device_index)

    def supports(self, *features: Feature) -> bool:
        return all(feature in self.features for feature in features)

    def has_capability(self, minimum: tuple[int, int]) -> bool:
        return self.capability is not None and self.capability >= minimum

    def describe(self) -> str:
        cap = "" if self.capability is None else f" sm_{self.capability[0]}{self.capability[1]}"
        feats = ",".join(sorted(f.value for f in self.features)) or "-"
        return f"{self.device_type}{cap} ({self.name or 'unknown'}) features={feats}"
