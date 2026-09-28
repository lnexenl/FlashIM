from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

import torch


class Vendor(str, Enum):
    NVIDIA = "nvidia"
    AMD = "amd"
    INTEL = "intel"
    HUAWEI = "huawei"
    APPLE = "apple"
    GENERIC = "generic"  # CPU and unknown accelerators


class Feature(str, Enum):
    """Capabilities that kernels and quantization methods can require.

    Toolchains say a kernel language is installed and targets the device; ISA features say the hardware has an
    instruction family. Kernels require features rather than compute-capability ranges whenever a feature is not
    monotonic in the capability (sm_120 has TMA but neither WGMMA nor TCGEN05).
    """

    # -- toolchains
    TRITON = "triton"
    # TileLang kernels (TMA / mbarrier pipelines: also require TMA).
    TILELANG = "tilelang"
    # ``nvcc`` for the JIT-built CUDA C++ kernels.
    NVCC = "nvcc"
    # -- NVIDIA ISA
    MMA_SYNC = "mma_sync"  # warp-level tensor-core MMA (sm_80+)
    TMA = "tma"  # tensor memory accelerator + mbarrier (sm_90+, incl. consumer Blackwell)
    WGMMA = "wgmma"  # warpgroup MMA (sm_90a only)
    TCGEN05 = "tcgen05"  # 5th-gen tensor cores / TMEM (datacenter Blackwell sm_100/103/110, not sm_120)
    # -- number formats
    FP8_E4M3 = "fp8_e4m3"  # OCP E4M3 (``torch.float8_e4m3fn``), not AMD's ``fnuz`` variant
    # Block-scaled MXFP8 GEMM through ``torch.nn.functional.scaled_mm``
    # (E4M3 values, E8M0 scale per 32 K-elements, SWIZZLE_32_4_4 layout).
    MXFP8_SCALED_MM = "mxfp8_scaled_mm"
    # Row-wise scaled FP8 GEMM through ``torch._scaled_mm``.
    FP8_ROWWISE_SCALED_MM = "fp8_rowwise_scaled_mm"
    # -- runtime
    CUDA_GRAPHS = "cuda_graphs"


@dataclass(frozen=True)
class Platform:
    """A resolved device plus the features it supports.

    ``device_type`` is the *dispatch key* kernels register for: the torch device type, except that ROCm devices
    (which torch also calls ``"cuda"``) are ``"rocm"``, so NVIDIA-tuned CUDA kernels never select on AMD GPUs.
    ``capability`` is the NVIDIA compute capability ``(major, minor)`` and ``None`` elsewhere; ``arch`` names the
    architecture (``"sm_120"``, ``"gfx942"``, ...).
    """

    device_type: str
    device_index: int | None = None
    name: str = ""
    capability: tuple[int, int] | None = None
    features: frozenset[Feature] = field(default_factory=frozenset)
    vendor: Vendor = Vendor.GENERIC
    arch: str = ""
    torch_device_type: str | None = None  # when it differs from ``device_type`` (ROCm: "cuda")

    @property
    def device(self) -> torch.device:
        return torch.device(self.torch_device_type or self.device_type, self.device_index)

    def supports(self, *features: Feature) -> bool:
        return all(feature in self.features for feature in features)

    def has_capability(self, minimum: tuple[int, int]) -> bool:
        return self.capability is not None and self.capability >= minimum

    def describe(self) -> str:
        arch = f" {self.arch}" if self.arch else ""
        feats = ",".join(sorted(f.value for f in self.features)) or "-"
        return f"{self.device_type}{arch} ({self.name or 'unknown'}) features={feats}"
