from __future__ import annotations

import importlib.util

import torch
import torch.nn.functional as F

from flashim.platforms.base import Feature, Platform


def _has_triton() -> bool:
    return importlib.util.find_spec("triton") is not None


def _has_mxfp8_api() -> bool:
    return (
        hasattr(F, "scaled_mm")
        and hasattr(F, "ScalingType")
        and hasattr(F, "SwizzleType")
        and hasattr(torch, "float8_e4m3fn")
        and hasattr(torch, "float8_e8m0fnu")
    )


def _cuda_platform(index: int) -> Platform:
    capability = torch.cuda.get_device_capability(index)
    features = {Feature.CUDA_GRAPHS}
    if _has_triton():
        features.add(Feature.TRITON)
    if hasattr(torch, "float8_e4m3fn") and capability >= (8, 9):
        features.add(Feature.FP8_E4M3)
        if hasattr(torch, "_scaled_mm"):
            features.add(Feature.FP8_ROWWISE_SCALED_MM)
    # cuBLAS block-scaled MXFP8 exists on Blackwell (sm_100+, incl. consumer sm_120).
    if capability >= (10, 0) and _has_mxfp8_api():
        features.add(Feature.MXFP8_SCALED_MM)
    return Platform(
        device_type="cuda",
        device_index=index,
        name=torch.cuda.get_device_name(index),
        capability=capability,
        features=frozenset(features),
    )


# Plain dict (not functools.cache) so torch.compile can trace lookups without graph breaks.
_PLATFORMS: dict[tuple[str, int | None], Platform] = {}


def platform_for(device: torch.device | str | None = None) -> Platform:
    """Resolve the :class:`Platform` of ``device`` (default: best available)."""
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)
    key = (device.type, device.index)
    platform = _PLATFORMS.get(key)
    if platform is None:
        if device.type == "cuda":
            index = device.index if device.index is not None else torch.cuda.current_device()
            platform = _cuda_platform(index)
        else:
            platform = Platform(device_type=device.type, device_index=device.index, name=device.type)
        _PLATFORMS[key] = platform
    return platform


def current_platform() -> Platform:
    return platform_for(None)
