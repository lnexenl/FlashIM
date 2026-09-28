"""NVIDIA CUDA devices: compute capability -> ISA features, plus installed toolchains."""

from __future__ import annotations

import functools
import importlib.util
import os
import shutil
from pathlib import Path

import torch
import torch.nn.functional as F

from flashim.platforms.base import Feature, Platform, Vendor


def has_module(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


@functools.cache
def nvcc_path() -> str | None:
    """``nvcc`` from ``CUDA_HOME`` / ``CUDA_PATH`` / the usual install prefixes / ``PATH``."""
    for home in (os.environ.get("CUDA_HOME"), os.environ.get("CUDA_PATH"), "/opt/cuda", "/usr/local/cuda"):
        if home and (Path(home) / "bin" / "nvcc").is_file():
            return str(Path(home) / "bin" / "nvcc")
    return shutil.which("nvcc")


def _has_mxfp8_api() -> bool:
    return (
        hasattr(F, "scaled_mm")
        and hasattr(F, "ScalingType")
        and hasattr(F, "SwizzleType")
        and hasattr(torch, "float8_e4m3fn")
        and hasattr(torch, "float8_e8m0fnu")
    )


def isa_features(capability: tuple[int, int]) -> set[Feature]:
    """Hardware features of compute capability ``capability`` (independent of installed software)."""
    features = set()
    if capability >= (8, 0):
        features.add(Feature.MMA_SYNC)
    if capability >= (9, 0):
        features.add(Feature.TMA)
    if capability == (9, 0):
        features.add(Feature.WGMMA)
    if capability[0] in (10, 11):  # sm_100 / sm_103 / sm_110; consumer Blackwell (sm_12x) has no tcgen05
        features.add(Feature.TCGEN05)
    return features


def detect(index: int) -> Platform:
    capability = torch.cuda.get_device_capability(index)
    features = {Feature.CUDA_GRAPHS, *isa_features(capability)}
    if has_module("triton"):
        features.add(Feature.TRITON)
    if has_module("tilelang"):
        features.add(Feature.TILELANG)
    if nvcc_path() is not None:
        features.add(Feature.NVCC)
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
        vendor=Vendor.NVIDIA,
        arch=f"sm_{capability[0]}{capability[1]}",
    )
