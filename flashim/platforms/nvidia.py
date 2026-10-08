"""NVIDIA CUDA devices: compute capability -> ISA features, plus installed toolchains."""

from __future__ import annotations

import functools
import importlib.util
import logging
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

import torch
import torch.nn.functional as F

from flashim.platforms.base import Feature, Platform, Vendor

logger = logging.getLogger(__name__)


def has_module(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


@functools.cache
def nvcc_path() -> str | None:
    """``nvcc`` from ``CUDA_HOME`` / ``CUDA_PATH`` / the usual install prefixes / ``PATH``."""
    for home in (os.environ.get("CUDA_HOME"), os.environ.get("CUDA_PATH"), "/opt/cuda", "/usr/local/cuda"):
        if home and (Path(home) / "bin" / "nvcc").is_file():
            return str(Path(home) / "bin" / "nvcc")
    return shutil.which("nvcc")


def _ptxas_assembles(ptxas: str, arch: str) -> bool:
    """Whether ``ptxas`` knows ``arch``: an empty module assembles."""
    with tempfile.TemporaryDirectory() as directory:
        source = Path(directory) / "probe.ptx"
        source.write_text(f".version 8.5\n.target {arch}\n.address_size 64\n")
        try:
            command = [ptxas, f"--gpu-name={arch}", str(source), "-o", os.devnull]
            return subprocess.run(command, capture_output=True, check=False).returncode == 0
        except OSError:
            return False


# Architectures Triton 3.6's bundled assemblers have been run with here (sm_100: compile-checked): no probe.
_TRITON_ASSEMBLES = frozenset({7, 8, 9, 10, 12})


def triton_targets(capability: tuple[int, int]) -> bool:
    """Whether the installed Triton can assemble kernels for ``capability``. Triton 3.6 bundles the CUDA 12.9 ``ptxas``
    for Blackwell, which does not know sm_110 (Jetson AGX Thor: "Value 'sm_110a' is not defined for option
    'gpu-name'"). There Triton, and with it Inductor's kernels, is pointed at the CUDA toolkit's ``ptxas`` (13.0 or
    newer; ``TRITON_PTXAS_BLACKWELL_PATH``, left alone when already set) if that one assembles for the device."""
    if capability[0] in _TRITON_ASSEMBLES:
        return True
    arch = f"sm_{capability[0]}{capability[1]}a"
    configured = os.environ.get("TRITON_PTXAS_BLACKWELL_PATH")
    bundled = Path(importlib.util.find_spec("triton").origin).parent / "backends" / "nvidia" / "bin" / "ptxas-blackwell"
    if _ptxas_assembles(configured or str(bundled), arch):
        return True
    nvcc = nvcc_path()
    toolkit = None if nvcc is None else Path(nvcc).with_name("ptxas")
    if configured is None and toolkit is not None and toolkit.is_file() and _ptxas_assembles(str(toolkit), arch):
        os.environ["TRITON_PTXAS_BLACKWELL_PATH"] = str(toolkit)
        logger.info("Triton's ptxas does not know %s: using %s", arch, toolkit)
        return True
    logger.warning("no ptxas for %s (Triton's, TRITON_PTXAS_BLACKWELL_PATH, the CUDA toolkit's): Triton kernels are off", arch)
    return False


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
    if has_module("triton") and triton_targets(capability):
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
