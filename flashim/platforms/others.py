"""Non-NVIDIA devices. Each gets its own dispatch key and no NVIDIA ISA features, so kernels written and validated
for NVIDIA never select there; ports register impls for these keys explicitly.

Features are deliberately conservative: a feature is only listed once some kernel has been validated with it on
that vendor (e.g. MI300's FP8 is the ``fnuz`` variant, not the OCP E4M3 that ``Feature.FP8_E4M3`` means).
"""

from __future__ import annotations

import torch

from flashim.platforms.base import Feature, Platform, Vendor
from flashim.platforms.nvidia import has_module


def detect_rocm(index: int) -> Platform:
    """AMD GPU under ROCm PyTorch (torch device type ``"cuda"``, dispatch key ``"rocm"``)."""
    arch = getattr(torch.cuda.get_device_properties(index), "gcnArchName", "").split(":")[0]
    features = {Feature.CUDA_GRAPHS}
    if has_module("triton"):
        features.add(Feature.TRITON)
    return Platform(
        device_type="rocm",
        device_index=index,
        name=torch.cuda.get_device_name(index),
        features=frozenset(features),
        vendor=Vendor.AMD,
        arch=arch,
        torch_device_type="cuda",
    )


def detect_xpu(index: int) -> Platform:
    """Intel GPU (``torch.xpu``)."""
    features = {Feature.TRITON} if has_module("triton") else set()
    return Platform(
        device_type="xpu",
        device_index=index,
        name=torch.xpu.get_device_name(index),
        features=frozenset(features),
        vendor=Vendor.INTEL,
    )


def detect_npu(index: int) -> Platform:
    """Huawei Ascend NPU (``torch_npu`` registers ``torch.npu``)."""
    name = torch.npu.get_device_name(index)  # type: ignore[attr-defined]
    return Platform(device_type="npu", device_index=index, name=name, vendor=Vendor.HUAWEI, arch=name)


def detect_mps(index: int) -> Platform:
    return Platform(device_type="mps", device_index=index, name="mps", vendor=Vendor.APPLE)
