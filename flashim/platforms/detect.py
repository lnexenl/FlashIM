from __future__ import annotations

import logging
from collections.abc import Callable
from importlib.metadata import entry_points

import torch

from flashim.platforms import nvidia, others
from flashim.platforms.base import Platform

logger = logging.getLogger(__name__)

Detector = Callable[[int], Platform]

# torch device type -> detector for one device index. ``"cuda"`` is resolved to NVIDIA or ROCm at detection time.
_DETECTORS: dict[str, Detector] = {
    "xpu": others.detect_xpu,
    "npu": others.detect_npu,
    "mps": others.detect_mps,
}
_ENTRY_POINTS_LOADED = False


def register_detector(device_type: str, detector: Detector) -> None:
    """Teach FlashIM a torch device type (out-of-tree platforms do this from a ``flashim.platforms`` entry point,
    whose object is called with no arguments)."""
    _DETECTORS[device_type] = detector
    _PLATFORMS.clear()


def _load_entry_points() -> None:
    global _ENTRY_POINTS_LOADED
    if _ENTRY_POINTS_LOADED:
        return
    _ENTRY_POINTS_LOADED = True
    for entry in entry_points(group="flashim.platforms"):
        try:
            entry.load()()
        except Exception:  # a broken plugin must not break FlashIM
            logger.exception("platform plugin %r failed to load", entry.name)


def _current_index(device_type: str) -> int | None:
    module = getattr(torch, device_type, None)
    current = getattr(module, "current_device", None)
    return current() if current is not None else None


@torch.compiler.disable  # feature probes (find_spec, device queries) must run eagerly: the result is cached
def _detect(device: torch.device) -> Platform:
    _load_entry_points()
    if device.type == "cpu":
        return Platform(device_type="cpu", name="cpu")
    index = device.index if device.index is not None else _current_index(device.type)
    if device.type == "cuda":
        return others.detect_rocm(index) if torch.version.hip else nvidia.detect(index)
    detector = _DETECTORS.get(device.type)
    if detector is None:
        return Platform(device_type=device.type, device_index=index, name=device.type)
    return detector(index)


# Plain dict (not functools.cache) so torch.compile can trace lookups without graph breaks.
_PLATFORMS: dict[tuple[str, int | None], Platform] = {}


def platform_for(device: torch.device | str | None = None) -> Platform:
    """Resolve the :class:`Platform` of ``device`` (default: :func:`default_device`)."""
    device = torch.device(device if device is not None else default_device())
    key = (device.type, device.index)
    platform = _PLATFORMS.get(key)
    if platform is None:
        platform = _PLATFORMS[key] = _detect(device)
    return platform


def current_platform() -> Platform:
    return platform_for(None)


def default_device() -> torch.device:
    """The best available accelerator (torch's current accelerator), else CPU."""
    accelerator = getattr(torch, "accelerator", None)
    if accelerator is not None and accelerator.is_available():
        return torch.device(accelerator.current_accelerator().type)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
