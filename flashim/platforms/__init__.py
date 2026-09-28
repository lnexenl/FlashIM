"""Hardware platforms: what a device is and which capabilities it offers."""

from flashim.platforms.base import Feature, Platform, Vendor
from flashim.platforms.detect import current_platform, default_device, platform_for, register_detector

__all__ = [
    "Feature",
    "Platform",
    "Vendor",
    "current_platform",
    "default_device",
    "platform_for",
    "register_detector",
]
