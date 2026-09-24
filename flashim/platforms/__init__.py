"""Hardware platforms: what a device is and which capabilities it offers."""

from flashim.platforms.base import Feature, Platform
from flashim.platforms.detect import current_platform, platform_for

__all__ = ["Feature", "Platform", "current_platform", "platform_for"]
