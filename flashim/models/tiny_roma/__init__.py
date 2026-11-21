"""Tiny-RoMa dense matcher (native FlashIM port of ``tiny_roma_v1_outdoor``)."""

from flashim.models.tiny_roma.config import TinyRoMaConfig
from flashim.models.tiny_roma.model import TinyRoMa, XFeatBackbone

__all__ = ["TinyRoMa", "TinyRoMaConfig", "XFeatBackbone"]
