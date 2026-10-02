"""DeDoDe detector + descriptor (native FlashIM port); also hosts the shared VGG / DINOv2 / ConvRefiner blocks of DaD."""

from flashim.models.dedode.model import DeDoDe

__all__ = ["DeDoDe"]
