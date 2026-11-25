"""OmniGlue (Jiang et al., CVPR 2024): SuperPoint ``sp_v6`` + DINOv2 ViT-B/14 + OmniGlue matcher."""

from flashim.models.omniglue.config import OmniGlueConfig
from flashim.models.omniglue.extractor import OmniGlueExtractor
from flashim.models.omniglue.model import OmniGlueFeatures, OmniGlueMatcher

__all__ = ["OmniGlueConfig", "OmniGlueExtractor", "OmniGlueFeatures", "OmniGlueMatcher"]
