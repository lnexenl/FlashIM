"""Detector/descriptor + LightGlue: keypoints and descriptors from one model, matches from another."""

from __future__ import annotations

import dataclasses

import torch
from torch import nn

from flashim.models.base import FeatureMatcher
from flashim.models.lightglue.model import LightGlue
from flashim.models.romav2.model import ImageLike
from flashim.types import Matches, SparseFeatures


class SparsePipeline(FeatureMatcher, nn.Module):
    """``extractor`` (e.g. RDD: ``extract(image, resize=, top_k=) -> SparseFeatures``) followed by ``matcher``
    (:class:`LightGlue`): ``match``'s ``resize`` / ``max_matches`` go to the extractor, ``threshold`` to the matcher."""

    def __init__(self, extractor: nn.Module, matcher: LightGlue) -> None:
        super().__init__()
        self.extractor, self.matcher = extractor, matcher

    @property
    def device(self) -> torch.device:
        return self.extractor.device

    @property
    def preset(self) -> str:
        return f"{self.extractor.preset}+{self.matcher.preset}"

    @property
    def engine(self) -> str | None:
        return self.matcher.engine

    @property
    def quant_report(self):
        return self.matcher.quant_report or self.extractor.quant_report

    @property
    def calibration_metadata(self) -> dict[str, str] | None:
        # getattr: an AttributeError raised inside an nn.Module property surfaces as "no attribute", so a bare access
        # here made getattr(pipeline, "calibration_metadata", None) return None for extractors without one (ALIKED)
        return {"extractor": getattr(self.extractor, "calibration_metadata", None),
                "matcher": self.matcher.calibration_metadata}

    def extract(self, image: ImageLike, **kwargs) -> SparseFeatures:
        return self.extractor.extract(image, **kwargs)

    def match_features(self, a: SparseFeatures, b: SparseFeatures, threshold: float | None = None) -> Matches:
        """LightGlue sees the keypoints shifted by the extractor's ``matcher_keypoint_offset`` (RaCo: back to the
        pixel-edge convention its LightGlue weights were trained on); the matches keep the extractor's keypoints."""
        offset = getattr(self.extractor, "matcher_keypoint_offset", 0.0)
        if not offset:
            return self.matcher.match_features(a, b, threshold)
        shifted = (dataclasses.replace(f, keypoints=f.keypoints + offset) for f in (a, b))
        result = self.matcher.match_features(*shifted, threshold)
        if result.indices_A is None:  # no keypoints in one image
            return result
        ia, ib = result.indices_A, result.indices_B
        return Matches(a.keypoints[0][ia], b.keypoints[0][ib], result.confidence, ia, ib)
