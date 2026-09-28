"""Detector/descriptor + LightGlue: keypoints and descriptors from one model, matches from another."""

from __future__ import annotations

import torch
from torch import nn

from flashim.models.lightglue.model import LightGlue
from flashim.models.romav2.model import ImageLike
from flashim.types import SparseFeatures, SparseMatchResult


class SparsePipeline(nn.Module):
    """``extractor`` (e.g. RDD: ``extract(image, resize=, top_k=) -> SparseFeatures``) followed by ``matcher``
    (:class:`LightGlue`). ``match`` accepts the extractor's ``resize`` / ``top_k`` and the matcher's ``threshold``."""

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
        return {"extractor": self.extractor.calibration_metadata, "matcher": self.matcher.calibration_metadata}

    def extract(self, image: ImageLike, **kwargs) -> SparseFeatures:
        return self.extractor.extract(image, **kwargs)

    def match_features(self, a: SparseFeatures, b: SparseFeatures, threshold: float | None = None) -> SparseMatchResult:
        return self.matcher.match_features(a, b, threshold)

    @torch.inference_mode()
    def match(
        self,
        img_a: ImageLike,
        img_b: ImageLike,
        *,
        resize: int | None = None,
        top_k: int | None = None,
        threshold: float | None = None,
    ) -> SparseMatchResult:
        a = self.extract(img_a, resize=resize, top_k=top_k)
        b = self.extract(img_b, resize=resize, top_k=top_k)
        return self.match_features(a, b, threshold)
