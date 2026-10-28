"""The matching interface every image-pair model implements (:class:`ImageMatcher`), and its detector/descriptor
form (:class:`FeatureMatcher`)."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch

from flashim.types import Matches

# Per-call options of ``match``; a run config's ``inference`` section sets their defaults.
MATCH_OPTIONS = ("resize", "max_matches", "threshold")


class ImageMatcher:
    """``match(img_a, img_b, *, resize=None, max_matches=None, threshold=None) -> Matches`` for every model that
    matches an image pair (dense, detector-free, detector + descriptor, pipelines). Images are paths, PIL images,
    ``[H, W, 3]`` uint8 arrays or ``[1, 3, H, W]`` tensors in ``[0, 1]``.

    - ``resize``: long side of the working resolution (keypoints still come back in the original image's pixels);
      models with a fixed working resolution (RoMaV2: ``arch.setting``) reject it.
    - ``max_matches``: at most this many matches, bounded the model's own way: keypoints per image for detector-based
      models, the most confident matches for LoFTR, balanced samples of the dense warp for RoMaV2.
    - ``threshold``: the model's match threshold (mutual-NN cosine, dual-softmax probability, LightGlue's filter,
      LoFTR's coarse confidence); RoMaV2 has none.

    ``None`` takes the default: the run config's ``inference`` section (set by the builder over
    :attr:`match_defaults`), else the model's own. Subclasses implement :meth:`_match`.
    """

    match_defaults: Mapping[str, Any] = {}

    def match(
        self, img_a, img_b, *, resize: int | None = None, max_matches: int | None = None,
        threshold: float | None = None, **options,
    ) -> Matches:
        defaults = self.match_defaults
        resize = defaults.get("resize") if resize is None else resize
        max_matches = defaults.get("max_matches") if max_matches is None else max_matches
        threshold = defaults.get("threshold") if threshold is None else threshold
        for name, value in (("resize", resize), ("max_matches", max_matches)):
            if value is not None and (isinstance(value, bool) or not isinstance(value, int) or value <= 0):
                raise ValueError(f"{name} must be a positive integer or None, got {value!r}")
        with torch.inference_mode():
            return self._match(img_a, img_b, resize=resize, max_matches=max_matches, threshold=threshold, **options)

    def _match(self, img_a, img_b, *, resize: int | None, max_matches: int | None, threshold: float | None) -> Matches:
        raise NotImplementedError


class FeatureMatcher(ImageMatcher):
    """Detector/descriptor models: ``extract(image, resize=, top_k=) -> SparseFeatures`` for each image (``top_k`` =
    ``max_matches``), then ``match_features(a, b, threshold) -> Matches``; ``match`` attaches both feature sets."""

    def _match(self, img_a, img_b, *, resize: int | None, max_matches: int | None, threshold: float | None) -> Matches:
        a = self.extract(img_a, resize=resize, top_k=max_matches)
        b = self.extract(img_b, resize=resize, top_k=max_matches)
        result = self.match_features(a, b, threshold)
        result.features_A, result.features_B = a, b
        return result
