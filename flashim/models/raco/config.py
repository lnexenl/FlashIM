"""RaCo (cvg/RaCo) detector + ranker configuration."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RaCoConfig:
    # Output keypoints and the detector candidate pool the ranker reorders (``candidate_multiplier`` x, capped).
    num_keypoints: int = 1024
    candidate_multiplier: int = 2
    max_num_candidates: int = 3840
    # Detector: NMS window (odd), soft-argmax sub-pixel refinement over that window.
    nms_radius: int = 3
    subpixel_temperature: float = 0.5
    # Ranker: 9 residual blocks of width 12 + a 5x5 head at full resolution; its receptive-field radius is 20 px.
    ranker_width: int = 12
    ranker_blocks: int = 9
    ranker_receptive_radius: int = 20
    ranker_patch_size: int = 42
    # ImageNet statistics applied inside the model (inputs are RGB in [0, 1]).
    mean: tuple[float, float, float] = (0.485, 0.456, 0.406)
    std: tuple[float, float, float] = (0.229, 0.224, 0.225)

    @property
    def num_candidates(self) -> int:
        return max(self.num_keypoints, min(self.num_keypoints * self.candidate_multiplier, self.max_num_candidates))
