"""LoMa's matcher: LightGlue's transformer (same modules and parameter names), LoMa's inputs and assignment.

Differences from :class:`~flashim.models.lightglue.model.LightGlue` (upstream ``loma.loma.LoMa.forward``):

- positions are DaD's normalized coordinates (``x = 2 (x_edge / W) - 1`` over the whole image, x by the width
  and y by the height), not LightGlue's centered / max-side scaling;
- no token confidences (no early stopping or pruning), every layer runs;
- the assignment of the last layer only, ``softmax(sim, -1) * softmax(sim, -2)`` without the matchabilities
  (which the checkpoints hold for training), and mutual maxima above ``filter_threshold`` (0.1).
"""

from __future__ import annotations

from dataclasses import replace

import torch
from torch import Tensor, nn

from flashim import ops
from flashim.models.lightglue import weights as lightglue_weights
from flashim.models.lightglue.config import LightGlueConfig
from flashim.models.lightglue.model import LightGlue, MatchAssignment
from flashim.models.loma import weights
from flashim.utils import device_constant

_B = LightGlueConfig(input_dim=256, dim=256, num_heads=4, num_layers=9, filter_threshold=0.1)
VARIANTS = {
    "b128": replace(_B, input_dim=128),
    "b": _B,
    "l": replace(_B, dim=512, num_heads=8),
    "g": replace(_B, dim=1024, num_heads=16),
    "r": _B,  # B's architecture, trained with rotation augmentation
}


def normalize_keypoints(keypoints: Tensor, image_size: tuple[int, int]) -> Tensor:
    """FlashIM pixel coordinates (integer = pixel center) -> DaD's ``[-1, 1]`` (``-1`` / ``1``: the image
    edges)."""
    height, width = image_size
    return (keypoints.float() + 0.5) * device_constant([2 / width, 2 / height], keypoints) - 1


class DualSoftmaxAssignment(MatchAssignment):
    """LoMa's inference assignment: ``log_softmax(sim, -1) + log_softmax(sim, -2)``; ``matchability`` is kept
    for the checkpoint (strict loading) but unused."""

    def forward(self, x0: Tensor, x1: Tensor) -> Tensor:
        dim = x0.shape[-1]
        m0, m1 = (self.final_proj(x).float() / dim**0.25 for x in (x0, x1))
        sim = m0 @ m1.transpose(-1, -2)
        return sim.log_softmax(-1) + sim.log_softmax(-2)

    def matches(self, x0: Tensor, x1: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        dim = x0.shape[-1]
        m0, m1 = (self.final_proj(x).float() / dim**0.25 for x in (x0, x1))
        return ops.dual_softmax_mutual((m0 @ m1.transpose(-1, -2))[0])


class LoMaMatcher(LightGlue):
    """LoMa's matcher (variants in :data:`VARIANTS`); ``match_features`` takes FlashIM :class:`SparseFeatures`
    (pixel-center keypoints, ``image_size`` of the original image) and returns mutual matches."""

    def __init__(self, config: LightGlueConfig | None = None) -> None:
        config = config or VARIANTS["b"]
        if config.depth_confidence > 0:
            raise ValueError("LoMa has no token confidences: depth_confidence must stay -1")
        super().__init__(config)
        self.token_confidence = nn.ModuleList()
        self.log_assignment = nn.ModuleList(
            DualSoftmaxAssignment(config.dim) for _ in range(config.num_layers)
        )

    @classmethod
    def from_variant(
        cls, variant: str, config: LightGlueConfig | None = None, *, checkpoint: str | None = None,
        device: torch.device | str | None = None,
    ) -> LoMaMatcher:
        config = config or VARIANTS[variant]
        state = weights.load_state_dicts(variant, config.num_layers, checkpoint).matcher
        state = lightglue_weights.convert_state_dict(state, config.num_heads, config.num_layers)
        return cls.from_pretrained(config, state_dict=state, device=device)

    def normalized(self, keypoints: Tensor, image_size: tuple[int, int]) -> Tensor:
        return normalize_keypoints(keypoints, image_size)
