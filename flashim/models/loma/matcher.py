"""LoMa's matcher: LightGlue's transformer (same modules and parameter names), LoMa's inputs and assignment.

Differences from :class:`~flashim.models.lightglue.model.LightGlue` (upstream ``loma.loma.LoMa.forward``):

- positions are DaD's normalized coordinates (``x = 2 (x_edge / W) - 1`` over the whole image, x by the width
  and y by the height), not LightGlue's centered / max-side scaling;
- no token confidences (no early stopping or pruning), every layer runs;
- the assignment of the last layer only, ``softmax(sim, -1) * softmax(sim, -2)`` without the matchabilities
  (which the checkpoints hold for training), and mutual maxima above ``filter_threshold`` (0.1).
"""

from __future__ import annotations

import logging
from collections import OrderedDict
from dataclasses import replace

import torch
from torch import Tensor, nn

from flashim import ops
from flashim.log import warn_once
from flashim.models.lightglue import weights as lightglue_weights
from flashim.models.lightglue.config import LightGlueConfig
from flashim.models.lightglue.model import LightGlue, MatchAssignment
from flashim.models.loma import weights
from flashim.utils import device_constant

logger = logging.getLogger(__name__)

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
    for the checkpoint (strict loading) but unused.

    ``low_precision_similarity`` (the ``low_precision_similarity`` pass of the reduced-precision presets): ``sim`` is
    ``ops.matmul_nt_fp32`` of ``final_proj``'s low-precision outputs (tensor cores, FP32 accumulation and output)
    instead of upstream's FP32 SGEMM of their FP32 casts; same inputs, only the accumulation differs."""

    low_precision_similarity = False

    def similarity(self, x0: Tensor, x1: Tensor) -> Tensor:
        """``[B, M, N]`` FP32 similarity ``final_proj(x0) final_proj(x1)^T / sqrt(dim)``."""
        dim = x0.shape[-1]
        if self.low_precision_similarity:
            m0, m1 = self.final_proj(x0), self.final_proj(x1)
            sims = [ops.matmul_nt_fp32(a, b, dim**-0.5) for a, b in zip(m0, m1)]
            return sims[0][None] if len(sims) == 1 else torch.stack(sims)
        m0, m1 = (self.final_proj(x).float() / dim**0.25 for x in (x0, x1))
        return m0 @ m1.transpose(-1, -2)

    def forward(self, x0: Tensor, x1: Tensor) -> Tensor:
        sim = self.similarity(x0, x1)
        return sim.log_softmax(-1) + sim.log_softmax(-2)

    def matches(self, x0: Tensor, x1: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        return ops.dual_softmax_mutual(self.similarity(x0, x1)[0])


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

    # ``cuda_graph`` pass: :meth:`best_matches` (all layers + the last assignment head; LoMa runs every layer, so no host
    # synchronization) replayed from one CUDA graph per input shape, the ``graphs`` most recent shapes kept; the
    # mutual-match filtering stays eager. 0 = eager.
    graphs = 0

    def best_matches(self, desc0: Tensor, desc1: Tensor, enc0: Tensor, enc1: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        if not self.graphs or not desc0.is_cuda or torch.is_grad_enabled():
            return super().best_matches(desc0, desc1, enc0, enc1)
        from flashim.engines.cuda_graph import CudaGraphRunner

        cache = self.__dict__.setdefault("_graph_runners", OrderedDict())
        key = tuple((t.shape, t.dtype, t.device) for t in (desc0, desc1, enc0, enc1))
        runner = cache.get(key)
        if runner is not None:
            cache.move_to_end(key)
            return runner(desc0, desc1, enc0, enc1)
        runner = CudaGraphRunner(super().best_matches)
        try:
            outputs = runner(desc0, desc1, enc0, enc1)
        except RuntimeError as exc:  # e.g. a kernel launched on the legacy default stream (unpatched SageAttention)
            warn_once(logger, f"loma-graph-{id(self)}", "LoMa matcher: CUDA graph capture failed, running eagerly "
                      "from now on (%s)", exc)
            self.graphs = 0
            torch.cuda.synchronize()
            return super().best_matches(desc0, desc1, enc0, enc1)
        cache[key] = runner
        while len(cache) > self.graphs:
            cache.popitem(last=False)
        return outputs
