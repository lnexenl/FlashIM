from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TinyRoMaConfig:
    """Architecture and sampling configuration for :class:`~flashim.models.tiny_roma.TinyRoMa`."""

    # Upstream ``tiny_roma_v1_outdoor`` defaults to ``exact_softmax=False`` (argmax + 4x-strided softmax).
    exact_softmax: bool = False
    sample_mode: str = "threshold_balanced"
    sample_thresh: float = 0.05
