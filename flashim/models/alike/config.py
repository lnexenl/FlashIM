"""ALIKE architecture configuration: one entry per released checkpoint (``alike-t/s/n/l``)."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ALIKEConfig:
    c1: int = 16
    c2: int = 32
    c3: int = 64
    c4: int = 128
    dim: int = 128
    single_head: bool = True
    radius: int = 2  # NMS radius (border removal is radius + 1)
    scores_th: float = 0.5  # threshold mode (no top_k)
    n_limit: int = 5000
    match_threshold: float = 0.0


VARIANTS: dict[str, ALIKEConfig] = {
    "alike-t": ALIKEConfig(c1=8, c2=16, c3=32, c4=64, dim=64),
    "alike-s": ALIKEConfig(c1=8, c2=16, c3=48, c4=96, dim=96),
    "alike-n": ALIKEConfig(c1=16, c2=32, c3=64, c4=128, dim=128),
    "alike-l": ALIKEConfig(c1=32, c2=64, c3=128, c4=128, dim=128, single_head=False),
}
