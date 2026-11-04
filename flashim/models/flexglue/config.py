"""FlexGlue architecture configuration."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class FlexGlueConfig:
    """Fixed at the released checkpoint (``flexglue.pth``); ``arch`` overrides are validated against them.

    The matcher consumes dense DINOv3-ConvNeXt features (``descriptor_dim``-wide stride-8 search map) and maps
    normalized points of one image to sub-pixel correspondences in the other.
    """

    input_dim: int = 256  # descriptor / search-map channel width
    descriptor_dim: int = 512  # transformer width
    search_dim: int = 64  # correspondence-head search projection
    num_layers: int = 9
    num_heads: int = 8
    # The matcher turns points only when they are within ``max_distance_px`` pixels of the resized image (the
    # covisibility / cycle gates); <= 0 disables the gates (always keep).
    max_distance_px: float = 3.0
    # Covisibility gate: drop all matches when the pair covisibility score is below this.
    min_covisibility: float = 0.0
    filter_threshold: float = 0.1  # min mutual-NN match confidence
    # Early stopping: once the batch's covisibility score decays below ``min_covisibility * (1-decay)**(remaining)``,
    # stop after this layer and use its head. ``None`` (the upstream default) runs every layer (deterministic).
    early_stop_decay: float | None = None
    describe_resize: tuple[int, int] = (512, 512)  # descriptor input resolution (squashed), divisible by 32
    query_chunk_size: int = 4096  # points per correspondence-head call, bounds the (B, N, h*w) logits
    # Fields fixed by the checkpoint: ``validate`` rejects ``arch`` overrides of these.
    fixed: tuple[str, ...] = ("input_dim", "descriptor_dim", "search_dim", "num_heads", "query_chunk_size")

    def validate(self) -> None:
        """Reject values that don't match the checkpoint or upstream's constraints (called by the spec's loader)."""
        default = FlexGlueConfig()
        changed = [name for name in self.fixed if getattr(self, name) != getattr(default, name)]
        if changed:
            raise ValueError(f"fixed by the checkpoint, can't be overridden: {changed}")
        if not 1 <= self.num_layers <= default.num_layers:
            raise ValueError(f"num_layers must be in [1, {default.num_layers}] (truncation): {self.num_layers}")
        if self.early_stop_decay is not None:
            if not 0 <= self.early_stop_decay < 1:
                raise ValueError(f"early_stop_decay must be in [0, 1): {self.early_stop_decay}")
            if self.min_covisibility <= 0:
                raise ValueError("early_stop_decay needs min_covisibility > 0 (the gate would never fire)")
        if not isinstance(self.describe_resize, (tuple, list)) or len(self.describe_resize) != 2 or any(s <= 0 or s % 32 for s in self.describe_resize):
            raise ValueError(f"describe_resize must be a (h, w) tuple of positive multiples of 32: {self.describe_resize}")
