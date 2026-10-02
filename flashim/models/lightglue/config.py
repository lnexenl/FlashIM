"""LightGlue architecture configuration (the RDD release, ``rdd_lg.pth``, is the default; see
``ALIKED_CONFIG`` for cvg/LightGlue's ALIKED-paired weights)."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class LightGlueConfig:
    input_dim: int = 256
    dim: int = 256
    num_heads: int = 4
    num_layers: int = 9
    filter_threshold: float = 0.01
    # Early stopping: after layer i < num_layers - 1 stop once the fraction of keypoints whose token confidence exceeds
    # the layer's threshold is above this value (LightGlue recommends 0.95); -1 runs every layer.
    depth_confidence: float = -1.0
    # RDD's reference pipeline hands ``(H, W)`` to LightGlue's keypoint normalization, which expects ``(W, H)``
    # for ``(x, y)`` keypoints. ``True`` reproduces that; ``False`` normalizes with the matching order.
    reference_size_order: bool = True


# cvg/LightGlue's ALIKED variant (``raco_aliked_lightglue.pth``): a 128 -> 256 input projection (ALIKED's
# descriptor dim is 128, but this LightGlue was trained with a 256-wide transformer), the correct (x, y)
# keypoint-normalization order, and the upstream default match threshold.
ALIKED_CONFIG = LightGlueConfig(input_dim=128, dim=256, filter_threshold=0.1, reference_size_order=False)

# cvg/LightGlue's SuperPoint and DISK variants (``superpoint_lightglue.pth`` / ``disk_lightglue.pth``): same
# transformer and keypoint normalization as the ALIKED one, input widths 256 / 128.
SUPERPOINT_CONFIG = LightGlueConfig(input_dim=256, dim=256, filter_threshold=0.1, reference_size_order=False)
DISK_CONFIG = ALIKED_CONFIG
# XFeat's "LighterGlue" (``xfeat-lighterglue.pt``): 64 -> 96 input projection, one head, six layers.
XFEAT_CONFIG = LightGlueConfig(input_dim=64, dim=96, num_heads=1, num_layers=6, filter_threshold=0.1, reference_size_order=False)
