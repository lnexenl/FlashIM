"""Inference-time fusion for RaCo: fold eval BatchNorm into the preceding convolution (exact)."""

from __future__ import annotations

from torch import nn

from flashim.models.raco.blocks import ConvBlock, ResBlock


def fold_batchnorm(model: nn.Module) -> int:
    """Fold every ``ConvBlock``/``ResBlock`` BatchNorm into its convolution. Returns the number of blocks folded
    (idempotent: already-folded blocks are skipped)."""
    if model.training:
        raise RuntimeError("fold_batchnorm requires model.eval()")
    folded = 0
    for module in model.modules():
        if isinstance(module, (ConvBlock, ResBlock)) and isinstance(module.bn1, nn.BatchNorm2d):
            module.conv1 = nn.utils.fusion.fuse_conv_bn_eval(module.conv1, module.bn1)
            module.bn1 = nn.Identity()
            module.conv2 = nn.utils.fusion.fuse_conv_bn_eval(module.conv2, module.bn2)
            module.bn2 = nn.Identity()
            folded += 1
    return folded
