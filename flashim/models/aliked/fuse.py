"""Inference-time fusion: fold eval BatchNorm into the preceding convolution.

Every ``ConvBlock``/``ResBlock`` is ``conv -> BN -> gate``; folding is exact (not an approximation) given
fixed BatchNorm running stats, and removes a kernel launch + a full read/write of the activation per conv.
For :class:`DeformableConv2d` the fold targets ``regular_conv`` (BatchNorm normalizes the block's *output*,
which only ``regular_conv``'s weight/bias produce; ``offset_conv`` reads the block's input and is untouched).
"""

from __future__ import annotations

from torch import nn

from flashim.models.aliked.blocks import ConvBlock, DeformableConv2d, ResBlock


def _fold(conv: nn.Module, bn: nn.BatchNorm2d) -> nn.Module:
    if isinstance(conv, DeformableConv2d):
        conv.regular_conv = nn.utils.fusion.fuse_conv_bn_eval(conv.regular_conv, bn)
        return conv
    return nn.utils.fusion.fuse_conv_bn_eval(conv, bn)


def fold_batchnorm(model: nn.Module) -> list[str]:
    """Fold every ``ConvBlock``/``ResBlock`` BatchNorm into its preceding convolution. Returns the folded
    module paths. Idempotent: a block already folded (``bn1``/``bn2`` are ``nn.Identity``) is skipped."""
    if model.training:
        raise RuntimeError("fold_batchnorm requires model.eval()")
    folded = []
    for name, module in model.named_modules():
        if isinstance(module, (ConvBlock, ResBlock)) and isinstance(module.bn1, nn.BatchNorm2d):
            module.conv1 = _fold(module.conv1, module.bn1)
            module.bn1 = nn.Identity()
            module.conv2 = _fold(module.conv2, module.bn2)
            module.bn2 = nn.Identity()
            folded.append(name)
    return folded
