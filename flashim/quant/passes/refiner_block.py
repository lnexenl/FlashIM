"""Refiner conv-block passes: BatchNorm folding and fused MXFP8 depthwise+pointwise blocks.

A refiner block is ``conv_depthwise (5x5) -> norm -> relu -> conv_pointwise (1x1)``.
:func:`fold_refiner_blocks` folds eval BatchNorm into the depthwise conv and switches
conv weights to channels-last. :class:`FusedMXFP8RefinerBlock` then runs one producer
kernel (BF16 depthwise + bias + ReLU + *dynamic* per-row MXFP8 quantization) feeding a
block-scaled MXFP8 pointwise GEMM, with BF16 channels-last output.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

from flashim import ops
from flashim.quant.formats import mx


def is_refiner_block(module: nn.Module) -> bool:
    return all(hasattr(module, attr) for attr in ("conv_depthwise", "norm", "relu", "conv_pointwise"))


def fold_refiner_blocks(model: nn.Module, *, channels_last: bool = True) -> list[str]:
    """Fold BN into depthwise convs of every refiner block; mark refiners channels-last."""
    folded = []
    for name, module in model.named_modules():
        if is_refiner_block(module) and isinstance(module.norm, nn.BatchNorm2d):
            module.conv_depthwise = nn.utils.fusion.fuse_conv_bn_eval(module.conv_depthwise, module.norm)
            module.norm = nn.Identity()
            folded.append(name)
        if hasattr(module, "block1") and hasattr(module, "hidden_blocks") and hasattr(module, "channels_last"):
            module.channels_last = channels_last
            if channels_last:
                for child in module.modules():
                    if isinstance(child, nn.Conv2d):
                        child.weight.data = child.weight.data.to(memory_format=torch.channels_last)
    return folded


class FusedMXFP8RefinerBlock(nn.Module):
    def __init__(self, block: nn.Module) -> None:
        super().__init__()
        depthwise, pointwise = block.conv_depthwise, block.conv_pointwise
        if not isinstance(block.norm, nn.Identity):
            raise TypeError("Fold BatchNorm first (fold_refiner_blocks)")
        channels = depthwise.in_channels
        if not (
            channels in (32, 128, 512)
            and depthwise.groups == channels
            and depthwise.kernel_size == (5, 5)
            and depthwise.padding == (2, 2)
            and depthwise.stride == (1, 1)
            and depthwise.bias is not None
            and pointwise.kernel_size == (1, 1)
            and pointwise.in_channels == pointwise.out_channels == channels
            and pointwise.bias is not None
        ):
            raise ValueError(f"Unsupported refiner block geometry: {depthwise}, {pointwise}")
        self.channels = channels
        self.register_buffer(
            "depthwise_weight",
            depthwise.weight.detach()[:, 0].permute(1, 2, 0).reshape(25, channels).to(torch.bfloat16).contiguous(),
        )
        self.register_buffer("depthwise_bias", depthwise.bias.detach().float().contiguous())
        weight = pointwise.weight.detach().reshape(channels, channels)
        values, exponent = mx.quantize_dynamic_2d(weight)
        self.register_buffer("pointwise_weight", values.view(torch.uint8))
        self.register_buffer("pointwise_scale", mx.swizzle_scales(exponent))
        self.register_buffer("pointwise_bias", pointwise.bias.detach().to(torch.bfloat16))
        self.call_count = 0

    def forward(self, x: Tensor) -> Tensor:
        if not torch.compiler.is_compiling():
            self.call_count += 1
        batch, _, height, width = x.shape
        quantized, scale = ops.depthwise5x5_relu_mxfp8(x, self.depthwise_weight, self.depthwise_bias)
        out = ops.mxfp8_scaled_mm(
            quantized,
            self.pointwise_weight.view(torch.float8_e4m3fn).t(),
            scale,
            self.pointwise_scale,
            self.pointwise_bias,
        )
        return out.reshape(batch, height, width, self.channels).permute(0, 3, 1, 2)

    def extra_repr(self) -> str:
        return f"channels={self.channels}, dwconv5x5+relu->mxfp8 dynamic, pointwise mxfp8"
