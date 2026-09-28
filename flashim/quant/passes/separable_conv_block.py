"""Depthwise-separable conv-block passes: BatchNorm folding and fused block modules.

A block is ``conv_depthwise (5x5) -> norm -> relu -> conv_pointwise (1x1)`` (matched by
attribute names, like :mod:`transformer_block`). :func:`fold_block_batchnorm` folds eval
BatchNorm into the depthwise conv. :class:`MXFP8SeparableConvBlock` then runs one producer
kernel (BF16 depthwise + bias + ReLU + *dynamic* per-row MXFP8 quantization) feeding a
block-scaled MXFP8 pointwise GEMM, with BF16 channels-last output;
:class:`BF16SeparableConvBlock` runs the whole BF16 block as one kernel.
:class:`Int8SeparableConvBlock` swaps the dynamic MXFP8 for a *static* per-tensor INT8 activation
feeding an INT8 pointwise GEMM.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

from flashim import ops
from flashim.quant.formats import int8, mx
from flashim.utils import set_submodule


def is_separable_conv_block(module: nn.Module) -> bool:
    return all(hasattr(module, attr) for attr in ("conv_depthwise", "norm", "relu", "conv_pointwise"))


def _check_block(block: nn.Module, channels_allowed: tuple[int, ...] | None = None) -> int:
    """Validate a folded ``dw5x5 -> relu -> pw1x1`` block; return its channel count."""
    depthwise, pointwise = block.conv_depthwise, block.conv_pointwise
    if not isinstance(block.norm, nn.Identity):
        raise TypeError("Fold BatchNorm first (fold_block_batchnorm)")
    channels = depthwise.in_channels
    if not (
        (channels_allowed is None or channels in channels_allowed)
        and depthwise.groups == channels
        and depthwise.kernel_size == (5, 5)
        and depthwise.padding == (2, 2)
        and depthwise.stride == (1, 1)
        and depthwise.bias is not None
        and pointwise.kernel_size == (1, 1)
        and pointwise.in_channels == pointwise.out_channels == channels
        and pointwise.bias is not None
    ):
        raise ValueError(f"Unsupported separable conv block geometry: {depthwise}, {pointwise}")
    return channels


def _packed_depthwise_weight(depthwise: nn.Conv2d) -> Tensor:
    """``[C, 1, 5, 5]`` -> BF16 ``[25, C]`` (ky, kx major), the layout the fused kernels load."""
    channels = depthwise.in_channels
    return depthwise.weight.detach()[:, 0].permute(1, 2, 0).reshape(25, channels).to(torch.bfloat16).contiguous()


def fold_block_batchnorm(model: nn.Module) -> list[str]:
    """Fold eval BatchNorm into the depthwise conv of every separable conv block."""
    folded = []
    for name, module in model.named_modules():
        if is_separable_conv_block(module) and isinstance(module.norm, nn.BatchNorm2d):
            module.conv_depthwise = nn.utils.fusion.fuse_conv_bn_eval(module.conv_depthwise, module.norm)
            module.norm = nn.Identity()
            folded.append(name)
    return folded


class MXFP8SeparableConvBlock(nn.Module):
    def __init__(self, block: nn.Module) -> None:
        super().__init__()
        depthwise, pointwise = block.conv_depthwise, block.conv_pointwise
        self.channels = channels = _check_block(block, (32, 128, 512))
        self.register_buffer("depthwise_weight", _packed_depthwise_weight(depthwise))
        self.register_buffer("depthwise_bias", depthwise.bias.detach().float().contiguous())
        weight = pointwise.weight.detach().reshape(channels, channels)
        values, exponent = mx.quantize_dynamic_2d(weight)
        self.register_buffer("pointwise_weight", values.view(torch.uint8))
        self.register_buffer("pointwise_scale", mx.swizzle_scales(exponent))
        # Plain biased E8M0 bytes [N, K/32] for the fused kernel (same exponents as above).
        self.register_buffer("pointwise_scale_u8", mx.exponent_to_e8m0(exponent).view(torch.uint8).contiguous())
        self.register_buffer("pointwise_bias", pointwise.bias.detach().to(torch.bfloat16))
        # Run depthwise + quantize + pointwise as one kernel where available (opt-in: the
        # summation order of the fused GEMM is not guaranteed to match cuBLAS's).
        self.fuse_pointwise = False

    def forward(self, x: Tensor) -> Tensor:
        if self.fuse_pointwise:
            return ops.depthwise5x5_relu_pointwise_mxfp8(
                x,
                self.depthwise_weight,
                self.depthwise_bias,
                self.pointwise_weight.view(torch.float8_e4m3fn),
                self.pointwise_scale,
                self.pointwise_scale_u8,
                self.pointwise_bias,
            )
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


class Int8SeparableConvBlock(nn.Module):
    """Depthwise 5x5 + ReLU -> static INT8 producer, then per-channel INT8 pointwise GEMM (BF16 out)."""

    def __init__(self, block: nn.Module, input_scale: Tensor) -> None:
        super().__init__()
        depthwise, pointwise = block.conv_depthwise, block.conv_pointwise
        self.channels = channels = _check_block(block, (32, 128, 512))
        self.register_buffer("depthwise_weight", _packed_depthwise_weight(depthwise))
        self.register_buffer("depthwise_bias", depthwise.bias.detach().float().contiguous())
        weight = pointwise.weight.detach().reshape(channels, channels)
        scale = int8.weight_scales(weight)
        self.register_buffer("pointwise_weight", int8.quantize_symmetric(weight, scale[:, None]))
        self.register_buffer("pointwise_scale", scale)
        self.register_buffer("pointwise_bias", pointwise.bias.detach().to(torch.bfloat16))
        self.register_buffer("input_scale", int8.safe_scale(input_scale.to(weight.device)).reshape(()))

    def forward(self, x: Tensor) -> Tensor:
        return ops.depthwise5x5_relu_pointwise_int8(
            x,
            self.depthwise_weight,
            self.depthwise_bias,
            self.pointwise_weight,
            self.pointwise_scale,
            self.pointwise_bias,
            self.input_scale,
        )

    def extra_repr(self) -> str:
        return f"channels={self.channels}, dwconv5x5+relu->int8 static, pointwise int8"


class BF16SeparableConvBlock(nn.Module):
    """BF16 block as one kernel: depthwise 5x5 + bias + ReLU + pointwise + bias.

    Same precision as the autocast BF16 block (BF16 weights and activations, FP32
    accumulation); differs only in reduction order and in adding biases before rounding.
    """

    def __init__(self, block: nn.Module) -> None:
        super().__init__()
        depthwise, pointwise = block.conv_depthwise, block.conv_pointwise
        self.channels = channels = _check_block(block)
        self.register_buffer("depthwise_weight", _packed_depthwise_weight(depthwise))
        self.register_buffer("depthwise_bias", depthwise.bias.detach().to(torch.bfloat16))
        self.register_buffer(
            "pointwise_weight", pointwise.weight.detach().reshape(channels, channels).to(torch.bfloat16).contiguous()
        )
        self.register_buffer("pointwise_bias", pointwise.bias.detach().to(torch.bfloat16))

    def forward(self, x: Tensor) -> Tensor:
        if not x.is_contiguous(memory_format=torch.channels_last):
            x = x.contiguous(memory_format=torch.channels_last)
        return ops.depthwise5x5_relu_pointwise(
            x, self.depthwise_weight, self.depthwise_bias, self.pointwise_weight, self.pointwise_bias
        )

    def extra_repr(self) -> str:
        return f"channels={self.channels}, bf16 dwconv5x5+relu+pointwise fused"


def fuse_bf16_blocks(model: nn.Module, prefixes: tuple[str, ...]) -> list[str]:
    """Replace folded separable conv blocks under ``prefixes`` with :class:`BF16SeparableConvBlock`."""
    replaced = []
    for name, module in list(model.named_modules()):
        if is_separable_conv_block(module) and any(name.startswith(p) for p in prefixes):
            set_submodule(model, name, BF16SeparableConvBlock(module).to(module.conv_depthwise.weight.device))
            replaced.append(name)
    return replaced
