# Vendored from https://github.com/Parskatt/RoMaV2 (romav2/fast_fp8_refiner.py @ 95c9968) as the reference implementation of FlashIM's parity tests.
# Copyright (c) 2025 Johan Edstedt. SPDX-License-Identifier: MIT (see LICENSE in this directory).
# Changes: none besides this header.

"""Explicit static FP8 refiner path: pointwise GEMM adapters and fused blocks.

Unlike :class:`romav2.full_fp8_runtime.FullFP8Model`, this path performs no
``TorchDispatchMode`` interception and no generic convolution fallback. Every
refiner pointwise 1x1 convolution is converted exactly once into a
:class:`FastFP8PointwiseConv` wrapping a pre-packed
:class:`romav2.full_fp8.StaticFP8Linear`, consuming the static convolution
calibration keyed by exact module name. Every structural refiner block
(``conv_depthwise``/``norm``/``relu``/``conv_pointwise``) is converted exactly
once into a :class:`FastFP8RefinerBlock`, which folds the eval-mode
BatchNorm2d into a deep-copied depthwise convolution and runs
depthwise → ReLU → static FP8 pointwise. Depthwise (and every other
non-1x1) convolution stays on the eager BF16/FP32 path and never enters
FullFP8 dispatch.

Graph-readiness invariant
-------------------------
After conversion the wrapped modules guarantee:

1. **No per-call module discovery** — the depthwise/pointwise pair is fixed
   at conversion time; forward traverses the ordinary module tree only.
2. **No per-call weight packing** — the pointwise weight is quantized once
   into static FP8 buffers during construction and never re-read; the
   BatchNorm fold happens once on the copied depthwise convolution.
3. **No dynamic quantization** — activations and outputs use the static
   block-32 E8M0 scales from calibration; no runtime amax/statistic passes.

The hot path is therefore quantize → FP8 GEMM → dequantize for the pointwise
plus one eager depthwise convolution, suitable for CUDA-graph capture or
``torch.compile``.
"""

from __future__ import annotations

import copy
from typing import Literal

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from romav2.full_fp8 import (
    QuantizedWeight,
    StaticFP8Linear,
    _mxfp8_input_scale,
    quantize_block32,
)
from romav2.full_fp8_runtime import ModuleCalibration

__all__ = [
    "FastFP8PointwiseConv",
    "FastFP8RefinerBlock",
    "select_dpt_convolution_path",
]

DPTConvolutionPath = Literal["fp8_pointwise", "bf16_eager"]


def _is_static_pointwise_candidate(module: nn.Module) -> bool:
    return (
        isinstance(module, nn.Conv2d)
        and not isinstance(module, nn.ConvTranspose2d)
        and module.kernel_size == (1, 1)
        and module.stride == (1, 1)
        and module.padding == (0, 0)
        and module.dilation == (1, 1)
        and module.groups == 1
    )


def select_dpt_convolution_path(module: nn.Module) -> DPTConvolutionPath:
    """Choose the execution path for one DPT convolution module.

    Valid dense 1x1 convolutions (the ``DPTHead`` ``output_conv2`` tail and
    ``FeatureFusionBlock.out_conv``) run as static FP8 pointwise GEMMs.
    Everything else — 3x3 scratch projections, ``layer*_rn`` ResNet-style
    convolutions, ``ResidualConvUnit`` pairs, and transposed convolutions —
    stays on the eager BF16 path.
    """
    if _is_static_pointwise_candidate(module):
        return "fp8_pointwise"
    return "bf16_eager"


class FastFP8PointwiseConv(nn.Module):
    """Dense-NCHW-in/dense-NCHW-out 1x1 convolution on a static FP8 GEMM.

    Wraps a pre-packed :class:`StaticFP8Linear`. Construction is the only
    place where the convolution weight is quantized and the bias is padded;
    ``forward`` flattens NCHW to an NHWC matrix, quantizes with the static
    calibrated input scale, runs one FP8 GEMM, and dequantizes back to the
    input dtype. Channel counts are padded to block-32 by the quantization
    helpers, so any channel count is accepted as long as the calibration
    scales match the padded block count.
    """

    static: StaticFP8Linear
    in_channels: int
    out_channels: int

    def __init__(
        self, static: StaticFP8Linear, in_channels: int, out_channels: int
    ) -> None:
        super().__init__()
        self.static = static
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.call_count = 0
        self._mxfp8_input_scales: dict[int, Tensor] = {}

    def _native_input_scale(self, rows: int) -> Tensor:
        cached = self._mxfp8_input_scales.get(rows)
        if cached is None:
            input_scale = self.static.input_scale_exponent
            if input_scale is None:
                raise RuntimeError("Native MXFP8 requires a static input scale")
            cached = _mxfp8_input_scale(input_scale, rows)
            self._mxfp8_input_scales[rows] = cached
        return cached

    @classmethod
    def from_conv(
        cls, conv: nn.Conv2d, calibration: ModuleCalibration
    ) -> FastFP8PointwiseConv:
        """Pack ``conv`` once using static calibrated input/output scales."""
        if not _is_static_pointwise_candidate(conv):
            raise ValueError(f"Unsupported pointwise convolution: {conv}")
        if not torch.isfinite(conv.weight.detach()).all():
            raise ValueError("Cannot quantize non-finite pointwise convolution weights")
        if not torch.cuda.is_available() or torch.cuda.get_device_capability() < (
            12,
            0,
        ):
            raise RuntimeError("Fast FP8 execution requires an SM120-or-newer CUDA GPU")
        if not conv.weight.is_cuda:
            raise ValueError("FastFP8PointwiseConv requires a CUDA nn.Conv2d")
        expected_input_blocks = (conv.in_channels + 31) // 32
        if calibration.input_scale_exponent.numel() != expected_input_blocks:
            raise ValueError(
                "Pointwise convolution input scale has "
                f"{calibration.input_scale_exponent.numel()} blocks, expected "
                f"{expected_input_blocks} for {conv.in_channels} channels"
            )
        expected_output_blocks = (conv.out_channels + 31) // 32
        if calibration.output_scale_exponent.numel() != expected_output_blocks:
            raise ValueError(
                "Pointwise convolution output scale has "
                f"{calibration.output_scale_exponent.numel()} blocks, expected "
                f"{expected_output_blocks} for {conv.out_channels} channels"
            )
        weight = QuantizedWeight.from_float(
            conv.weight.detach().reshape(conv.out_channels, conv.in_channels)
        )
        bias = None
        if conv.bias is not None:
            bias = F.pad(
                conv.bias.detach(), (0, weight.out_features - conv.out_channels)
            )
        static = StaticFP8Linear(
            weight,
            bias,
            calibration.output_scale_exponent,
            input_scale_exponent=calibration.input_scale_exponent,
        )
        return cls(static, conv.in_channels, conv.out_channels)

    def forward(self, value: Tensor) -> Tensor:
        """Flatten NCHW to NHWC, run the FP8 GEMM, dequantize densely to NCHW.

        The output dtype matches the input dtype.
        """
        self.call_count += 1
        if value.ndim != 4:
            raise ValueError(f"Expected an NCHW tensor, got shape {tuple(value.shape)}")
        batch, channels, height, width = value.shape
        if channels != self.in_channels:
            raise ValueError(
                f"Expected {self.in_channels} input channels, got {channels}"
            )
        if value.is_contiguous(memory_format=torch.channels_last):
            matrix = value.permute(0, 2, 3, 1).reshape(-1, channels)
        else:
            matrix = value.permute(0, 2, 3, 1).contiguous().reshape(-1, channels)
        if self.static.supports_native_mxfp8:
            quantized = quantize_block32(matrix, self.static.input_scale_exponent)
            output = self.static.native_mxfp8(
                quantized, self._native_input_scale(quantized.matrix().shape[0])
            )
            if self.static.mxfp8_bias is not None:
                output = output + self.static.mxfp8_bias
            output = output[:, : self.out_channels].to(value.dtype)
        else:
            output = self.static(matrix).dequantize(dtype=value.dtype)
        return output.reshape(batch, height, width, self.out_channels).permute(
            0, 3, 1, 2
        )

    def extra_repr(self) -> str:
        return f"in_channels={self.in_channels}, out_channels={self.out_channels}"


def _fold_batch_norm_into_depthwise(conv: nn.Conv2d, norm: nn.BatchNorm2d) -> None:
    """Fold eval-mode BatchNorm2d statistics into ``conv`` in place."""
    if norm.running_mean is None or norm.running_var is None:
        raise ValueError("Cannot fold BatchNorm2d without running statistics")
    if norm.num_features != conv.out_channels:
        raise ValueError(
            f"BatchNorm2d has {norm.num_features} features, expected "
            f"{conv.out_channels} for the depthwise convolution"
        )
    scale = (
        norm.weight.detach().float()
        if norm.weight is not None
        else torch.ones(
            conv.out_channels, device=conv.weight.device, dtype=torch.float32
        )
    )
    shift = (
        norm.bias.detach().float()
        if norm.bias is not None
        else torch.zeros(
            conv.out_channels, device=conv.weight.device, dtype=torch.float32
        )
    )
    scale = scale / torch.sqrt(norm.running_var.detach().float() + norm.eps)
    weight = conv.weight.detach().float() * scale.view(-1, 1, 1, 1)
    if conv.bias is not None:
        bias = conv.bias.detach().float()
    else:
        bias = torch.zeros(
            conv.out_channels, device=conv.weight.device, dtype=torch.float32
        )
    bias = (bias - norm.running_mean.detach().float()) * scale + shift
    if not torch.isfinite(weight).all() or not torch.isfinite(bias).all():
        raise ValueError("Folded depthwise convolution contains non-finite values")
    with torch.no_grad():
        conv.weight.copy_(weight.to(conv.weight.dtype))
        if conv.bias is None:
            conv.bias = nn.Parameter(bias.to(conv.weight.dtype), requires_grad=False)
        else:
            conv.bias.copy_(bias.to(conv.bias.dtype))


class FastFP8RefinerBlock(nn.Module):
    """Eval-mode refiner block: folded depthwise → ReLU → static FP8 pointwise.

    Construction deep-copies the source block exactly once, folds the
    BatchNorm2d running statistics into the copied depthwise convolution
    (replaced by :class:`nn.Identity`), casts the depthwise weights to
    ``depthwise_dtype``, and packs the pointwise 1x1 convolution as a
    :class:`FastFP8PointwiseConv`. The source block is never mutated.
    Forward runs the eager depthwise in ``depthwise_dtype`` and returns an
    NCHW tensor in the input dtype.
    """

    conv_depthwise: nn.Conv2d
    norm: nn.Module
    relu: nn.Module
    conv_pointwise: FastFP8PointwiseConv

    def __init__(
        self,
        conv_depthwise: nn.Conv2d,
        norm: nn.Module,
        relu: nn.Module,
        conv_pointwise: FastFP8PointwiseConv,
        depthwise_dtype: torch.dtype,
    ) -> None:
        super().__init__()
        self.conv_depthwise = conv_depthwise
        self.norm = norm
        self.relu = relu
        self.conv_pointwise = conv_pointwise
        self.depthwise_dtype = depthwise_dtype

    @classmethod
    def from_block(
        cls,
        block: nn.Module,
        pointwise_calibration: ModuleCalibration | None = None,
        *,
        depthwise_dtype: torch.dtype = torch.bfloat16,
    ) -> FastFP8RefinerBlock:
        """Convert an eval refiner block into the fused static-FP8 form."""
        if block.training:
            raise ValueError("FastFP8RefinerBlock.from_block requires block.eval()")
        if not isinstance(getattr(block, "conv_depthwise", None), nn.Conv2d):
            raise TypeError(
                "FastFP8RefinerBlock requires a conv_depthwise nn.Conv2d, got "
                f"{type(getattr(block, 'conv_depthwise', None)).__name__}"
            )
        norm = getattr(block, "norm", None)
        if not isinstance(norm, (nn.BatchNorm2d, nn.Identity)):
            raise TypeError(f"Unsupported refiner normalization: {type(norm).__name__}")
        if not hasattr(block, "relu"):
            raise TypeError("FastFP8RefinerBlock requires a relu module")
        pointwise = getattr(block, "conv_pointwise", None)
        if isinstance(pointwise, FastFP8PointwiseConv):
            if pointwise_calibration is not None:
                raise ValueError(
                    "pointwise_calibration must not be provided when "
                    "conv_pointwise is already a FastFP8PointwiseConv"
                )
        elif isinstance(pointwise, nn.Conv2d):
            if pointwise_calibration is None:
                raise ValueError(
                    "FastFP8RefinerBlock.from_block requires a pointwise "
                    "ModuleCalibration for an nn.Conv2d conv_pointwise"
                )
        else:
            raise TypeError(
                "FastFP8RefinerBlock requires a conv_pointwise nn.Conv2d or "
                f"FastFP8PointwiseConv, got {type(pointwise).__name__}"
            )
        if not depthwise_dtype.is_floating_point:
            raise ValueError(
                f"depthwise_dtype must be floating point, got {depthwise_dtype}"
            )

        copied = copy.deepcopy(block)
        if isinstance(copied.norm, nn.BatchNorm2d):
            _fold_batch_norm_into_depthwise(copied.conv_depthwise, copied.norm)
            copied.norm = nn.Identity()
        copied.conv_depthwise = copied.conv_depthwise.to(depthwise_dtype)
        if isinstance(copied.conv_pointwise, nn.Conv2d):
            assert pointwise_calibration is not None
            copied.conv_pointwise = FastFP8PointwiseConv.from_conv(
                copied.conv_pointwise, pointwise_calibration
            )
        return cls(
            copied.conv_depthwise,
            copied.norm,
            copied.relu,
            copied.conv_pointwise,
            depthwise_dtype,
        )

    def forward(self, value: Tensor) -> Tensor:
        """Run depthwise → ReLU → static pointwise, preserving NCHW and dtype."""
        compute = value
        if compute.dtype != self.depthwise_dtype:
            compute = compute.to(self.depthwise_dtype)
        compute = self.conv_depthwise(compute)
        compute = self.norm(compute)
        compute = self.relu(compute)
        if compute.dtype != value.dtype:
            compute = compute.to(value.dtype)
        return self.conv_pointwise(compute)

    def extra_repr(self) -> str:
        return f"depthwise_dtype={self.depthwise_dtype}"
