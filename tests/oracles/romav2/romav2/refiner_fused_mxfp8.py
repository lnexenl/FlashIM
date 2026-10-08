# Vendored from https://github.com/Parskatt/RoMaV2 (romav2/refiner_fused_mxfp8.py @ 95c9968) as the reference implementation of FlashIM's parity tests.
# Copyright (c) 2025 Johan Edstedt. SPDX-License-Identifier: MIT (see LICENSE in this directory).
# Changes: none besides this header.

"""Fused MXFP8 refiner blocks: depthwise 5x5 + ReLU + MXFP8 quantization.

This module promotes the C=512 refiner blocks (``refiners.4``) to a fused
inference path:

```text
channels-last BF16 input
  -> depthwise 5x5 convolution (folded bias)
  -> ReLU
  -> MXFP8 block-32 E4M3 activation quantization (fused producer kernel)
  -> block-scaled MXFP8 pointwise GEMM (torch F.scaled_mm)
  -> channels-last BF16 output
```

The producer Triton kernel fuses the depthwise convolution, the folded
BatchNorm bias, the ReLU, and the MXFP8 quantization into one launch and
emits the ``SWIZZLE_32_4_4`` E8M0 scale layout consumed directly by
``F.scaled_mm``. Compared with the eager BF16 block this removes the
activation round-trips between four kernels; compared with the unfused
``PortablePointwiseConv2d`` refiner path it removes the quantizer kernel
boundary that erased the isolated GEMM gain end to end.

Accuracy evidence (full Mega1500, paired against the all-transformer MXFP8 +
folded + channels-last baseline, 300 pairs/scene):

| metric | seed 17 | seed 29 | seed 42 | mean |
| --- | ---: | ---: | ---: | ---: |
| AUC@5 | +0.000751 | -0.003752 | -0.001009 | -0.001337 |
| AUC@10 | -0.000305 | -0.002361 | +0.000834 | -0.000610 |
| AUC@20 | +0.000009 | -0.000407 | +0.000779 | +0.000127 |
| mAP@5 | -0.002667 | +0.000667 | +0.005200 | +0.001067 |
| mAP@10 | -0.000667 | -0.001267 | +0.003133 | +0.000400 |
| mAP@20 | -0.000867 | -0.000033 | +0.001467 | +0.000189 |

Every three-seed mean stays within +/-0.0014, inside the benchmark's own
seed-to-seed noise (the accepted all-transformer baseline itself swings
-0.0034/+0.0006 on AUC@5 across the same seeds). Latency: 296.9 ms vs
320.5 ms paired median (-23.6 ms, 1.080x) on the same Toronto Mega-1500
benchmark; see ``profile/romav2_refiner_fused_mxfp8_sm120_v1``.

Usage:

```python
from romav2 import (
    convert_for_inference,
    fuse_refiner_mxfp8_for_inference,
    optimize_refiner_for_inference,
)

convert_for_inference(model, backend="mxfp8", linear_scope="all_transformer")
optimize_refiner_for_inference(model, channels_last=True)  # required first
fuse_refiner_mxfp8_for_inference(model)
```

``optimize_refiner_for_inference(channels_last=True)`` must run first: the
fused block requires the eval-mode BatchNorm folded into the depthwise
convolution and channels-last activations. Conversion is inference-only and
validated on CUDA devices with native block-scaled ``F.scaled_mm`` support
(SM120); other devices keep the BF16 path.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import TYPE_CHECKING

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from romav2.inference import (
    InferenceReport,
    PortablePointwiseConv2d,
    resolve_inference_backend,
)

if TYPE_CHECKING:
    from romav2.refiner import Block

try:
    import triton
    import triton.language as tl
except ImportError:  # PyTorch builds without Triton cannot run the fused path.
    triton = None
    tl = None

__all__ = [
    "FusedRefinerMXFP8Block",
    "FusedRefinerMXFP8PartsBlock",
    "fuse_refiner_mxfp8_for_inference",
    "fused_depthwise_relu_mxfp8",
]

REFINER_FUSED_RECIPE = "refiner_fused_mxfp8_scope_wa"

_CHANNELS = 512
_DEPTHWISE_KERNEL = (5, 5)
_POINTWISE_KERNEL = (1, 1)
_GROUP_SIZE = 32
_GROUPS = _CHANNELS // _GROUP_SIZE
_FP8_MAX = 448.0

if triton is not None:

    @triton.jit
    def _depthwise_relu_mxfp8_kernel(
        input_ptr,
        weight_ptr,
        bias_ptr,
        output_ptr,
        scale_bits_ptr,
        rows,
        rows_per_image,
        HEIGHT: tl.constexpr,
        WIDTH: tl.constexpr,
        BLOCK_M: tl.constexpr,
        CHANNELS: tl.constexpr,
        SCALE_TILES: tl.constexpr,
    ):
        pid_m = tl.program_id(0)
        group = tl.program_id(1)
        row = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        channel = group * 32 + tl.arange(0, 32)
        batch = row // rows_per_image
        image_row = row - batch * rows_per_image
        y = image_row // WIDTH
        x = image_row - y * WIDTH
        valid_row = row < rows
        batch_base = batch * rows_per_image * CHANNELS

        accumulator = tl.zeros((BLOCK_M, 32), dtype=tl.float32)
        for ky in range(5):
            input_y = y + ky - 2
            valid_y = (input_y >= 0) & (input_y < HEIGHT)
            for kx in range(5):
                input_x = x + kx - 2
                valid_x = (input_x >= 0) & (input_x < WIDTH)
                input_offset = (
                    batch_base[:, None]
                    + (input_y[:, None] * WIDTH + input_x[:, None]) * CHANNELS
                    + channel[None, :]
                )
                values = (
                    tl.load(
                        input_ptr + input_offset,
                        mask=valid_row[:, None] & valid_y[:, None] & valid_x[:, None],
                        other=0.0,
                    )
                    .to(tl.bfloat16)
                    .to(tl.float32)
                )
                weight = tl.load(weight_ptr + (ky * 5 + kx) * CHANNELS + channel).to(
                    tl.float32
                )
                accumulator += values * weight[None, :]

        bias = tl.load(bias_ptr + channel).to(tl.float32)
        accumulator = tl.maximum(accumulator + bias[None, :], 0.0)
        amax = tl.max(tl.abs(accumulator), axis=1)
        exponent = tl.ceil(tl.log2(amax / 448.0))
        exponent = tl.maximum(tl.minimum(exponent, 127.0), -127.0)
        exponent = tl.where(amax == 0.0, -127.0, exponent)
        quantized = tl.maximum(
            tl.minimum(accumulator * tl.exp2(-exponent)[:, None], 448.0), -448.0
        )
        output_offset = row[:, None] * CHANNELS + channel[None, :]
        tl.store(
            output_ptr + output_offset,
            quantized,
            mask=valid_row[:, None],
        )

        row_in_tile = row % 128
        scale_offset = (
            ((row // 128) * SCALE_TILES + group // 4) * 512
            + (row_in_tile % 32) * 16
            + (row_in_tile // 32) * 4
            + group % 4
        )
        scale_bits = tl.where(valid_row, exponent + 127.0, 0.0).to(tl.uint8)
        tl.store(scale_bits_ptr + scale_offset, scale_bits)

    @triton.jit
    def _depthwise_relu_mxfp8_parts_kernel(
        f_a_ptr,
        f_ba_ptr,
        displacement_ptr,
        correlation_ptr,
        weight_ptr,
        bias_ptr,
        output_ptr,
        scale_bits_ptr,
        rows,
        rows_per_image,
        f_a_stride_b,
        f_a_stride_h,
        f_a_stride_w,
        f_a_stride_c,
        f_ba_stride_b,
        f_ba_stride_h,
        f_ba_stride_w,
        f_ba_stride_c,
        displacement_stride_b,
        displacement_stride_c,
        displacement_stride_h,
        displacement_stride_w,
        correlation_stride_b,
        correlation_stride_c,
        correlation_stride_h,
        correlation_stride_w,
        HEIGHT: tl.constexpr,
        WIDTH: tl.constexpr,
        BLOCK_M: tl.constexpr,
    ):
        pid_m = tl.program_id(0)
        group = tl.program_id(1)
        row = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        channel = group * 32 + tl.arange(0, 32)
        batch = row // rows_per_image
        image_row = row - batch * rows_per_image
        y = image_row // WIDTH
        x = image_row - y * WIDTH
        valid_row = row < rows

        accumulator = tl.zeros((BLOCK_M, 32), dtype=tl.float32)
        for ky in range(5):
            input_y = y + ky - 2
            valid_y = (input_y >= 0) & (input_y < HEIGHT)
            for kx in range(5):
                input_x = x + kx - 2
                valid_x = (input_x >= 0) & (input_x < WIDTH)
                valid_position = (
                    valid_row[:, None] & valid_y[:, None] & valid_x[:, None]
                )
                f_a_offset = (
                    batch[:, None] * f_a_stride_b
                    + input_y[:, None] * f_a_stride_h
                    + input_x[:, None] * f_a_stride_w
                    + channel[None, :] * f_a_stride_c
                )
                f_ba_offset = (
                    batch[:, None] * f_ba_stride_b
                    + input_y[:, None] * f_ba_stride_h
                    + input_x[:, None] * f_ba_stride_w
                    + (channel[None, :] - 192) * f_ba_stride_c
                )
                displacement_offset = (
                    batch[:, None] * displacement_stride_b
                    + (channel[None, :] - 384) * displacement_stride_c
                    + input_y[:, None] * displacement_stride_h
                    + input_x[:, None] * displacement_stride_w
                )
                correlation_offset = (
                    batch[:, None] * correlation_stride_b
                    + (channel[None, :] - 463) * correlation_stride_c
                    + input_y[:, None] * correlation_stride_h
                    + input_x[:, None] * correlation_stride_w
                )
                is_f_a = channel[None, :] < 192
                is_f_ba = (channel[None, :] >= 192) & (channel[None, :] < 384)
                is_displacement = (channel[None, :] >= 384) & (channel[None, :] < 463)
                values = (
                    tl.load(
                        f_a_ptr + f_a_offset,
                        mask=valid_position & is_f_a,
                        other=0.0,
                    )
                    .to(tl.bfloat16)
                    .to(tl.float32)
                )
                values += (
                    tl.load(
                        f_ba_ptr + f_ba_offset,
                        mask=valid_position & is_f_ba,
                        other=0.0,
                    )
                    .to(tl.bfloat16)
                    .to(tl.float32)
                )
                values += (
                    tl.load(
                        displacement_ptr + displacement_offset,
                        mask=valid_position & is_displacement,
                        other=0.0,
                    )
                    .to(tl.bfloat16)
                    .to(tl.float32)
                )
                values += (
                    tl.load(
                        correlation_ptr + correlation_offset,
                        mask=valid_position & (channel[None, :] >= 463),
                        other=0.0,
                    )
                    .to(tl.bfloat16)
                    .to(tl.float32)
                )
                weight = tl.load(weight_ptr + (ky * 5 + kx) * 512 + channel).to(
                    tl.float32
                )
                accumulator += values * weight[None, :]

        bias = tl.load(bias_ptr + channel).to(tl.float32)
        accumulator = tl.maximum(accumulator + bias[None, :], 0.0)
        amax = tl.max(tl.abs(accumulator), axis=1)
        exponent = tl.ceil(tl.log2(amax / 448.0))
        exponent = tl.maximum(tl.minimum(exponent, 127.0), -127.0)
        exponent = tl.where(amax == 0.0, -127.0, exponent)
        quantized = tl.maximum(
            tl.minimum(accumulator * tl.exp2(-exponent)[:, None], 448.0), -448.0
        )
        output_offset = row[:, None] * 512 + channel[None, :]
        tl.store(output_ptr + output_offset, quantized, mask=valid_row[:, None])

        row_in_tile = row % 128
        scale_offset = (
            ((row // 128) * 4 + group // 4) * 512
            + (row_in_tile % 32) * 16
            + (row_in_tile // 32) * 4
            + group % 4
        )
        scale_bits = tl.where(valid_row, exponent + 127.0, 0.0).to(tl.uint8)
        tl.store(scale_bits_ptr + scale_offset, scale_bits)
else:
    _depthwise_relu_mxfp8_kernel = None
    _depthwise_relu_mxfp8_parts_kernel = None


def fused_depthwise_relu_mxfp8(
    value: Tensor,
    weight: Tensor,
    bias: Tensor,
    *,
    block_m: int = 32,
    num_warps: int = 4,
) -> tuple[Tensor, Tensor]:
    """Fuse depthwise 5x5 + bias + ReLU + MXFP8 quantization in one kernel.

    ``value`` is a channels-last BF16 tensor of shape ``[B, C, H, W]`` with
    ``C`` equal to 32, 128, or 512. ``weight`` is the packed ``[25, C]``
    depthwise kernel and ``bias`` the folded ``[C]`` bias. Returns the E4M3
    ``[B*H*W, C]`` activations and the hardware-swizzled E8M0 scales expected
    by ``F.scaled_mm`` with ``ScalingType.BlockWise1x32`` /
    ``SwizzleType.SWIZZLE_32_4_4``. The depthwise neighborhood is computed
    per image, so batch-boundary rows never mix pixels across images.
    """
    if _depthwise_relu_mxfp8_kernel is None or triton is None:
        raise RuntimeError("fused_depthwise_relu_mxfp8 requires Triton")
    if value.ndim != 4 or int(value.shape[1]) not in (32, 128, 512):
        raise ValueError(
            f"Expected [B, 32, 128, or 512, H, W], got {tuple(value.shape)}"
        )
    channels = int(value.shape[1])
    groups = channels // _GROUP_SIZE
    scale_tiles = (groups + 3) // 4
    if not value.is_contiguous(memory_format=torch.channels_last):
        raise ValueError("Expected channels-last input")
    if tuple(weight.shape) != (25, channels):
        raise ValueError(
            f"Expected packed [25, {channels}] weight, got {tuple(weight.shape)}"
        )
    if tuple(bias.shape) != (channels,):
        raise ValueError(f"Expected [{channels}] bias, got {tuple(bias.shape)}")
    if block_m not in (4, 8, 16, 32):
        raise ValueError(f"Unsupported block_m={block_m}")
    if num_warps not in (4, 8):
        raise ValueError(f"Unsupported num_warps={num_warps}")

    batch, _, height, width = value.shape
    rows_per_image = height * width
    rows = batch * rows_per_image
    padded_rows = triton.cdiv(rows, 128) * 128
    quantized = torch.empty(
        (rows, channels), device=value.device, dtype=torch.float8_e4m3fn
    )
    scale_bits = torch.zeros(
        padded_rows * scale_tiles * 4, device=value.device, dtype=torch.uint8
    )
    grid = (triton.cdiv(padded_rows, block_m), groups)
    _depthwise_relu_mxfp8_kernel[grid](
        value,
        weight,
        bias,
        quantized,
        scale_bits,
        rows,
        rows_per_image,
        HEIGHT=height,
        WIDTH=width,
        BLOCK_M=block_m,
        CHANNELS=channels,
        SCALE_TILES=scale_tiles,
        num_warps=num_warps,
    )
    return quantized, scale_bits.view(torch.float8_e8m0fnu)


def _validate_parts(
    f_a: Tensor,
    f_ba: Tensor,
    displacement: Tensor,
    correlation: Tensor,
) -> tuple[int, int, int]:
    tensors = (f_a, f_ba, displacement, correlation)
    if any(tensor.ndim != 4 for tensor in tensors):
        raise ValueError("Expected four rank-4 source tensors")
    batch, height, width, channels = f_a.shape
    if channels != 192:
        raise ValueError(f"Expected f_a [B, H, W, 192], got {tuple(f_a.shape)}")
    if f_ba.shape != (batch, height, width, 192):
        raise ValueError(f"Expected f_ba [B, H, W, 192], got {tuple(f_ba.shape)}")
    if displacement.shape != (batch, 79, height, width):
        raise ValueError(
            f"Expected displacement [B, 79, H, W], got {tuple(displacement.shape)}"
        )
    if correlation.shape != (batch, 49, height, width):
        raise ValueError(
            f"Expected correlation [B, 49, H, W], got {tuple(correlation.shape)}"
        )
    if any(tensor.device != f_a.device for tensor in tensors[1:]):
        raise ValueError("All source tensors must be on the same device")
    return batch, height, width


def _assembled_parts_dtype(
    f_a: Tensor,
    f_ba: Tensor,
    displacement: Tensor,
    correlation: Tensor,
) -> torch.dtype:
    """Return the dtype produced by the generic ``torch.cat`` assembly."""
    dtype = f_a.dtype
    for source in (f_ba, displacement, correlation):
        dtype = torch.promote_types(dtype, source.dtype)
    return dtype


def _fused_depthwise_relu_mxfp8_parts(
    f_a: Tensor,
    f_ba: Tensor,
    displacement: Tensor,
    correlation: Tensor,
    weight: Tensor,
    bias: Tensor,
    *,
    block_m: int = 32,
    num_warps: int = 4,
) -> tuple[Tensor, Tensor]:
    """Produce MXFP8 activations from the four C512 block1 source planes."""
    if _depthwise_relu_mxfp8_parts_kernel is None:
        raise RuntimeError("_fused_depthwise_relu_mxfp8_parts requires Triton")
    batch, height, width = _validate_parts(f_a, f_ba, displacement, correlation)
    if tuple(weight.shape) != (25, _CHANNELS):
        raise ValueError(f"Expected packed [25, 512] weight, got {tuple(weight.shape)}")
    if tuple(bias.shape) != (_CHANNELS,):
        raise ValueError(f"Expected [512] bias, got {tuple(bias.shape)}")
    if block_m not in (4, 8, 16, 32):
        raise ValueError(f"Unsupported block_m={block_m}")
    if num_warps not in (4, 8):
        raise ValueError(f"Unsupported num_warps={num_warps}")
    assembled_dtype = _assembled_parts_dtype(f_a, f_ba, displacement, correlation)
    # The generic path promotes sources in torch.cat, then the producer casts
    # each assembled value to BF16. Match that ordering without assembling C512.
    if f_a.dtype != assembled_dtype:
        f_a = f_a.to(dtype=assembled_dtype)
    if f_ba.dtype != assembled_dtype:
        f_ba = f_ba.to(dtype=assembled_dtype)
    if displacement.dtype != assembled_dtype:
        displacement = displacement.to(dtype=assembled_dtype)
    if correlation.dtype != assembled_dtype:
        correlation = correlation.to(dtype=assembled_dtype)

    rows_per_image = height * width
    rows = batch * rows_per_image
    padded_rows = triton.cdiv(rows, 128) * 128
    quantized = torch.empty(
        (rows, _CHANNELS), device=f_a.device, dtype=torch.float8_e4m3fn
    )
    scale_bits = torch.empty(
        padded_rows * _GROUPS, device=f_a.device, dtype=torch.uint8
    )
    grid = (triton.cdiv(padded_rows, block_m), _GROUPS)
    _depthwise_relu_mxfp8_parts_kernel[grid](
        f_a,
        f_ba,
        displacement,
        correlation,
        weight,
        bias,
        quantized,
        scale_bits,
        rows,
        rows_per_image,
        *f_a.stride(),
        *f_ba.stride(),
        *displacement.stride(),
        *correlation.stride(),
        HEIGHT=height,
        WIDTH=width,
        BLOCK_M=block_m,
        num_warps=num_warps,
    )
    return quantized, scale_bits.view(torch.float8_e8m0fnu)


def _has_fused_geometry(
    depthwise: nn.Conv2d, pointwise: nn.Conv2d, channels: int
) -> bool:
    return (
        depthwise.in_channels == channels
        and depthwise.out_channels == channels
        and depthwise.groups == channels
        and depthwise.kernel_size == _DEPTHWISE_KERNEL
        and depthwise.stride == (1, 1)
        and depthwise.padding == (2, 2)
        and depthwise.dilation == (1, 1)
        and not depthwise.transposed
        and depthwise.padding_mode == "zeros"
        and pointwise.in_channels == channels
        and pointwise.out_channels == channels
        and pointwise.groups == 1
        and pointwise.kernel_size == _POINTWISE_KERNEL
        and pointwise.stride == (1, 1)
        and pointwise.padding == (0, 0)
        and pointwise.dilation == (1, 1)
        and not pointwise.transposed
        and pointwise.padding_mode == "zeros"
    )


class FusedRefinerMXFP8Block(nn.Module):
    """Inference-only C=128 or C=512 refiner block on a fused MXFP8 producer.

    Replaces a ``Block`` whose BatchNorm has been folded (``norm`` is
    ``nn.Identity``) by :func:`optimize_refiner_for_inference`. The depthwise
    convolution runs in BF16 inside the fused producer; the pointwise weight
    is packed once into the canonical MXFP8 layout used by
    :class:`romav2.inference.PortablePointwiseConv2d`. Input and output are
    channels-last BF16 tensors. The depthwise neighborhood never crosses
    batch-image boundaries.
    """

    def __init__(self, module: Block, *, compile_quantizers: bool = True) -> None:
        super().__init__()
        depthwise = module.conv_depthwise
        pointwise = module.conv_pointwise
        if not isinstance(module.norm, nn.Identity):
            raise TypeError(
                "Refiner BatchNorm must be folded first; run "
                "optimize_refiner_for_inference(model, channels_last=True)"
            )
        if not isinstance(depthwise, nn.Conv2d) or depthwise.bias is None:
            raise TypeError("Expected a folded depthwise convolution with bias")
        if not isinstance(pointwise, nn.Conv2d) or pointwise.bias is None:
            raise TypeError("Expected a pointwise convolution with bias")
        channels = next(
            (
                width
                for width in (512, 128, 32)
                if _has_fused_geometry(depthwise, pointwise, width)
            ),
            None,
        )
        if channels is None:
            raise ValueError(
                "FusedRefinerMXFP8Block requires an exact C=512 depthwise "
                "5x5 (stride=1, padding=2, dilation=1, groups=512) and "
                "pointwise 1x1 (stride=1, padding=0, dilation=1, groups=1) "
                "convolution geometry, or the same geometry at C=128 or C=32, "
                f"got depthwise={depthwise}, pointwise={pointwise}"
            )
        if depthwise.weight.device != pointwise.weight.device:
            raise RuntimeError(
                "FusedRefinerMXFP8Block requires depthwise and pointwise "
                "convolutions on the same device, got "
                f"{depthwise.weight.device} and {pointwise.weight.device}"
            )
        resolve_inference_backend("mxfp8", depthwise.weight.device)
        if _depthwise_relu_mxfp8_kernel is None:
            raise RuntimeError("FusedRefinerMXFP8Block requires Triton")
        self.channels = channels
        depthwise_weight = depthwise.weight.detach()
        self.register_buffer(
            "depthwise_weight",
            depthwise_weight[:, 0]
            .permute(1, 2, 0)
            .reshape(25, channels)
            .to(device=depthwise.weight.device, dtype=torch.bfloat16)
            .contiguous(),
        )
        self.register_buffer(
            "depthwise_bias", depthwise.bias.detach().float().contiguous()
        )
        self.pointwise = PortablePointwiseConv2d.from_conv(
            pointwise,
            backend="mxfp8",
            compile_quantizers=compile_quantizers,
        )

    def _forward_quantized(
        self,
        quantized: Tensor,
        scale: Tensor,
        batch: int,
        height: int,
        width: int,
    ) -> Tensor:
        output = F.scaled_mm(
            quantized,
            self.pointwise.linear._weight.transpose(0, 1),
            scale,
            F.ScalingType.BlockWise1x32,
            self.pointwise.linear._weight_scale,
            F.ScalingType.BlockWise1x32,
            F.SwizzleType.SWIZZLE_32_4_4,
            F.SwizzleType.SWIZZLE_32_4_4,
            bias=self.pointwise.linear._bias,
            output_dtype=torch.bfloat16,
        )
        return output.reshape(batch, height, width, self.channels).permute(0, 3, 1, 2)

    def forward(self, value: Tensor) -> Tensor:
        if value.ndim != 4 or int(value.shape[1]) != self.channels:
            raise ValueError(
                f"Expected [B, {self.channels}, H, W], got {tuple(value.shape)}"
            )
        quantized, scale = fused_depthwise_relu_mxfp8(
            value,
            self.depthwise_weight,
            self.depthwise_bias,
        )
        batch, _, height, width = value.shape
        return self._forward_quantized(quantized, scale, batch, height, width)

    def extra_repr(self) -> str:
        return f"channels={self.channels}, kernel=5x5 depthwise + 1x1 mxfp8 pointwise"


class FusedRefinerMXFP8PartsBlock(FusedRefinerMXFP8Block):
    """C512 block1 variant that consumes its projected feature planes directly."""

    supports_refiner_parts = True

    def forward_parts(
        self,
        f_a: Tensor,
        f_ba: Tensor,
        displacement: Tensor,
        correlation: Tensor,
    ) -> Tensor:
        quantized, scale = _fused_depthwise_relu_mxfp8_parts(
            f_a,
            f_ba,
            displacement,
            correlation,
            self.depthwise_weight,
            self.depthwise_bias,
        )
        batch, height, width, _ = f_a.shape
        return self._forward_quantized(quantized, scale, batch, height, width)

    def extra_repr(self) -> str:
        return (
            "channels=512, block1 parts producer + 5x5 depthwise + 1x1 mxfp8 pointwise"
        )


def _is_eligible_block(module: nn.Module, channels: int = 512) -> bool:
    depthwise = getattr(module, "conv_depthwise", None)
    pointwise = getattr(module, "conv_pointwise", None)
    return (
        isinstance(getattr(module, "norm", None), nn.Identity)
        and isinstance(depthwise, nn.Conv2d)
        and depthwise.bias is not None
        and isinstance(pointwise, nn.Conv2d)
        and pointwise.bias is not None
        and _has_fused_geometry(depthwise, pointwise, channels)
    )


def _is_parts_eligible(module: nn.Module, parent: nn.Module) -> bool:
    cfg = getattr(parent, "cfg", None)
    return (
        _is_eligible_block(module, 512)
        and getattr(cfg, "proj_dim", None) == 192
        and getattr(cfg, "displacement_emb_dim", None) == 79
        and getattr(cfg, "local_corr_radius", None) == 3
    )


def _fused_state_size(module: FusedRefinerMXFP8Block) -> tuple[int, int, int]:
    """Count parameters, bf16 source bytes, and packed bytes of a fused block."""
    linear = module.pointwise.linear
    weight_bits = linear.weight_bits
    exponent = linear.scale_exponent
    bias_numel = 0 if linear.bias_value is None else linear.bias_value.numel()
    parameters = (
        weight_bits.numel()
        + module.depthwise_weight.numel()
        + module.depthwise_bias.numel()
    )
    source_bytes = (parameters + bias_numel) * 2
    packed_bytes = (
        weight_bits.numel() * weight_bits.element_size()
        + exponent.numel() * exponent.element_size()
        + bias_numel * linear.bias_value.element_size()
        + module.depthwise_weight.numel() * module.depthwise_weight.element_size()
        + module.depthwise_bias.numel() * module.depthwise_bias.element_size()
    )
    return parameters, source_bytes, packed_bytes


def fuse_refiner_mxfp8_for_inference(
    model: nn.Module,
    *,
    skip_names: Iterable[str] = (),
    compile_quantizers: bool = True,
    enable_refiner_parts: bool = False,
    channels: int = 512,
) -> InferenceReport:
    """Replace eligible refiner blocks with fused MXFP8 blocks.

    ``channels`` is 512, 128, or 32. Requires
    :func:`romav2.optimize_refiner_for_inference` with ``channels_last=True``
    first: eligible blocks have a folded ``nn.Identity`` norm, a 5x5 depthwise
    convolution, and a 1x1 pointwise convolution at that width. ``skip_names``
    keeps specific blocks on the BF16 path. ``enable_refiner_parts`` is
    experimental, C=512 only, and defaults to ``False``. Conversion is
    idempotent; already-fused blocks are left untouched.
    """
    if channels not in (32, 128, 512):
        raise ValueError(
            f"fused refiner channels must be 32, 128, or 512, got {channels}"
        )
    if enable_refiner_parts and channels != 512:
        raise ValueError("refiner parts fusion is only supported at C=512")
    if model.training:
        raise ValueError("fuse_refiner_mxfp8_for_inference requires model.eval()")

    already_fused = 0
    blocks: dict[str, nn.Module] = {}
    eligible: dict[str, nn.Module] = {}
    for name, module in model.named_modules():
        if isinstance(module, FusedRefinerMXFP8Block):
            already_fused += 1
            continue
        if not all(
            hasattr(module, attr)
            for attr in ("conv_depthwise", "norm", "relu", "conv_pointwise")
        ):
            continue
        blocks[name] = module
        if _is_eligible_block(module, channels):
            eligible[name] = module

    skipped = set(skip_names)
    unknown = skipped - set(blocks)
    if unknown:
        raise ValueError(f"Unknown refiner block names: {sorted(unknown)}")
    selected = [name for name in eligible if name not in skipped]
    if not selected and not eligible and already_fused == 0:
        raise ValueError(
            f"No eligible C={channels} refiner blocks; run "
            "optimize_refiner_for_inference(model, channels_last=True) first"
        )
    # Validate every target before replacing any module so an unsupported or
    # mixed-device refiner cannot be left partially converted.
    for name in selected:
        depthwise = eligible[name].conv_depthwise
        pointwise = eligible[name].conv_pointwise
        if depthwise.weight.device != pointwise.weight.device:
            raise RuntimeError(
                "fuse_refiner_mxfp8_for_inference requires each block's "
                "depthwise and pointwise convolutions on the same device, got "
                f"{depthwise.weight.device} and {pointwise.weight.device} for {name}"
            )
        resolve_inference_backend("mxfp8", depthwise.weight.device)
    if selected and _depthwise_relu_mxfp8_kernel is None:
        raise RuntimeError("fuse_refiner_mxfp8_for_inference requires Triton")

    parameters = source_bytes = packed_bytes = 0
    with torch.no_grad():
        for name in selected:
            parent_name, _, child_name = name.rpartition(".")
            parent = model.get_submodule(parent_name) if parent_name else model
            block_type = (
                FusedRefinerMXFP8PartsBlock
                if enable_refiner_parts
                and child_name == "block1"
                and _is_parts_eligible(eligible[name], parent)
                else FusedRefinerMXFP8Block
            )
            replacement = block_type(
                eligible[name], compile_quantizers=compile_quantizers
            ).eval()
            parent._modules[child_name] = replacement
            layer_parameters, layer_source, layer_packed = _fused_state_size(
                replacement
            )
            parameters += layer_parameters
            source_bytes += layer_source
            packed_bytes += layer_packed

    return InferenceReport(
        recipe=REFINER_FUSED_RECIPE,
        backend="mxfp8",
        layer_count=len(selected),
        linear_parameters=parameters,
        source_bytes=source_bytes,
        packed_bytes=packed_bytes,
    )
