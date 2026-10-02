"""Static INT8 / MXFP8 ``nn.Conv2d`` (any kernel, stride, padding, dilation; ``groups == 1``).

The activation is quantized once with its static scale(s) on ``[B, H, W, C]``. Both dispatch to a fused
implicit-GEMM (``ops.int8_conv2d`` / ``ops.mxfp8_conv2d``: one tap per full-``Cin`` ``tl.dot`` (INT8) or one
per-32-channel-block ``tl.dot`` (MXFP8), gathered straight from the activation; no materialized im2col buffer —
see ``backends/triton/conv.py``); their reference implementations go through the unfused pipeline instead: the
*quantized bytes* gathered into the ``[B*Ho*Wo, kh*kw*C]`` im2col matrix (``ops.im2col_gather``; zero padding is
exactly zero in both formats), so the k*k-fold expansion moves 1-byte elements, not BF16, then the layer's
existing static linear (``ops.int8_linear`` / ``ops.mxfp8_linear``); the weight is laid out ``(ky, kx, c)`` to
match (also exercised directly by ``im2col_bytes`` below and its test).

MXFP8 block-32 scales are per input-channel block and shared by all taps (the calibrator takes the amax over
taps: a tap only changes which weight reads a pixel, not the pixel's own scale). The weight's own scale does
vary per tap, so the fused kernel's weight layout stays the *tiled* ``[Cout, kh*kw*C/32]`` blocks; only the
activation side is read untiled (``channel_exponent``, ``[C/32]``), once per tap.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from flashim import ops
from flashim.ops.conv import fused_mxfp8_conv2d_worthwhile
from flashim.quant.formats import mx
from flashim.quant.formats.int8 import quantize_symmetric, weight_scales
from flashim.quant.methods.int8_linear import StaticInt8Linear
from flashim.quant.methods.linear import StaticMXFP8Linear
from flashim.quant.tensor import QuantizedTensor, QuantizedWeight


def _pair(value) -> tuple[int, int]:
    return (value, value) if isinstance(value, int) else tuple(value)


def im2col_bytes(x: Tensor, kernel, stride, padding, dilation) -> tuple[Tensor, tuple[int, int]]:
    """``x`` ``[B, H, W, C]`` of a 1-byte dtype (int8 / float8) -> ``[B*Ho*Wo, kh*kw*C]`` in ``(ky, kx, c)`` order."""
    (kh, kw), (sh, sw), (ph, pw), (dh, dw) = map(_pair, (kernel, stride, padding, dilation))
    _, h, w, _ = x.shape
    ho = (h + 2 * ph - dh * (kh - 1) - 1) // sh + 1
    wo = (w + 2 * pw - dw * (kw - 1) - 1) // sw + 1
    cols = ops.im2col_gather(x, kh, kw, sh, sw, ph, pw, dh, dw)
    return cols, (ho, wo)


def _geometry(conv: nn.Conv2d) -> dict:
    if conv.groups != 1:
        raise ValueError("quantized convolutions must have groups == 1")
    if conv.padding_mode != "zeros":
        raise ValueError("quantized convolutions need zero padding")
    padding = conv.padding
    if padding == "same":  # stride 1, odd kernels: symmetric padding
        if any(k % 2 == 0 for k in conv.kernel_size) or _pair(conv.stride) != (1, 1):
            raise ValueError("padding='same' is supported for odd kernels at stride 1")
        padding = tuple(d * (k - 1) // 2 for k, d in zip(conv.kernel_size, _pair(conv.dilation)))
    elif isinstance(padding, str):
        raise ValueError(f"unsupported padding {padding!r}")
    return {"kernel": conv.kernel_size, "stride": conv.stride, "padding": padding, "dilation": conv.dilation}


class StaticInt8Conv2d(StaticInt8Linear):
    """``nn.Conv2d`` as an INT8 GEMM (per-output-channel weights, one static activation scale); BF16 out."""

    def __init__(self, weight_int8, weight_scale, bias, input_scale, *, in_channels: int, **geometry) -> None:
        super().__init__(weight_int8, weight_scale, bias, input_scale)
        self.in_channels, self.geometry = in_channels, geometry

    @classmethod
    def from_conv(cls, conv: nn.Conv2d, input_scale: Tensor) -> StaticInt8Conv2d:
        geometry = _geometry(conv)
        weight = conv.weight.detach().permute(0, 2, 3, 1).reshape(conv.out_channels, -1)  # (ky, kx, c)
        scale = weight_scales(weight)
        bias = None if conv.bias is None else conv.bias.detach().to(torch.bfloat16)
        return cls(
            quantize_symmetric(weight, scale[:, None]), scale, bias, input_scale,
            in_channels=conv.in_channels, **geometry,
        )

    def forward(self, x: Tensor) -> Tensor:
        pixels = x.permute(0, 2, 3, 1)
        q = ops.quantize_int8(pixels if pixels.is_contiguous() else pixels.contiguous(), self.input_scale)
        (kh, kw), (sh, sw), (ph, pw), (dh, dw) = (
            self.geometry["kernel"], self.geometry["stride"], self.geometry["padding"], self.geometry["dilation"],
        )
        out = ops.int8_conv2d(
            q.values, self.weight_int8, self.input_scale, self.weight_scale, self.bias,
            kh, kw, sh, sw, ph, pw, dh, dw,
        )
        return out.permute(0, 3, 1, 2).to(x.dtype)

    def extra_repr(self) -> str:
        return f"{self.in_channels}->{self.out_features}, {self.geometry['kernel']}, int8"


class StaticMXFP8Conv2d(StaticMXFP8Linear):
    """``nn.Conv2d`` as a block-scaled MXFP8 GEMM; ``input_exponent`` is per padded input-channel block.

    Two forward paths, chosen once at construction by shape (:func:`fused_mxfp8_conv2d_worthwhile`, mirroring
    ``backends/triton/conv2d``'s own eligibility gate): the fused implicit-GEMM ``ops.mxfp8_conv2d`` for shapes
    it's measured faster on, else the unfused ``im2col_gather`` + this layer's own **bound** ``self.gemm``
    (inherited from ``StaticMXFP8Linear``, prepared once at load time). The op's *reference* fallback would also
    reach the unfused pipeline for an ineligible shape, but through an *unbound* ``mxfp8_linear`` call -- for any
    shape cuBLAS's block-scaled kernel accepts (N, K both multiples of 128, common for a 256-channel 3x3 conv),
    that redoes the weight transpose + swizzle every forward call instead of once (measured: LoFTR fp8 semi-dense
    match 46.7ms bf16 -> 222.7ms with the op's reference on an ineligible large-channel conv, vs 41.3ms int8).
    Always taking this layer's own bound path for the unfused case avoids that trap regardless of future
    eligibility-gate tuning.
    """

    def __init__(self, weight: QuantizedWeight, bias, input_exponent, *, in_channels: int, taps: int, **geometry):
        super().__init__(weight, bias, input_exponent.repeat(taps))
        self.in_channels, self.taps, self.geometry = in_channels, taps, geometry
        self.register_buffer("channel_exponent", input_exponent.to(device=weight.values.device, dtype=torch.int8))
        self.use_fused = fused_mxfp8_conv2d_worthwhile(taps, input_exponent.shape[0])

    @classmethod
    def from_conv(cls, conv: nn.Conv2d, input_exponent: Tensor) -> StaticMXFP8Conv2d:
        geometry = _geometry(conv)
        cp = mx.padded_channels(conv.in_channels)
        if tuple(input_exponent.shape) != (cp // mx.BLOCK,):
            raise ValueError(f"exponent shape {tuple(input_exponent.shape)} != {(cp // mx.BLOCK,)}")
        w = F.pad(conv.weight.detach().permute(0, 2, 3, 1), (0, cp - conv.in_channels))  # [N, kh, kw, Cp]
        taps = w.shape[1] * w.shape[2]
        bias = None if conv.bias is None else conv.bias.detach()
        return cls(
            QuantizedWeight.from_float(w.reshape(conv.out_channels, -1)), bias, input_exponent,
            in_channels=conv.in_channels, taps=taps, **geometry,
        )

    def forward(self, x: Tensor) -> Tensor:
        pixels = x.permute(0, 2, 3, 1)
        q = ops.quantize_mxfp8(pixels if pixels.is_contiguous() else pixels.contiguous(), self.channel_exponent)
        (kh, kw), (sh, sw), (ph, pw), (dh, dw) = (
            self.geometry["kernel"], self.geometry["stride"], self.geometry["padding"], self.geometry["dilation"],
        )
        if self.use_fused:
            weight = self.weight_values.view(torch.float8_e4m3fn)
            out = ops.mxfp8_conv2d(
                q.values, self.channel_exponent, weight, self.weight_exponent, self.bias_bf16,
                kh, kw, sh, sw, ph, pw, dh, dw,
            )
            return out[..., : self.out_features].to(x.dtype).permute(0, 3, 1, 2)
        cols, (ho, wo) = im2col_bytes(q.values, (kh, kw), (sh, sw), (ph, pw), (dh, dw))
        act = QuantizedTensor(cols, self.input_exponent, (cols.shape[0], cols.shape[1]))
        out = self._finish(self.gemm(act, None), (cols.shape[0],), x.dtype)
        return out.reshape(x.shape[0], ho, wo, -1).permute(0, 3, 1, 2)

    def extra_repr(self) -> str:
        return f"{self.in_channels}->{self.out_features}, {self.geometry['kernel']}, kernel={self.kernel_name}"
