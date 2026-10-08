# Vendored from https://github.com/Parskatt/RoMaV2 (romav2/fast_fp8.py @ 95c9968) as the reference implementation of FlashIM's parity tests.
# Copyright (c) 2025 Johan Edstedt. SPDX-License-Identifier: MIT (see LICENSE in this directory).
# Changes: none besides this header.

"""Explicit static FP8 runtime built on pre-packed calibrated linears.

Unlike :class:`romav2.full_fp8_runtime.FullFP8Model`, this path performs no
``TorchDispatchMode`` interception, no runtime fingerprints, and no generic
fallback dispatch. Every calibrated ``nn.Linear`` is converted exactly once
into a :class:`FastFP8Linear` wrapping a pre-packed
:class:`romav2.full_fp8.StaticFP8Linear`. Structural self-attention modules
(``qkv``/``proj``/``num_heads``) are converted exactly once into a
:class:`FastFP8Attention`, which runs one static FP8 QKV GEMM, dequantizes the
packed QKV once for the native BF16 FlashAttention SDPA kernel, requantizes
its output under the projection's static input scale, and runs one static FP8
projection GEMM. This avoids the occupancy-limited FP32 Triton attention
kernel while retaining static FP8 projection boundaries.

Structural pre-norm transformer blocks (``norm1``/``attn``/``ls1``/``norm2``/
``mlp``/``ls2`` with structural attention) are recognized before both generic
linear replacement and standalone attention replacement, and converted exactly
once into a :class:`FastFP8TransformerBlock`. The block owns its
:class:`FastFP8Attention` and static MLP linears directly. On the SDPA core
it quantizes ``norm1`` into QKV, the second residual LayerNorm into ``fc1``,
and eval GELU into ``fc2``, all with the next linear's static input scale.
Training blocks and drop-path (``sample_drop_ratio``) blocks are rejected
explicitly rather than silently approximated.

Structural RoMa refiner blocks (``conv_depthwise``/``norm``/``relu``/
``conv_pointwise`` with convolution calibration at ``<name>.conv_pointwise``)
are recognized before generic linear replacement and converted exactly once
into a :class:`romav2.fast_fp8_refiner.FastFP8RefinerBlock`, which folds the
eval-mode BatchNorm2d into a deep-copied BF16 depthwise convolution and runs
depthwise → ReLU → static FP8 pointwise GEMM through
:class:`romav2.fast_fp8_refiner.FastFP8PointwiseConv`. Heads, spatial,
transposed, and DPT convolutions are never converted: they stay on the eager
BF16/FP32 path, and a structural refiner block missing its pointwise
calibration fails packing explicitly rather than silently falling back.

Graph-readiness invariant
-------------------------
After :class:`FastFP8Model` construction the wrapped model guarantees:

1. **No per-call module discovery** — the set of FP8 linears is fixed at
   pack time; forward traverses the ordinary module tree only.
2. **No per-call weight packing or fingerprinting** — weights are quantized
   once into static FP8 buffers during construction and never re-read.
3. **No dynamic quantization** — activations and outputs use the static
   block-32 E8M0 scales from calibration; no runtime amax/statistic passes.

The hot path is therefore a fixed sequence of quantize → FP8 GEMM →
dequantize per linear, suitable for CUDA-graph capture or ``torch.compile``.
"""

from __future__ import annotations

import copy
from collections.abc import Callable, Collection
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from romav2.fast_fp8_attention import fused_fp8_attention
from romav2.fast_fp8_fusion import (
    fused_residual_layernorm,
    gelu_quantize_block32,
    layernorm_quantize_block32,
    residual_layernorm_quantize_block32,
)
from romav2.fast_fp8_refiner import FastFP8PointwiseConv, FastFP8RefinerBlock
from romav2.full_fp8 import (
    QuantizedTensor,
    QuantizedWeight,
    StaticFP8Linear,
    _mxfp8_input_scale,
    quantize_block32,
)
from romav2.full_fp8_runtime import FullFP8Calibration, ModuleCalibration
from romav2.vit.attention import rope_apply
from romav2.vit.utils import cat_keep_shapes, uncat_with_shapes

__all__ = [
    "FastFP8Attention",
    "FastFP8Linear",
    "FastFP8Model",
    "FastFP8PackingReport",
    "FastFP8TransformerBlock",
]


class FastFP8Linear(nn.Module):
    """Dense-in/dense-out FP8 linear with statically packed weights and scales.

    Wraps a pre-packed :class:`StaticFP8Linear`. Construction is the only
    place where weights are quantized and the (possibly masked) bias is
    folded; ``forward`` only quantizes the input with the static calibrated
    scale, runs one FP8 GEMM, and dequantizes back to the input dtype.
    """

    static: StaticFP8Linear
    in_features: int
    out_features: int

    def __init__(
        self, static: StaticFP8Linear, in_features: int, out_features: int
    ) -> None:
        super().__init__()
        self.static = static
        self.in_features = in_features
        self.out_features = out_features
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

    def _native_forward(self, value: Tensor) -> Tensor:
        quantized = quantize_block32(value, self.static.input_scale_exponent)
        output = self.static.native_mxfp8(
            quantized, self._native_input_scale(quantized.matrix().shape[0])
        )
        if self.static.mxfp8_bias is not None:
            output = output + self.static.mxfp8_bias
        return (
            output[:, : self.out_features]
            .reshape(*value.shape[:-1], self.out_features)
            .to(value.dtype)
        )

    @classmethod
    def from_linear(
        cls, linear: nn.Linear, calibration: ModuleCalibration
    ) -> FastFP8Linear:
        """Pack ``linear`` once using static calibrated input/output scales."""
        weight = QuantizedWeight.from_float(linear.weight.detach())
        bias = None
        if linear.bias is not None:
            effective = linear.bias.detach()
            bias_mask = getattr(linear, "bias_mask", None)
            if bias_mask is not None:
                # DINO LinearKMaskedBias: fold the registered mask once so the
                # hot path carries an ordinary dense bias.
                if not torch.isfinite(bias_mask).all():
                    raise ValueError(
                        f"bias_mask of {type(linear).__name__} contains non-finite values"
                    )
                effective = effective * bias_mask.to(effective.dtype)
            bias = F.pad(effective, (0, weight.out_features - linear.out_features))
        static = StaticFP8Linear(
            weight,
            bias,
            calibration.output_scale_exponent,
            input_scale_exponent=calibration.input_scale_exponent,
        )
        return cls(static, linear.in_features, linear.out_features)

    def forward(self, value: Tensor) -> Tensor:
        """Quantize with the static scale, run the FP8 GEMM, dequantize densely.

        Arbitrary leading dimensions are preserved; the output dtype matches
        the input dtype.
        """
        self.call_count += 1
        if self.static.supports_native_mxfp8:
            return self._native_forward(value)
        return self.static(value).dequantize(dtype=value.dtype)

    def _matches_static_input_scale(self, scale: Tensor) -> bool:
        expected = self.static.input_scale_exponent
        if expected is None or not isinstance(scale, Tensor):
            return False
        if scale is expected:
            return True
        if (
            scale.dtype != expected.dtype
            or scale.device != expected.device
            or scale.shape != expected.shape
        ):
            return False
        return bool(torch.equal(scale, expected))

    def forward_prequantized(self, quantized: QuantizedTensor) -> Tensor:
        """Run native MXFP8 on a packed activation and return BF16.

        ``quantized.scale_exponent`` must be this linear's static input scale:
        the same tensor, or an equal int8 tensor on the same device. Dense
        ``forward`` is unchanged and still quantizes its own input.
        """
        self.call_count += 1
        if not isinstance(quantized, QuantizedTensor):
            raise TypeError(
                "forward_prequantized expects a QuantizedTensor, got "
                f"{type(quantized).__name__}"
            )
        if not self.static.supports_native_mxfp8:
            raise ValueError(
                "forward_prequantized requires a native MXFP8 weight "
                "(padded K and N divisible by 128)"
            )
        if not self._matches_static_input_scale(quantized.scale_exponent):
            raise ValueError(
                "forward_prequantized scale_exponent does not match the "
                "static input scale"
            )
        output = self.static.native_mxfp8(
            quantized, self._native_input_scale(quantized.matrix().shape[0])
        )
        if self.static.mxfp8_bias is not None:
            output = output + self.static.mxfp8_bias
        leading = quantized.logical_shape[:-1]
        return (
            output[:, : self.out_features]
            .reshape(*leading, self.out_features)
            .to(torch.bfloat16)
        )

    def forward_quantized(self, value: Tensor | QuantizedTensor) -> QuantizedTensor:
        """Run the FP8 GEMM and return the packed output without dequantizing.

        Dense inputs are quantized with the static calibrated input scale;
        an already-packed :class:`QuantizedTensor` (e.g. a fused-attention
        output produced under this linear's input scale) is consumed directly.
        """
        self.call_count += 1
        return self.static(value)


class FastFP8Attention(nn.Module):
    """Static FP8 attention with selectable SDPA or fused FP8 core."""

    qkv: nn.Module
    proj: nn.Module
    num_heads: int
    attention_core: str

    def __init__(
        self,
        module: nn.Module,
        qkv_calibration: ModuleCalibration,
        proj_calibration: ModuleCalibration,
        *,
        attention_core: str = "sdpa",
        keep_qkv: bool = False,
        keep_proj: bool = False,
    ) -> None:
        super().__init__()
        if attention_core not in ("sdpa", "fp8"):
            raise ValueError(
                "attention_core must be one of {'sdpa', 'fp8'}, "
                f"got {attention_core!r}"
            )
        if attention_core == "fp8" and (keep_qkv or keep_proj):
            raise ValueError(
                "attention_core='fp8' requires both attention projections to be FP8"
            )
        qkv_linear = getattr(module, "qkv", None)
        proj_linear = getattr(module, "proj", None)
        if not isinstance(qkv_linear, nn.Linear) or not isinstance(
            proj_linear, nn.Linear
        ):
            raise TypeError(
                "FastFP8Attention requires qkv/proj nn.Linear submodules, got "
                f"{type(qkv_linear).__name__}/{type(proj_linear).__name__}"
            )
        dim = qkv_linear.in_features
        if qkv_linear.out_features != 3 * dim or proj_linear.in_features != dim:
            raise ValueError(
                "Fast FP8 attention requires a standard QKV packing "
                f"(qkv {dim} -> {3 * dim}, proj {dim} -> {dim}); grouped-query "
                "or asymmetric attention is not supported"
            )
        num_heads = getattr(module, "num_heads", None)
        if not isinstance(num_heads, int) or num_heads <= 0 or dim % num_heads != 0:
            raise ValueError(
                f"Fast FP8 attention requires a positive int num_heads dividing "
                f"the dim {dim}, got {num_heads}"
            )
        head_dim = dim // num_heads
        if head_dim % 32 != 0 or head_dim & (head_dim - 1) != 0:
            raise ValueError(
                "Fast FP8 attention requires a head_dim that is a power of two "
                f"divisible by 32, got {head_dim}"
            )
        for submodule in module.modules():
            if isinstance(submodule, nn.Dropout) and submodule.p != 0:
                raise ValueError(
                    "Fast FP8 attention does not support dropout, got "
                    f"nn.Dropout(p={submodule.p})"
                )
        for name in ("attn_drop", "proj_drop"):
            value = getattr(module, name, None)
            if isinstance(value, (int, float)) and float(value) != 0.0:
                raise ValueError(
                    f"Fast FP8 attention does not support dropout, got {name}={value}"
                )
        self.qkv = (
            qkv_linear
            if keep_qkv
            else FastFP8Linear.from_linear(qkv_linear, qkv_calibration)
        )
        self.proj = (
            proj_linear
            if keep_proj
            else FastFP8Linear.from_linear(proj_linear, proj_calibration)
        )
        self.num_heads = num_heads
        self.attention_core = attention_core
        self.call_count = 0

    @staticmethod
    def _normalize_is_causal(attn_bias: Any, is_causal: bool) -> bool:
        """Normalize positional causal calls into the causal flag."""
        if isinstance(attn_bias, bool):
            return bool(attn_bias) or is_causal
        if attn_bias is not None:
            raise ValueError(
                "Fast FP8 attention does not support attention masks/bias, got "
                f"attn_bias of type {type(attn_bias).__name__}"
            )
        return is_causal

    @staticmethod
    def _normalize_rope(
        rope: tuple[Tensor, Tensor] | None,
    ) -> tuple[Tensor, Tensor] | None:
        if rope is None:
            return None
        sin, cos = rope
        # RoPE tables are per-position and shared across batch/heads; flatten
        # leading singleton dims ([1, 1, N, D] -> [N, D]) for the kernel.
        return sin.reshape(-1, sin.shape[-1]), cos.reshape(-1, cos.shape[-1])

    def _attention(self, qkv: Tensor, rope, is_causal: bool) -> Tensor:
        """Run native BF16 FlashAttention after the static FP8 QKV projection."""
        batch, tokens, three_channels = qkv.shape
        channels = three_channels // 3
        head_dim = channels // self.num_heads
        query, key, value = (
            qkv.view(batch, tokens, 3, self.num_heads, head_dim)
            .permute(2, 0, 3, 1, 4)
            .unbind(0)
        )
        normalized_rope = self._normalize_rope(rope)
        if normalized_rope is not None:
            sin, cos = normalized_rope
            sin = sin.to(device=query.device, dtype=query.dtype)
            cos = cos.to(device=query.device, dtype=query.dtype)
            prefix = tokens - sin.shape[-2]
            if prefix < 0:
                raise ValueError(
                    f"RoPE length {sin.shape[-2]} exceeds token count {tokens}"
                )
            if prefix != tokens:
                query = torch.cat(
                    (
                        query[..., :prefix, :],
                        rope_apply(query[..., prefix:, :], sin, cos),
                    ),
                    dim=-2,
                )
                key = torch.cat(
                    (key[..., :prefix, :], rope_apply(key[..., prefix:, :], sin, cos)),
                    dim=-2,
                )
        attended = F.scaled_dot_product_attention(
            query, key, value, dropout_p=0.0, is_causal=is_causal
        )
        return attended.transpose(1, 2).reshape(batch, tokens, channels)

    def forward(
        self,
        x: Tensor,
        attn_bias=None,
        rope: tuple[Tensor, Tensor] | None = None,
        is_causal: bool = False,
    ) -> Tensor:
        self.call_count += 1
        is_causal = self._normalize_is_causal(attn_bias, is_causal)
        if self.attention_core == "fp8":
            return self._attention_fp8(
                self.qkv.forward_quantized(x), rope, is_causal, x.dtype
            )
        return self.proj(self._attention(self.qkv(x), rope, is_causal))

    def _attention_fp8(
        self,
        qkv: QuantizedTensor,
        rope: tuple[Tensor, Tensor] | None,
        is_causal: bool,
        dtype: torch.dtype,
    ) -> Tensor:
        attended = fused_fp8_attention(
            qkv,
            num_heads=self.num_heads,
            output_scale_exponent=self.proj.static.input_scale_exponent,
            rope=self._normalize_rope(rope),
            is_causal=is_causal,
        )
        return self.proj.forward_quantized(attended).dequantize(dtype=dtype)

    def forward_list(
        self,
        x_list: list[Tensor],
        attn_bias=None,
        rope_list: list[tuple[Tensor, Tensor] | None] | None = None,
        is_causal: bool = False,
    ) -> list[Tensor]:
        """Run attention independently per list item while sharing projections."""
        self.call_count += 1
        is_causal = self._normalize_is_causal(attn_bias, is_causal)
        if rope_list is None:
            rope_list = [None] * len(x_list)
        if len(x_list) != len(rope_list):
            raise ValueError(
                f"x_list and rope_list must have the same length, got "
                f"{len(x_list)} and {len(rope_list)}"
            )
        x_flat, shapes, num_tokens = cat_keep_shapes(x_list)
        if self.attention_core != "fp8":
            qkv_flat = self.qkv(x_flat)
            attended_items: list[Tensor] = []
            start = 0
            for shape, tokens, rope in zip(shapes, num_tokens, rope_list):
                stop = start + tokens
                leading = shape[:-1]
                qkv_item = qkv_flat[start:stop].reshape(*leading, qkv_flat.shape[-1])
                attended_items.append(self._attention(qkv_item, rope, is_causal))
                start = stop
            attended = torch.cat([item.flatten(0, -2) for item in attended_items])
            return uncat_with_shapes(self.proj(attended), shapes, num_tokens)

        qkv_flat = self.qkv.forward_quantized(x_flat)
        attended_items: list[QuantizedTensor] = []
        start = 0
        for shape, tokens, rope in zip(shapes, num_tokens, rope_list):
            stop = start + tokens
            leading = shape[:-1]
            qkv_item = QuantizedTensor(
                qkv_flat.values[start:stop].reshape(*leading, qkv_flat.channels),
                qkv_flat.scale_exponent,
                tuple(leading) + (qkv_flat.logical_shape[-1],),
                qkv_flat.layout,
            )
            attended_items.append(
                fused_fp8_attention(
                    qkv_item,
                    num_heads=self.num_heads,
                    output_scale_exponent=self.proj.static.input_scale_exponent,
                    rope=self._normalize_rope(rope),
                    is_causal=is_causal,
                )
            )
            start = stop
        attended = QuantizedTensor(
            torch.cat(
                [item.values.reshape(-1, item.channels) for item in attended_items]
            ),
            attended_items[0].scale_exponent,
            (sum(num_tokens), attended_items[0].logical_shape[-1]),
            attended_items[0].layout,
        )
        projected = self.proj.forward_quantized(attended).dequantize(dtype=x_flat.dtype)
        return uncat_with_shapes(projected, shapes, num_tokens)

    def forward_list_prequantized(
        self,
        quantized: QuantizedTensor,
        shapes: list[tuple[int, ...]],
        num_tokens: list[int],
        rope_list: list[tuple[Tensor, Tensor] | None] | None = None,
        dtype: torch.dtype | None = None,
    ) -> list[Tensor]:
        """SDPA list path whose QKV activation is already packed.

        Proj stays on dense :meth:`FastFP8Linear.forward`. ``attention_core``
        ``"fp8"`` is rejected so that core keeps ``forward_quantized``.
        """
        self.call_count += 1
        if self.attention_core == "fp8":
            raise ValueError(
                "forward_list_prequantized only covers the SDPA core; "
                "attention_core='fp8' keeps the packed QKV kernel"
            )
        if not isinstance(quantized, QuantizedTensor):
            raise TypeError(
                "forward_list_prequantized expects a QuantizedTensor, got "
                f"{type(quantized).__name__}"
            )
        if rope_list is None:
            rope_list = [None] * len(shapes)
        if len(shapes) != len(rope_list) or len(shapes) != len(num_tokens):
            raise ValueError(
                "shapes, num_tokens, and rope_list must have the same length, "
                f"got {len(shapes)}, {len(num_tokens)}, and {len(rope_list)}"
            )
        qkv_out = self.qkv.forward_prequantized(quantized)
        if dtype is not None and qkv_out.dtype != dtype:
            qkv_out = qkv_out.to(dtype)
        qkv_flat = qkv_out.reshape(-1, qkv_out.shape[-1])
        if int(qkv_flat.shape[0]) != sum(num_tokens):
            raise ValueError(
                f"prequantized QKV has {qkv_flat.shape[0]} rows, but "
                f"num_tokens sums to {sum(num_tokens)}"
            )
        attended_items: list[Tensor] = []
        start = 0
        for shape, tokens, rope in zip(shapes, num_tokens, rope_list):
            stop = start + tokens
            leading = shape[:-1]
            qkv_item = qkv_flat[start:stop].reshape(*leading, qkv_flat.shape[-1])
            attended_items.append(self._attention(qkv_item, rope, False))
            start = stop
        attended = torch.cat([item.flatten(0, -2) for item in attended_items])
        return uncat_with_shapes(self.proj(attended), shapes, num_tokens)


def _native_static_linear(module: nn.Module | None) -> FastFP8Linear | None:
    if (
        isinstance(module, FastFP8Linear)
        and module.static.supports_native_mxfp8
        and module.static.input_scale_exponent is not None
    ):
        return module
    return None


def _dropout_changes_values(module: nn.Module) -> bool:
    drop = getattr(module, "drop", None)
    if drop is not None and not isinstance(drop, (nn.Dropout, nn.Identity)):
        return True
    for child in module.modules():
        if isinstance(child, nn.Dropout) and child.training and child.p != 0:
            return True
    return False


class FastFP8TransformerBlock(nn.Module):
    """Static pre-norm transformer block owning FP8 projections and MLP.

    Replaces a structural ``SelfAttentionBlock``-style module
    (``norm1``/``attn``/``ls1``/``norm2``/``mlp``/``ls2`` with structural
    attention). The block owns its children outright: ``norm1``/``ls1``/
    ``norm2``/``ls2`` are the deep-copied source modules, ``attn`` is a
    :class:`FastFP8Attention`, and every ``nn.Linear`` inside ``mlp`` has
    been replaced by a :class:`FastFP8Linear`. The forward mirrors the source
    evaluation path —

    ``x_attn = x + ls1(attn(norm1(x), rope))``

    ``x_ffn  = x_attn + ls2(mlp(norm2(x_attn)))``

    — and, on the SDPA core, quantizes those boundaries into the next native
    GEMM. ``norm1`` is packed with QKV's static input scale, the residual
    LayerNorm is packed with ``fc1``'s scale, and eval GELU is packed with
    ``fc2``'s scale. Proj, both layer-scales, both residuals, and a
    dropout-after-``fc2`` stay outside that packing. The FP8 attention core
    keeps its packed QKV kernel. List inputs still run over the flattened
    token matrix: one QKV GEMM, one SDPA call per item, one projection GEMM,
    and one GEMM per MLP linear. Inference-only: conversion rejects training
    blocks and drop-path (``sample_drop_ratio``) blocks explicitly.
    """

    norm1: nn.LayerNorm
    attn: FastFP8Attention
    ls1: nn.Module
    norm2: nn.LayerNorm
    mlp: nn.Module
    ls2: nn.Module

    def __init__(
        self,
        norm1: nn.LayerNorm,
        attn: FastFP8Attention,
        ls1: nn.Module,
        norm2: nn.LayerNorm,
        mlp: nn.Module,
        ls2: nn.Module,
    ) -> None:
        super().__init__()
        if not isinstance(norm1, nn.LayerNorm) or not isinstance(norm2, nn.LayerNorm):
            raise TypeError(
                "FastFP8TransformerBlock requires nn.LayerNorm pre-norms, got "
                f"{type(norm1).__name__}/{type(norm2).__name__}"
            )
        if not isinstance(attn, FastFP8Attention):
            raise TypeError(
                "FastFP8TransformerBlock requires a FastFP8Attention, got "
                f"{type(attn).__name__}"
            )
        for name, child in (("ls1", ls1), ("mlp", mlp), ("ls2", ls2)):
            if not isinstance(child, nn.Module):
                raise TypeError(
                    f"FastFP8TransformerBlock requires {name} to be an nn.Module, "
                    f"got {type(child).__name__}"
                )
        self.norm1 = norm1
        self.attn = attn
        self.ls1 = ls1
        self.norm2 = norm2
        self.mlp = mlp
        self.ls2 = ls2
        self.call_count = 0
        self.eval()

    def _mlp_from_fc1(self, hidden: Tensor, dtype: torch.dtype) -> Tensor:
        """Finish the MLP after ``fc1``, quantizing eval GELU into ``fc2``."""
        if hidden.dtype != dtype:
            hidden = hidden.to(dtype)
        mlp = self.mlp
        fc2 = _native_static_linear(getattr(mlp, "fc2", None))
        act = getattr(mlp, "act", None)
        drop = getattr(mlp, "drop", None)
        if (
            fc2 is not None
            and isinstance(act, nn.GELU)
            and not self.training
            and not _dropout_changes_values(mlp)
        ):
            packed = gelu_quantize_block32(
                hidden,
                fc2.static.input_scale_exponent,
                approximate=act.approximate,
            )
            hidden = fc2.forward_prequantized(packed)
            if hidden.dtype != dtype:
                hidden = hidden.to(dtype)
            if drop is not None:
                hidden = drop(hidden)
            return hidden
        hidden = act(hidden)
        if drop is not None:
            hidden = drop(hidden)
        hidden = mlp.fc2(hidden)
        if drop is not None:
            hidden = drop(hidden)
        if hidden.dtype != dtype:
            hidden = hidden.to(dtype)
        return hidden

    def _forward_list(
        self,
        x_list: list[Tensor],
        rope_list: list[tuple[Tensor, Tensor] | None],
    ) -> list[Tensor]:
        """Flattened list forward with static-scale boundary quantization.

        SDPA QKV, native ``fc1``, and eval GELU into native ``fc2`` consume a
        packed activation. Proj and the FP8 attention core do not.
        """
        if len(x_list) != len(rope_list):
            raise ValueError(
                f"x_list and rope_list must have the same length, got "
                f"{len(x_list)} and {len(rope_list)}"
            )
        if not x_list:
            return []
        x_flat, shapes, num_tokens = cat_keep_shapes(x_list)
        qkv = _native_static_linear(self.attn.qkv)
        if qkv is not None and self.attn.attention_core != "fp8":
            normed1 = layernorm_quantize_block32(
                x_flat,
                self.norm1.normalized_shape,
                qkv.static.input_scale_exponent,
                weight=self.norm1.weight,
                bias=self.norm1.bias,
                eps=self.norm1.eps,
            )
            attended_list = self.attn.forward_list_prequantized(
                normed1,
                shapes,
                num_tokens,
                rope_list=rope_list,
                dtype=x_flat.dtype,
            )
        else:
            normed1 = uncat_with_shapes(self.norm1(x_flat), shapes, num_tokens)
            attended_list = self.attn.forward_list(normed1, rope_list=rope_list)
        attended = cat_keep_shapes(attended_list)[0]
        attended = self.ls1(attended)
        x_attn = x_flat + attended
        fc1 = _native_static_linear(getattr(self.mlp, "fc1", None))
        if (
            fc1 is not None
            and attended.dtype == x_flat.dtype
            and hasattr(self.mlp, "act")
            and hasattr(self.mlp, "fc2")
        ):
            normed2 = residual_layernorm_quantize_block32(
                attended,
                x_flat,
                self.norm2.normalized_shape,
                fc1.static.input_scale_exponent,
                weight=self.norm2.weight,
                bias=self.norm2.bias,
                eps=self.norm2.eps,
            )
            hidden = self._mlp_from_fc1(fc1.forward_prequantized(normed2), x_flat.dtype)
        elif attended.dtype == x_flat.dtype:
            normed2 = fused_residual_layernorm(
                attended,
                x_flat,
                self.norm2.normalized_shape,
                weight=self.norm2.weight,
                bias=self.norm2.bias,
                eps=self.norm2.eps,
            )
            hidden = self.mlp(normed2)
        else:
            hidden = self.mlp(self.norm2(x_attn))
        x_ffn = x_attn + self.ls2(hidden)
        return uncat_with_shapes(x_ffn, shapes, num_tokens)

    def forward(
        self,
        x_or_x_list: Tensor | list[Tensor],
        rope_or_rope_list: tuple[Tensor, Tensor]
        | list[tuple[Tensor, Tensor] | None]
        | None = None,
    ) -> Tensor | list[Tensor]:
        """Mirror ``SelfAttentionBlock.forward``: tensor in, tensor out; list in,
        list out. A tensor input is wrapped into a single-item list exactly like
        the source implementation."""
        self.call_count += 1
        if isinstance(x_or_x_list, Tensor):
            return self._forward_list([x_or_x_list], [rope_or_rope_list])[0]
        if isinstance(x_or_x_list, list):
            if rope_or_rope_list is None:
                rope_or_rope_list = [None for _ in x_or_x_list]
            return self._forward_list(x_or_x_list, rope_or_rope_list)
        raise AssertionError(
            "FastFP8TransformerBlock expects a Tensor or a list of Tensors, got "
            f"{type(x_or_x_list).__name__}"
        )


@dataclass(frozen=True)
class FastFP8PackingReport:
    """Static byte-footprint facts for a packed :class:`FastFP8Model`.

    ``parameters``/``parameter_bytes_before``/``parameter_bytes_after`` cover
    every statically packed GEMM — the replaced ``nn.Linear`` modules plus the
    refiner pointwise convolutions packed as :class:`FastFP8PointwiseConv`.
    Eager modules (norms, depthwise convolutions, heads, DPT convolutions)
    are not counted.
    """

    linear_modules: int
    parameters: int
    parameter_bytes_before: int
    parameter_bytes_after: int
    pointwise_conv_modules: int = 0
    refiner_blocks: int = 0

    @property
    def compression_ratio(self) -> float:
        return self.parameter_bytes_before / self.parameter_bytes_after


def _is_structural_attention(
    module: nn.Module, full_name: str, calibration: FullFP8Calibration
) -> bool:
    """Recognize DINO/local self-attention modules by structure.

    A module qualifies when it exposes ``qkv``/``proj`` ``nn.Linear``
    projections, an integer ``num_heads``, and calibration entries at
    ``<full_name>.qkv`` / ``<full_name>.proj``.
    """
    return (
        isinstance(getattr(module, "qkv", None), nn.Linear)
        and isinstance(getattr(module, "proj", None), nn.Linear)
        and isinstance(getattr(module, "num_heads", None), int)
        and f"{full_name}.qkv" in calibration.linear
        and f"{full_name}.proj" in calibration.linear
    )


def _is_structural_transformer_block(
    module: nn.Module, full_name: str, calibration: FullFP8Calibration
) -> bool:
    """Recognize DINO pre-norm self-attention blocks by structure.

    A module qualifies when it exposes the ``norm1``/``attn``/``ls1``/
    ``norm2``/``mlp``/``ls2`` seam with ``nn.LayerNorm`` pre-norms, layer-scale
    (or identity) residuals, an MLP child, and a structural self-attention
    child at ``attn`` (calibration entries at ``<full_name>.attn.qkv`` /
    ``<full_name>.attn.proj``).
    """
    return (
        isinstance(getattr(module, "norm1", None), nn.LayerNorm)
        and isinstance(getattr(module, "norm2", None), nn.LayerNorm)
        and isinstance(getattr(module, "ls1", None), nn.Module)
        and isinstance(getattr(module, "ls2", None), nn.Module)
        and isinstance(getattr(module, "mlp", None), nn.Module)
        and _is_structural_attention(
            getattr(module, "attn", None), f"{full_name}.attn", calibration
        )
    )


def _is_structural_refiner_block(module: nn.Module) -> bool:
    """Recognize RoMa refiner blocks by structure.

    A module qualifies when it exposes the ``conv_depthwise``/``norm``/
    ``relu``/``conv_pointwise`` seam with a depthwise ``nn.Conv2d``, an
    eval-foldable normalization (``nn.BatchNorm2d``, or ``nn.Identity`` when
    the source model was already BN-folded), a ``relu`` attribute, and a
    pointwise ``nn.Conv2d``. Conversion additionally requires a convolution
    calibration entry at ``<full_name>.conv_pointwise``; the remaining shape
    constraints are validated explicitly by
    :class:`FastFP8RefinerBlock`/:class:`FastFP8PointwiseConv` construction.
    """
    return (
        isinstance(getattr(module, "conv_depthwise", None), nn.Conv2d)
        and isinstance(getattr(module, "norm", None), (nn.BatchNorm2d, nn.Identity))
        and hasattr(module, "relu")
        and isinstance(getattr(module, "conv_pointwise", None), nn.Conv2d)
    )


def _attention_group(model: nn.Module, full_name: str) -> str | None:
    """Return a configured attention group, without guessing model topology."""
    if full_name == "f" or full_name.startswith("f."):
        return "dinov3"
    prefix = "matcher.mv_vit.blocks."
    if not full_name.startswith(prefix):
        return None
    remainder = full_name[len(prefix) :]
    index_text = remainder.split(".", 1)[0]
    if not index_text.isdigit():
        return None
    matcher = getattr(model, "matcher", None)
    vit = getattr(matcher, "mv_vit", None)
    if (
        getattr(vit, "multiview", False) is not True
        or getattr(vit, "mv_attention_mode", None) != "alternating"
    ):
        return None
    return "mv_frame" if int(index_text) % 2 else "mv_global"


class FastFP8Model(nn.Module):
    """Deep-copied eval model whose calibrated linears run as static FP8.

        Every ``nn.Linear`` (including DINO ``LinearKMaskedBias`` subclasses) is
        validated against the calibration coverage and then replaced exactly once
        by a :class:`FastFP8Linear`. Structural self-attention modules
        (``qkv``/``proj``/``num_heads``) are recognized first and replaced exactly
        once by a :class:`FastFP8Attention` running the fused single-launch FP8
    attention kernel; their projections are packed as part of the wrapper and
    never visited by the generic linear replacement. Structural pre-norm
    transformer blocks (``norm1``/``attn``/``ls1``/``norm2``/``mlp``/``ls2``)
    are recognized before both and replaced exactly once by a
    :class:`FastFP8TransformerBlock` owning its fused attention and static MLP
    linears; blocks with training or drop-path behavior are rejected
    explicitly. Structural refiner blocks (``conv_depthwise``/``norm``/
    ``relu``/``conv_pointwise``) are recognized before generic linear
    replacement and replaced exactly once by a
    :class:`FastFP8RefinerBlock` running a folded eager BF16 depthwise
    convolution followed by a static FP8 pointwise GEMM; a refiner block
    missing its ``<name>.conv_pointwise`` convolution calibration raises
    instead of silently staying eager, and head/spatial/transposed/DPT
    convolutions are never converted. ``keep_bf16_modules`` may name any module
    path (or subtree) to retain in BF16, allowing precision-sensitive regions
    to be protected while all other eligible modules use FP8. For a RoMa pair
    with matching tensor shapes, the copied model batches the independent
    refiner image branches. The input model is never mutated.
    """

    def __init__(
        self,
        model: nn.Module,
        calibration: FullFP8Calibration,
        *,
        keep_bf16_modules: Collection[str] = (),
        attention_core: str = "sdpa",
        attention_groups: Collection[str] = ("dinov3", "mv_frame", "mv_global"),
    ) -> None:
        super().__init__()
        if attention_core not in ("sdpa", "fp8"):
            raise ValueError(
                "attention_core must be one of {'sdpa', 'fp8'}, "
                f"got {attention_core!r}"
            )
        valid_attention_groups = frozenset(("dinov3", "mv_frame", "mv_global"))
        selected_attention_groups = frozenset(attention_groups)
        unknown_attention_groups = sorted(
            selected_attention_groups - valid_attention_groups
        )
        if unknown_attention_groups:
            raise ValueError(
                "attention_groups contains unsupported groups: "
                + ", ".join(unknown_attention_groups)
            )
        self.attention_core = attention_core
        self.attention_groups = selected_attention_groups
        if not torch.cuda.is_available() or torch.cuda.get_device_capability() < (
            12,
            0,
        ):
            raise RuntimeError("Fast FP8 execution requires an SM120-or-newer CUDA GPU")
        self.model = copy.deepcopy(model).eval()
        self.model._batch_refiner_pair_branches = True

        protected = frozenset(keep_bf16_modules)
        if any(not name for name in protected):
            raise ValueError("keep_bf16_modules cannot contain an empty module path")
        module_names = {name for name, _ in self.model.named_modules()}
        unknown = sorted(protected - module_names)
        if unknown:
            raise ValueError(
                "keep_bf16_modules contains unknown module paths: " + ", ".join(unknown)
            )
        self._keep_bf16_modules = protected

        def is_protected(name: str) -> bool:
            return any(
                name == root or name.startswith(f"{root}.") for root in protected
            )

        linears = {
            name: module
            for name, module in self.model.named_modules()
            if isinstance(module, nn.Linear) and not is_protected(name)
        }
        missing = sorted(name for name in linears if name not in calibration.linear)
        if missing:
            raise RuntimeError(
                "Fast FP8 calibration is missing scales for linear modules: "
                + ", ".join(missing)
            )

        def attention_core_for(name: str) -> str:
            group = _attention_group(self.model, name)
            if self.attention_core == "fp8" and (
                group is None or group in self.attention_groups
            ):
                return "fp8"
            return "sdpa"

        refiner_block_names = {
            name
            for name, module in self.model.named_modules()
            if _is_structural_refiner_block(module) and not is_protected(name)
        }
        missing_convolution = sorted(
            f"{name}.conv_pointwise"
            for name in refiner_block_names
            if f"{name}.conv_pointwise" not in calibration.convolution
        )
        if missing_convolution:
            raise RuntimeError(
                "Fast FP8 calibration is missing scales for refiner pointwise "
                "convolutions: " + ", ".join(missing_convolution)
            )

        parameters = 0
        bytes_before = 0
        bytes_after = 0
        replaced: list[FastFP8Linear] = []
        attentions: list[FastFP8Attention] = []
        transformer_blocks: list[FastFP8TransformerBlock] = []
        refiner_blocks: list[FastFP8RefinerBlock] = []
        pointwise_convs: list[FastFP8PointwiseConv] = []

        def _account(
            original: nn.Module,
            packed: FastFP8Linear | FastFP8PointwiseConv,
            registry: list,
        ) -> None:
            nonlocal parameters, bytes_before, bytes_after
            registry.append(packed)
            parameters += sum(p.numel() for p in original.parameters())
            bytes_before += sum(
                p.numel() * p.element_size() for p in original.parameters()
            )
            bytes_after += sum(
                b.numel() * b.element_size() for b in packed.static.buffers()
            )

        def _replace(container: nn.Module, prefix: str) -> None:
            for child_name, child in container.named_children():
                full_name = f"{prefix}{child_name}"
                if is_protected(full_name):
                    continue
                if _is_structural_transformer_block(
                    child, full_name, calibration
                ) and not is_protected(f"{full_name}.attn"):
                    if child.training:
                        raise ValueError(
                            "Fast FP8 does not support training structural "
                            f"transformer blocks, got training=True at {full_name!r}"
                        )
                    sample_drop_ratio = getattr(child, "sample_drop_ratio", 0.0)
                    if sample_drop_ratio:
                        raise ValueError(
                            "Fast FP8 does not support drop-path structural "
                            "transformer blocks, got "
                            f"sample_drop_ratio={sample_drop_ratio} at {full_name!r}"
                        )
                    keep_qkv = is_protected(f"{full_name}.attn.qkv")
                    keep_proj = is_protected(f"{full_name}.attn.proj")
                    fast_attention = FastFP8Attention(
                        child.attn,
                        calibration.linear[f"{full_name}.attn.qkv"],
                        calibration.linear[f"{full_name}.attn.proj"],
                        attention_core=attention_core_for(f"{full_name}.attn"),
                        keep_qkv=keep_qkv,
                        keep_proj=keep_proj,
                    )
                    if isinstance(fast_attention.qkv, FastFP8Linear):
                        _account(child.attn.qkv, fast_attention.qkv, replaced)
                    if isinstance(fast_attention.proj, FastFP8Linear):
                        _account(child.attn.proj, fast_attention.proj, replaced)
                    mlp = child.mlp
                    if not is_protected(f"{full_name}.mlp"):
                        _replace(mlp, f"{full_name}.mlp.")
                    fast_block = FastFP8TransformerBlock(
                        norm1=child.norm1,
                        attn=fast_attention,
                        ls1=child.ls1,
                        norm2=child.norm2,
                        mlp=mlp,
                        ls2=child.ls2,
                    )
                    setattr(container, child_name, fast_block)
                    transformer_blocks.append(fast_block)
                    attentions.append(fast_attention)
                elif _is_structural_attention(child, full_name, calibration):
                    keep_qkv = is_protected(f"{full_name}.qkv")
                    keep_proj = is_protected(f"{full_name}.proj")
                    fast_attention = FastFP8Attention(
                        child,
                        calibration.linear[f"{full_name}.qkv"],
                        calibration.linear[f"{full_name}.proj"],
                        attention_core=attention_core_for(full_name),
                        keep_qkv=keep_qkv,
                        keep_proj=keep_proj,
                    )
                    setattr(container, child_name, fast_attention)
                    attentions.append(fast_attention)
                    if isinstance(fast_attention.qkv, FastFP8Linear):
                        _account(child.qkv, fast_attention.qkv, replaced)
                    if isinstance(fast_attention.proj, FastFP8Linear):
                        _account(child.proj, fast_attention.proj, replaced)
                elif _is_structural_refiner_block(child):
                    fast_refiner_block = FastFP8RefinerBlock.from_block(
                        child, calibration.convolution[f"{full_name}.conv_pointwise"]
                    )
                    setattr(container, child_name, fast_refiner_block)
                    refiner_blocks.append(fast_refiner_block)
                    _account(
                        child.conv_pointwise,
                        fast_refiner_block.conv_pointwise,
                        pointwise_convs,
                    )
                elif isinstance(child, nn.Linear):
                    fast = FastFP8Linear.from_linear(
                        child, calibration.linear[full_name]
                    )
                    setattr(container, child_name, fast)
                    _account(child, fast, replaced)
                else:
                    _replace(child, full_name + ".")

        _replace(self.model, "")
        self.model.eval()

        self._fast_linears = replaced
        self._fast_attentions = attentions
        self._fast_transformer_blocks = transformer_blocks
        self._fast_refiner_blocks = refiner_blocks
        self._fast_pointwise_convs = pointwise_convs
        self._packing_report = FastFP8PackingReport(
            linear_modules=len(replaced),
            parameters=parameters,
            parameter_bytes_before=bytes_before,
            parameter_bytes_after=bytes_after,
            pointwise_conv_modules=len(pointwise_convs),
            refiner_blocks=len(refiner_blocks),
        )
        self.last_dispatch_stats: dict[str, int] = {}

    @property
    def packing_report(self) -> FastFP8PackingReport:
        return self._packing_report

    def _execute(self, function: Callable[..., Any], *args, **kwargs):
        for fast in self._fast_linears:
            fast.call_count = 0
        for attention in self._fast_attentions:
            attention.call_count = 0
        for block in self._fast_transformer_blocks:
            block.call_count = 0
        for conv in self._fast_pointwise_convs:
            conv.call_count = 0
        result = function(*args, **kwargs)
        self.last_dispatch_stats = {
            "fp8_linear_calls": sum(fast.call_count for fast in self._fast_linears),
            "fp8_attention_calls": sum(
                attention.call_count for attention in self._fast_attentions
            ),
            "fp8_transformer_block_calls": sum(
                block.call_count for block in self._fast_transformer_blocks
            ),
            "fp8_pointwise_conv_calls": sum(
                conv.call_count for conv in self._fast_pointwise_convs
            ),
            # FastFP8RefinerBlock carries no counter of its own; its forward
            # runs its owned pointwise GEMM exactly once, so the block call
            # count is the sum over the block-owned pointwise convs.
            "fp8_refiner_block_calls": sum(
                block.conv_pointwise.call_count for block in self._fast_refiner_blocks
            ),
            "fallback_calls": 0,
        }
        return result

    def forward(self, *args, **kwargs):
        return self._execute(self.model, *args, **kwargs)

    def capture_graph(self, *args, **kwargs):
        """Capture fixed-shape forward execution in a CUDA graph."""
        from romav2.fast_fp8_graph import FastFP8CudaGraph

        return FastFP8CudaGraph(self).capture(*args, **kwargs)

    def match(self, img_like_A, img_like_B):
        from romav2.romav2 import _map_confidence

        img_A = self.model._load_image(img_like_A)
        img_B = self.model._load_image(img_like_B)
        img_A_lr = F.interpolate(
            img_A,
            size=(self.model.H_lr, self.model.W_lr),
            mode="bicubic",
            align_corners=False,
            antialias=True,
        )
        img_B_lr = F.interpolate(
            img_B,
            size=(self.model.H_lr, self.model.W_lr),
            mode="bicubic",
            align_corners=False,
            antialias=True,
        )
        if self.model.H_hr is None or self.model.W_hr is None:
            img_A_hr = img_B_hr = None
        else:
            img_A_hr = F.interpolate(
                img_A,
                size=(self.model.H_hr, self.model.W_hr),
                mode="bicubic",
                align_corners=False,
                antialias=True,
            )
            img_B_hr = F.interpolate(
                img_B,
                size=(self.model.H_hr, self.model.W_hr),
                mode="bicubic",
                align_corners=False,
                antialias=True,
            )
        predictions = self._execute(
            self.model,
            img_A_lr,
            img_B_lr,
            img_A_hr=img_A_hr,
            img_B_hr=img_B_hr,
        )
        warp_AB = predictions["warp_AB"]
        confidence_AB = predictions["confidence_AB"]
        warp_BA = predictions["warp_BA"]
        confidence_BA = predictions["confidence_BA"]
        overlap_AB, precision_AB = _map_confidence(
            confidence=confidence_AB, threshold=self.model.threshold
        )
        if self.model.bidirectional:
            overlap_BA, precision_BA = _map_confidence(
                confidence=confidence_BA, threshold=self.model.threshold
            )
        else:
            overlap_BA = precision_BA = None
        return {
            "warp_AB": warp_AB.clone(),
            "confidence_AB": confidence_AB.clone(),
            "overlap_AB": overlap_AB.clone(),
            "precision_AB": precision_AB.clone(),
            "warp_BA": warp_BA.clone() if warp_BA is not None else None,
            "confidence_BA": confidence_BA.clone()
            if confidence_BA is not None
            else None,
            "overlap_BA": overlap_BA.clone() if overlap_BA is not None else None,
            "precision_BA": precision_BA.clone() if precision_BA is not None else None,
        }

    def sample(self, *args, **kwargs):
        return self.model.sample(*args, **kwargs)

    def to_pixel_coordinates(self, *args, **kwargs):
        return self.model.to_pixel_coordinates(*args, **kwargs)
