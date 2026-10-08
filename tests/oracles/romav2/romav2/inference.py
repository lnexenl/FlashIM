# Vendored from https://github.com/Parskatt/RoMaV2 (romav2/inference.py @ 95c9968) as the reference implementation of FlashIM's parity tests.
# Copyright (c) 2025 Johan Edstedt. SPDX-License-Identifier: MIT (see LICENSE in this directory).
# Changes: none besides this header.

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import torch
import torch.nn.functional as F
from torch import Tensor, nn

try:
    import triton
    import triton.language as tl
except ImportError:  # PyTorch builds without Triton use the compiled tensor fallback.
    triton = None
    tl = None

logger = logging.getLogger(__name__)

InferenceBackend = Literal["auto", "mxfp8", "fp8_rowwise", "bf16", "torch"]
ResolvedBackend = Literal["mxfp8", "fp8_rowwise", "bf16", "torch"]
PortableLinearScope = Literal["validated", "all_transformer", "full"]

PORTABLE_FORMAT = "romav2.portable-linear"
PORTABLE_VERSION = 1
VALIDATED_RECIPE = "revised_mxfp8_scope_wa"
ALL_TRANSFORMER_RECIPE = "all_transformer_mxfp8_scope_wa"
FULL_RECIPE = "full_mxfp8_scope_wa"
REFINER_POINTWISE_RECIPE = "refiner_pointwise_mxfp8_scope_wa"
_FP8_MAX = 448.0

_DESC_BLOCK = re.compile(r"f\.blocks\.(\d+)\.")
_MATCHER_BLOCK = re.compile(r"matcher\.mv_vit\.blocks\.(\d+)\.")

_COMPILED_MX_QUANTIZER = None
_COMPILED_MX_RELU_QUANTIZER = None
_COMPILED_ROW_QUANTIZER = None


@dataclass(frozen=True)
class InferenceReport:
    recipe: str
    backend: str | None
    layer_count: int
    linear_parameters: int
    source_bytes: int
    packed_bytes: int

    @property
    def compression_ratio(self) -> float:
        return self.source_bytes / self.packed_bytes


def _block_index(name: str, pattern: re.Pattern[str]) -> int | None:
    match = pattern.match(name)
    return int(match.group(1)) if match is not None else None


def is_validated_layer(name: str) -> bool:
    """Return whether a linear belongs to the validated RoMaV2 FP8 recipe."""
    descriptor_idx = _block_index(name, _DESC_BLOCK)
    matcher_idx = _block_index(name, _MATCHER_BLOCK)
    return (descriptor_idx is not None and 0 <= descriptor_idx <= 5) or (
        matcher_idx is not None and 4 <= matcher_idx <= 7
    )


def is_layer_in_scope(name: str, scope: PortableLinearScope) -> bool:
    """Return whether a linear belongs to a portable quantization scope."""
    if scope == "validated":
        return is_validated_layer(name)
    if scope == "all_transformer":
        return (
            _block_index(name, _DESC_BLOCK) is not None
            or _block_index(name, _MATCHER_BLOCK) is not None
            or name
            in {
                "matcher.mv_vit.projector",
                "matcher.mv_vit.output_projector",
            }
        )
    if scope == "full":
        return True
    raise ValueError(f"Unknown portable linear scope: {scope}")


def _recipe_for_scope(scope: PortableLinearScope) -> str:
    if scope == "validated":
        return VALIDATED_RECIPE
    if scope == "all_transformer":
        return ALL_TRANSFORMER_RECIPE
    if scope == "full":
        return FULL_RECIPE
    raise ValueError(f"Unknown portable linear scope: {scope}")


def _scope_for_recipe(recipe: str) -> PortableLinearScope:
    if recipe == VALIDATED_RECIPE:
        return "validated"
    if recipe == ALL_TRANSFORMER_RECIPE:
        return "all_transformer"
    if recipe == FULL_RECIPE:
        return "full"
    raise ValueError(f"Unsupported quantization recipe: {recipe!r}")


def optimize_refiner_for_inference(
    model: nn.Module, *, channels_last: bool = False
) -> int:
    """Fold refiner Conv-BatchNorm pairs and optionally use channels-last convs."""
    if model.training:
        raise ValueError("optimize_refiner_for_inference requires model.eval()")

    folded_blocks = 0
    for module in model.modules():
        is_refiner_block = all(
            hasattr(module, name)
            for name in ("conv_depthwise", "norm", "relu", "conv_pointwise")
        )
        if is_refiner_block:
            conv = module.conv_depthwise
            norm = module.norm
            if isinstance(conv, nn.Conv2d) and isinstance(norm, nn.BatchNorm2d):
                module.conv_depthwise = nn.utils.fusion.fuse_conv_bn_eval(conv, norm)
                module.norm = nn.Identity()
                folded_blocks += 1
            elif not isinstance(norm, nn.Identity):
                raise TypeError(
                    f"Unsupported refiner normalization: {type(norm).__name__}"
                )

        is_conv_refiner = all(
            hasattr(module, name)
            for name in ("block1", "hidden_blocks", "disp_emb", "warp_head")
        )
        if is_conv_refiner:
            module._inference_channels_last = channels_last
            if channels_last:
                for child in module.modules():
                    if isinstance(child, nn.Conv2d):
                        child.weight.data = child.weight.data.to(
                            memory_format=torch.channels_last
                        )

    return folded_blocks


def optimize_fine_features_for_inference(
    model: nn.Module, *, channels_last: bool = False, fold_batch_norm: bool = True
) -> int:
    """Optionally fold VGG Conv-BatchNorm pairs and use channels-last convs."""
    if model.training:
        raise ValueError("optimize_fine_features_for_inference requires model.eval()")

    from .features import VGG

    folded_pairs = 0
    for module in model.modules():
        if not isinstance(module, VGG):
            continue

        layers = module.layers
        if fold_batch_norm:
            for index in range(len(layers) - 1):
                conv = layers[index]
                norm = layers[index + 1]
                if isinstance(conv, nn.Conv2d) and isinstance(norm, nn.BatchNorm2d):
                    layers[index] = nn.utils.fusion.fuse_conv_bn_eval(conv, norm)
                    layers[index + 1] = nn.Identity()
                    folded_pairs += 1

        if channels_last:
            module._inference_channels_last = True
            for layer in layers:
                if isinstance(layer, nn.Conv2d) and not layer.weight.is_contiguous(
                    memory_format=torch.channels_last
                ):
                    layer.weight.data = layer.weight.data.to(
                        memory_format=torch.channels_last
                    )

    return folded_pairs


def _device_capability(device: torch.device) -> tuple[int, int] | None:
    if device.type != "cuda" or not torch.cuda.is_available():
        return None
    index = device.index if device.index is not None else torch.cuda.current_device()
    return torch.cuda.get_device_capability(index)


def _has_mxfp8_api() -> bool:
    return (
        hasattr(F, "scaled_mm")
        and hasattr(F, "ScalingType")
        and hasattr(F, "SwizzleType")
        and hasattr(torch, "float8_e4m3fn")
        and hasattr(torch, "float8_e8m0fnu")
    )


def _has_rowwise_fp8_api() -> bool:
    return hasattr(torch, "_scaled_mm") and hasattr(torch, "float8_e4m3fn")


def resolve_inference_backend(
    requested: InferenceBackend = "auto", device: torch.device | str | None = None
) -> ResolvedBackend:
    """Resolve a portable linear backend from the target device and available APIs."""
    if device is None:
        device = (
            torch.device("cuda", torch.cuda.current_device())
            if torch.cuda.is_available()
            else torch.device("cpu")
        )
    else:
        device = torch.device(device)

    capability = _device_capability(device)
    supported = {
        "mxfp8": capability is not None and capability >= (10, 0) and _has_mxfp8_api(),
        "fp8_rowwise": capability is not None
        and capability >= (8, 9)
        and _has_rowwise_fp8_api(),
        "bf16": capability is not None and capability >= (8, 0),
        "torch": True,
    }

    if requested == "auto":
        for backend in ("mxfp8", "fp8_rowwise", "bf16", "torch"):
            if supported[backend]:
                return backend  # type: ignore[return-value]
        raise AssertionError("The torch backend is always available")

    if requested not in supported:
        raise ValueError(f"Unknown inference backend: {requested}")
    if not supported[requested]:
        capability_text = (
            "CPU" if capability is None else f"sm_{capability[0]}{capability[1]}"
        )
        raise RuntimeError(
            f"Backend {requested!r} is unavailable on {capability_text} with torch {torch.__version__}"
        )
    return requested  # type: ignore[return-value]


def _mxfp8_exponents(blocks: Tensor) -> Tensor:
    amax = blocks.abs().amax(dim=-1)
    exponent = torch.ceil(torch.log2(amax / _FP8_MAX))
    exponent = torch.where(amax == 0, torch.full_like(exponent, -127), exponent)
    return exponent.clamp(-127, 127).to(torch.int8)


def _quantize_mxfp8_2d(value: Tensor) -> tuple[Tensor, Tensor]:
    if value.ndim != 2:
        raise ValueError(f"Expected a matrix, got shape {tuple(value.shape)}")
    rows, cols = value.shape
    if cols % 32 != 0:
        raise ValueError(f"MXFP8 requires K divisible by 32, got K={cols}")
    blocks = value.float().contiguous().reshape(rows, cols // 32, 32)
    exponent = _mxfp8_exponents(blocks)
    inverse_scale = torch.exp2(-exponent.float())
    quantized = (blocks * inverse_scale[..., None]).clamp(-_FP8_MAX, _FP8_MAX)
    return quantized.to(torch.float8_e4m3fn).reshape(rows, cols), exponent


def _scale_exponents_to_e8m0(exponent: Tensor) -> Tensor:
    biased = (exponent.to(torch.int16) + 127).to(torch.uint8)
    return biased.view(torch.float8_e8m0fnu)


def _swizzle_mxfp8_scales(exponent: Tensor) -> Tensor:
    scales = _scale_exponents_to_e8m0(exponent)
    rows, cols = scales.shape
    padded_rows = ((rows + 127) // 128) * 128
    padded_cols = ((cols + 3) // 4) * 4
    if rows != padded_rows or cols != padded_cols:
        padded = torch.zeros(
            (padded_rows, padded_cols), device=scales.device, dtype=scales.dtype
        )
        padded[:rows, :cols] = scales
        scales = padded
    return (
        scales.view(padded_rows // 128, 128, padded_cols // 4, 4)
        .permute(0, 2, 1, 3)
        .reshape(-1, 4, 32, 4)
        .transpose(1, 2)
        .reshape(-1)
        .contiguous()
    )


if triton is not None:

    @triton.jit
    def _fused_mxfp8_quantize_kernel(
        input_ptr,
        output_ptr,
        scale_bits_ptr,
        rows,
        K: tl.constexpr,
        K_TILES: tl.constexpr,
        GROUP_PROGRAMS: tl.constexpr,
        GROUPS_PER_PROGRAM: tl.constexpr,
        APPLY_RELU: tl.constexpr,
    ):
        pid = tl.program_id(0)
        row = pid // GROUP_PROGRAMS
        group_program = pid - row * GROUP_PROGRAMS
        element_offsets = tl.arange(0, GROUPS_PER_PROGRAM * 32)
        group_offsets = group_program * GROUPS_PER_PROGRAM + element_offsets // 32
        k_offsets = group_offsets * 32 + element_offsets % 32
        valid = (row < rows) & (k_offsets < K)
        input_offsets = row * K + k_offsets
        values = tl.load(input_ptr + input_offsets, mask=valid, other=0.0).to(
            tl.float32
        )
        if APPLY_RELU:
            values = tl.maximum(values, 0.0)
        blocks = tl.reshape(values, (GROUPS_PER_PROGRAM, 32))
        amax = tl.max(tl.abs(blocks), axis=1)
        exponent = tl.ceil(tl.log2(amax / 448.0))
        exponent = tl.maximum(tl.minimum(exponent, 127.0), -127.0)
        exponent = tl.where(amax == 0.0, -127.0, exponent)
        quantized = tl.maximum(
            tl.minimum(blocks * tl.exp2(-exponent)[:, None], 448.0),
            -448.0,
        )
        tl.store(
            output_ptr + input_offsets,
            tl.reshape(quantized, (GROUPS_PER_PROGRAM * 32,)),
            mask=valid,
        )

        groups = group_program * GROUPS_PER_PROGRAM + tl.arange(0, GROUPS_PER_PROGRAM)
        row_in_tile = row % 128
        scale_offsets = (
            ((row // 128) * K_TILES + groups // 4) * 512
            + (row_in_tile % 32) * 16
            + (row_in_tile // 32) * 4
            + groups % 4
        )
        padded_groups = K_TILES * 4
        valid_scales = (row < rows) & (groups < K // 32)
        scale_bits = tl.where(valid_scales, exponent + 127.0, 0.0).to(tl.uint8)
        tl.store(
            scale_bits_ptr + scale_offsets,
            scale_bits,
            mask=groups < padded_groups,
        )

else:
    _fused_mxfp8_quantize_kernel = None


def _quantize_mxfp8_activation_triton(
    value: Tensor, *, relu: bool = False
) -> tuple[Tensor, Tensor]:
    """Quantize values and emit hardware-swizzled E8M0 scales in one kernel."""
    if triton is None or _fused_mxfp8_quantize_kernel is None:
        raise RuntimeError("The fused MXFP8 quantizer requires Triton")
    if value.ndim != 2:
        raise ValueError(f"Expected a matrix, got shape {tuple(value.shape)}")
    rows, width = value.shape
    if width % 32 != 0:
        raise ValueError(f"MXFP8 requires K divisible by 32, got K={width}")
    value = value.contiguous()
    groups = width // 32
    if groups <= 4:
        groups_per_program, num_warps = 4, 4
    elif groups <= 8:
        groups_per_program, num_warps = 8, 4
    elif groups <= 24:
        groups_per_program, num_warps = 16, 4
    elif groups <= 32:
        groups_per_program, num_warps = 32, 8
    elif groups <= 96:
        groups_per_program, num_warps = 64, 4
    else:
        groups_per_program, num_warps = 128, 8
    k_tiles = triton.cdiv(groups, 4)
    padded_groups = k_tiles * 4
    group_programs = triton.cdiv(padded_groups, groups_per_program)
    padded_rows = triton.cdiv(rows, 128) * 128
    quantized = torch.empty_like(value, dtype=torch.float8_e4m3fn)
    scale_bits = torch.empty(
        padded_rows * padded_groups,
        device=value.device,
        dtype=torch.uint8,
    )
    _fused_mxfp8_quantize_kernel[(padded_rows * group_programs,)](
        value,
        quantized,
        scale_bits,
        rows,
        K=width,
        K_TILES=k_tiles,
        GROUP_PROGRAMS=group_programs,
        GROUPS_PER_PROGRAM=groups_per_program,
        APPLY_RELU=relu,
        num_warps=num_warps,
    )
    return quantized, scale_bits.view(torch.float8_e8m0fnu)


def _quantize_mxfp8_activation_triton_relu(value: Tensor) -> tuple[Tensor, Tensor]:
    return _quantize_mxfp8_activation_triton(value, relu=True)


def _quantize_mxfp8_activation(value: Tensor) -> tuple[Tensor, Tensor]:
    quantized, exponent = _quantize_mxfp8_2d(value)
    return quantized, _swizzle_mxfp8_scales(exponent)


def _quantize_rowwise_activation(value: Tensor) -> tuple[Tensor, Tensor]:
    value_float = value.float().contiguous()
    scale = (value_float.abs().amax(dim=1, keepdim=True) / _FP8_MAX).clamp_min(
        torch.finfo(torch.float32).tiny
    )
    quantized = (value_float / scale).clamp(-_FP8_MAX, _FP8_MAX)
    return quantized.to(torch.float8_e4m3fn), scale


def _compiled_quantizer(kind: Literal["mxfp8", "mxfp8_relu", "fp8_rowwise"]):
    global _COMPILED_MX_QUANTIZER, _COMPILED_MX_RELU_QUANTIZER, _COMPILED_ROW_QUANTIZER
    if not hasattr(torch, "compile"):
        if kind == "mxfp8_relu":
            raise RuntimeError("The fused ReLU MXFP8 quantizer requires Triton")
        return (
            _quantize_mxfp8_activation
            if kind == "mxfp8"
            else _quantize_rowwise_activation
        )
    if kind == "mxfp8":
        if _COMPILED_MX_QUANTIZER is None:
            if triton is not None:
                _COMPILED_MX_QUANTIZER = _quantize_mxfp8_activation_triton
            else:
                _COMPILED_MX_QUANTIZER = torch.compile(
                    _quantize_mxfp8_activation, fullgraph=True, dynamic=True
                )
        return _COMPILED_MX_QUANTIZER
    if kind == "mxfp8_relu":
        if triton is None:
            raise RuntimeError("The fused ReLU MXFP8 quantizer requires Triton")
        if _COMPILED_MX_RELU_QUANTIZER is None:
            _COMPILED_MX_RELU_QUANTIZER = _quantize_mxfp8_activation_triton_relu
        return _COMPILED_MX_RELU_QUANTIZER
    if _COMPILED_ROW_QUANTIZER is None:
        _COMPILED_ROW_QUANTIZER = torch.compile(
            _quantize_rowwise_activation, fullgraph=True, dynamic=True
        )
    return _COMPILED_ROW_QUANTIZER


def _decode_e4m3fn_bits(bits: Tensor) -> Tensor:
    raw = bits.to(torch.int32)
    sign = 1.0 - 2.0 * ((raw >> 7) & 1).float()
    exponent = (raw >> 3) & 0xF
    mantissa = raw & 0x7
    subnormal = mantissa.float() * (2.0**-9)
    normal = (1.0 + mantissa.float() / 8.0) * torch.exp2(exponent.float() - 7.0)
    decoded = torch.where(exponent == 0, subnormal, normal) * sign
    nan_mask = (exponent == 0xF) & (mantissa == 0x7)
    return torch.where(nan_mask, torch.full_like(decoded, torch.nan), decoded)


def _dequantize_portable_weight(
    weight_bits: Tensor, scale_exponent: Tensor, device: torch.device
) -> Tensor:
    rows, cols = weight_bits.shape
    quantized = _decode_e4m3fn_bits(weight_bits.to(device))
    scales = torch.exp2(scale_exponent.to(device).float()).repeat_interleave(32, dim=1)
    return (quantized * scales).reshape(rows, cols)


def _effective_bias(module: nn.Linear) -> Tensor | None:
    if module.bias is None:
        return None
    bias = module.bias.detach().float()
    bias_mask = getattr(module, "bias_mask", None)
    if bias_mask is not None:
        if not torch.isfinite(bias_mask).all():
            raise ValueError("Linear bias_mask contains non-finite values")
        bias = bias * bias_mask.float()
    bias = bias.cpu().contiguous()
    return None if torch.count_nonzero(bias).item() == 0 else bias


def _pack_linear(module: nn.Linear) -> dict[str, Tensor | None]:
    weight = module.weight.detach()
    if weight.ndim != 2 or weight.shape[1] % 32 != 0:
        raise ValueError(f"Unsupported linear weight shape: {tuple(weight.shape)}")
    if not torch.isfinite(weight).all():
        raise ValueError("Cannot quantize a linear with non-finite weights")
    quantized, exponent = _quantize_mxfp8_2d(weight)
    return {
        "weight_bits": quantized.view(torch.uint8).cpu().contiguous(),
        "scale_exponent": exponent.cpu().contiguous(),
        "bias": _effective_bias(module),
    }


class PortableLinear(nn.Module):
    """Inference-only linear with a canonical MXFP8 checkpoint and GPU dispatch."""

    _CANONICAL_BUFFERS = ("weight_bits", "scale_exponent", "bias_value")

    def __init__(
        self,
        weight_bits: Tensor,
        scale_exponent: Tensor,
        bias: Tensor | None,
        *,
        backend: ResolvedBackend,
        device: torch.device,
        compile_quantizers: bool,
        output_dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        if weight_bits.dtype != torch.uint8 or weight_bits.ndim != 2:
            raise ValueError("weight_bits must be a 2D uint8 tensor")
        if scale_exponent.dtype != torch.int8:
            raise ValueError("scale_exponent must use int8 unbiased exponents")
        expected_scale_shape = (weight_bits.shape[0], weight_bits.shape[1] // 32)
        if tuple(scale_exponent.shape) != expected_scale_shape:
            raise ValueError(
                f"Expected scale shape {expected_scale_shape}, got {tuple(scale_exponent.shape)}"
            )
        if bias is not None and tuple(bias.shape) != (weight_bits.shape[0],):
            raise ValueError(
                f"Expected bias shape {(weight_bits.shape[0],)}, got {tuple(bias.shape)}"
            )

        self.in_features = weight_bits.shape[1]
        self.out_features = weight_bits.shape[0]
        self.backend = backend
        self.compile_quantizers = compile_quantizers

        if output_dtype is not None and not output_dtype.is_floating_point:
            raise ValueError(f"output_dtype must be floating point, got {output_dtype}")
        self.output_dtype = output_dtype
        self.register_buffer("weight_bits", weight_bits.cpu().contiguous())
        self.register_buffer("scale_exponent", scale_exponent.cpu().contiguous())
        self.register_buffer(
            "bias_value", None if bias is None else bias.float().cpu().contiguous()
        )
        self.register_buffer("_weight", None, persistent=False)
        self.register_buffer("_weight_scale", None, persistent=False)
        self.register_buffer("_bias", None, persistent=False)
        self._prepare(device)

    @classmethod
    def from_linear(
        cls,
        module: nn.Linear,
        *,
        backend: ResolvedBackend,
        compile_quantizers: bool,
        output_dtype: torch.dtype | None = None,
    ) -> PortableLinear:
        packed = _pack_linear(module)
        return cls(
            packed["weight_bits"],
            packed["scale_exponent"],
            packed["bias"],
            backend=backend,
            device=module.weight.device,
            compile_quantizers=compile_quantizers,
            output_dtype=output_dtype,
        )

    def portable_state(self) -> dict[str, Tensor | None]:
        return {
            "weight_bits": self.weight_bits.detach().cpu(),
            "scale_exponent": self.scale_exponent.detach().cpu(),
            "bias": None if self.bias_value is None else self.bias_value.detach().cpu(),
        }

    def _prepare(self, device: torch.device) -> None:
        if self.backend == "mxfp8":
            self._mxfp8_native = (
                self.in_features % 16 == 0 and self.out_features % 16 == 0
            )
            if self._mxfp8_native:
                self._weight = self.weight_bits.to(device).view(torch.float8_e4m3fn)
                exponent = self.scale_exponent.to(device)
                self._weight_scale = _swizzle_mxfp8_scales(exponent)
            else:
                self._weight = _dequantize_portable_weight(
                    self.weight_bits, self.scale_exponent, device
                ).to(torch.bfloat16)
                self._weight_scale = None
            bias_dtype = torch.bfloat16
        elif self.backend == "fp8_rowwise":
            weight = _dequantize_portable_weight(
                self.weight_bits, self.scale_exponent, device
            )
            scale = (weight.abs().amax(dim=1, keepdim=True) / _FP8_MAX).clamp_min(
                torch.finfo(torch.float32).tiny
            )
            self._weight = (
                (weight / scale).clamp(-_FP8_MAX, _FP8_MAX).to(torch.float8_e4m3fn)
            )
            self._weight_scale = scale.transpose(0, 1).contiguous()
            bias_dtype = torch.bfloat16
        elif self.backend == "bf16":
            self._weight = _dequantize_portable_weight(
                self.weight_bits, self.scale_exponent, device
            ).to(torch.bfloat16)
            self._weight_scale = None
            bias_dtype = torch.bfloat16
        else:
            self._weight = _dequantize_portable_weight(
                self.weight_bits, self.scale_exponent, device
            )
            self._weight_scale = None
            bias_dtype = torch.float32

        self._bias = (
            None
            if self.bias_value is None
            else self.bias_value.to(device=device, dtype=bias_dtype)
        )

    def _apply(self, fn, recurse: bool = True):
        canonical = {name: self._buffers.pop(name) for name in self._CANONICAL_BUFFERS}
        try:
            result = super()._apply(fn, recurse=recurse)
        finally:
            self._buffers.update(canonical)
        return result

    def forward(self, value: Tensor) -> Tensor:
        if value.shape[-1] != self.in_features:
            raise ValueError(
                f"Expected input width {self.in_features}, got {value.shape[-1]}"
            )
        output_shape = (*value.shape[:-1], self.out_features)
        matrix = value.reshape(-1, self.in_features).contiguous()

        if self.backend == "mxfp8" and self._mxfp8_native:
            quantizer = (
                _compiled_quantizer("mxfp8")
                if self.compile_quantizers
                else _quantize_mxfp8_activation
            )
            quantized, scale = quantizer(matrix)
            output = F.scaled_mm(
                quantized,
                self._weight.transpose(0, 1),
                scale,
                F.ScalingType.BlockWise1x32,
                self._weight_scale,
                F.ScalingType.BlockWise1x32,
                F.SwizzleType.SWIZZLE_32_4_4,
                F.SwizzleType.SWIZZLE_32_4_4,
                bias=self._bias,
                output_dtype=torch.bfloat16,
            )
        elif self.backend == "mxfp8":
            output = F.linear(matrix.to(torch.bfloat16), self._weight, self._bias)
        elif self.backend == "fp8_rowwise":
            quantizer = (
                _compiled_quantizer("fp8_rowwise")
                if self.compile_quantizers
                else _quantize_rowwise_activation
            )
            quantized, scale = quantizer(matrix)
            output = torch._scaled_mm(
                quantized,
                self._weight.transpose(0, 1),
                scale,
                self._weight_scale,
                bias=self._bias,
                out_dtype=torch.bfloat16,
            )
        elif self.backend == "bf16":
            output = F.linear(matrix.to(torch.bfloat16), self._weight, self._bias)
        else:
            output = F.linear(matrix.float(), self._weight, self._bias)
        output = output.reshape(output_shape)
        return output if self.output_dtype is None else output.to(self.output_dtype)

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"backend={self.backend}, compile_quantizers={self.compile_quantizers}"
        )


def _pack_pointwise_conv(module: nn.Conv2d) -> dict[str, Tensor | None]:
    if (
        module.kernel_size != (1, 1)
        or module.stride != (1, 1)
        or module.padding != (0, 0)
        or module.dilation != (1, 1)
        or module.groups != 1
    ):
        raise ValueError(f"Unsupported pointwise convolution: {module}")
    weight = module.weight.detach().reshape(module.out_channels, module.in_channels)
    if weight.shape[1] % 32 != 0:
        raise ValueError(
            f"Pointwise convolution input channels must be divisible by 32, got {weight.shape[1]}"
        )
    if not torch.isfinite(weight).all():
        raise ValueError("Cannot quantize non-finite pointwise convolution weights")
    quantized, exponent = _quantize_mxfp8_2d(weight)
    bias = None if module.bias is None else module.bias.detach().float().cpu()
    return {
        "weight_bits": quantized.view(torch.uint8).cpu().contiguous(),
        "scale_exponent": exponent.cpu().contiguous(),
        "bias": None if bias is None else bias.contiguous(),
    }


class PortablePointwiseConv2d(nn.Module):
    """Inference-only 1x1 convolution backed by the portable linear dispatch."""

    def __init__(
        self,
        weight_bits: Tensor,
        scale_exponent: Tensor,
        bias: Tensor | None,
        *,
        backend: ResolvedBackend,
        device: torch.device,
        compile_quantizers: bool,
        fuse_relu: bool = False,
    ) -> None:
        super().__init__()
        self.linear = PortableLinear(
            weight_bits,
            scale_exponent,
            bias,
            backend=backend,
            device=device,
            compile_quantizers=compile_quantizers,
        )
        self.in_channels = self.linear.in_features
        self.out_channels = self.linear.out_features
        self.backend = self.linear.backend
        self.compile_quantizers = self.linear.compile_quantizers
        self.expects_relu_input = fuse_relu and self.backend == "mxfp8"

    @classmethod
    def from_conv(
        cls,
        module: nn.Conv2d,
        *,
        backend: ResolvedBackend,
        compile_quantizers: bool,
        fuse_relu: bool = False,
    ) -> PortablePointwiseConv2d:
        packed = _pack_pointwise_conv(module)
        return cls(
            packed["weight_bits"],
            packed["scale_exponent"],
            packed["bias"],
            backend=backend,
            device=module.weight.device,
            compile_quantizers=compile_quantizers,
            fuse_relu=fuse_relu,
        )

    def portable_state(self) -> dict[str, Tensor | None]:
        return self.linear.portable_state()

    def forward(self, value: Tensor) -> Tensor:
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
        if self.expects_relu_input and self.backend == "mxfp8":
            quantizer = (
                _compiled_quantizer("mxfp8_relu")
                if self.compile_quantizers
                else _quantize_mxfp8_activation_triton_relu
            )
            quantized, scale = quantizer(matrix)
            output = F.scaled_mm(
                quantized,
                self.linear._weight.transpose(0, 1),
                scale,
                F.ScalingType.BlockWise1x32,
                self.linear._weight_scale,
                F.ScalingType.BlockWise1x32,
                F.SwizzleType.SWIZZLE_32_4_4,
                F.SwizzleType.SWIZZLE_32_4_4,
                bias=self.linear._bias,
                output_dtype=torch.bfloat16,
            )
        else:
            output = self.linear(matrix)
        return output.reshape(batch, height, width, self.out_channels).permute(
            0, 3, 1, 2
        )

    def extra_repr(self) -> str:
        return (
            f"in_channels={self.in_channels}, out_channels={self.out_channels}, "
            f"backend={self.backend}, compile_quantizers={self.compile_quantizers}"
        )


def _replace_module(root: nn.Module, name: str, replacement: nn.Module) -> None:
    parent_name, _, child_name = name.rpartition(".")
    parent = root.get_submodule(parent_name) if parent_name else root
    parent._modules[child_name] = replacement


def _selected_linears(
    model: nn.Module, scope: PortableLinearScope = "validated"
) -> list[tuple[str, nn.Linear | PortableLinear]]:
    selected: list[tuple[str, nn.Linear | PortableLinear]] = []
    for name, module in model.named_modules():
        if is_layer_in_scope(name, scope) and isinstance(
            module, (nn.Linear, PortableLinear)
        ):
            selected.append((name, module))
    return selected


def _state_size(state: dict[str, Tensor | None]) -> tuple[int, int, int]:
    weight_bits = state["weight_bits"]
    exponent = state["scale_exponent"]
    bias = state["bias"]
    assert weight_bits is not None and exponent is not None
    parameters = weight_bits.numel()
    source_bytes = parameters * 2 + (0 if bias is None else bias.numel() * 2)
    packed_bytes = weight_bits.numel() + exponent.numel()
    if bias is not None:
        packed_bytes += bias.numel() * bias.element_size()
    return parameters, source_bytes, packed_bytes


def quantize_refiner_pointwise_for_inference(
    model: nn.Module,
    *,
    backend: InferenceBackend = "auto",
    compile_quantizers: bool = True,
    min_channels: int = 512,
) -> InferenceReport:
    """Replace refiner hidden pointwise convolutions with portable modules."""
    if model.training:
        raise ValueError(
            "quantize_refiner_pointwise_for_inference requires model.eval()"
        )

    selected: list[tuple[str, nn.Conv2d | PortablePointwiseConv2d]] = []
    for name, module in model.named_modules():
        if not all(
            hasattr(module, attr)
            for attr in ("conv_depthwise", "norm", "relu", "conv_pointwise")
        ):
            continue
        pointwise = module.conv_pointwise
        if not isinstance(pointwise, (nn.Conv2d, PortablePointwiseConv2d)):
            continue
        if isinstance(pointwise, nn.Conv2d) and (
            pointwise.kernel_size != (1, 1)
            or pointwise.in_channels != pointwise.out_channels
            or pointwise.in_channels % 32 != 0
            or pointwise.in_channels < min_channels
        ):
            continue
        selected.append((f"{name}.conv_pointwise", pointwise))

    if not selected:
        raise ValueError("No refiner pointwise convolutions were found")

    first = selected[0][1]
    device = (
        first.linear._weight.device
        if isinstance(first, PortablePointwiseConv2d)
        else first.weight.device
    )
    resolved = resolve_inference_backend(backend, device)

    parameters = source_bytes = packed_bytes = 0
    with torch.no_grad():
        for name, module in selected:
            state = (
                module.portable_state()
                if isinstance(module, PortablePointwiseConv2d)
                else _pack_pointwise_conv(module)
            )
            replacement = PortablePointwiseConv2d(
                state["weight_bits"],
                state["scale_exponent"],
                state["bias"],
                backend=resolved,
                device=device,
                compile_quantizers=compile_quantizers,
                fuse_relu=True,
            )
            layer_parameters, layer_source, layer_packed = _state_size(state)
            parameters += layer_parameters
            source_bytes += layer_source
            packed_bytes += layer_packed
            _replace_module(model, name, replacement)

    return InferenceReport(
        recipe=REFINER_POINTWISE_RECIPE,
        backend=resolved,
        layer_count=len(selected),
        linear_parameters=parameters,
        source_bytes=source_bytes,
        packed_bytes=packed_bytes,
    )


def convert_for_inference(
    model: nn.Module,
    *,
    backend: InferenceBackend = "auto",
    compile_quantizers: bool = True,
    linear_scope: PortableLinearScope = "validated",
) -> InferenceReport:
    """Replace a selected RoMaV2 linear scope with portable inference modules.

    The ``full`` scope includes every linear in the model. Refiner projection
    linears return FP32 to preserve the grid-sampling and local-correlation
    dtype contract; transformer linears retain the input dtype.
    """
    if model.training:
        raise ValueError("convert_for_inference requires model.eval()")
    selected = _selected_linears(model, linear_scope)
    if not selected:
        raise ValueError("No linear layers were found in the requested scope")

    first_module = selected[0][1]
    if isinstance(first_module, PortableLinear):
        device = first_module._weight.device
    else:
        device = first_module.weight.device
    resolved = resolve_inference_backend(backend, device)

    parameters = source_bytes = packed_bytes = 0
    with torch.no_grad():
        for name, module in selected:
            output_dtype = (
                torch.float32
                if linear_scope == "full" and name.startswith("refiners.")
                else None
            )
            if isinstance(module, PortableLinear):
                state = module.portable_state()
                replacement = PortableLinear(
                    state["weight_bits"],
                    state["scale_exponent"],
                    state["bias"],
                    backend=resolved,
                    device=device,
                    compile_quantizers=compile_quantizers,
                    output_dtype=output_dtype,
                )
            else:
                state = _pack_linear(module)
                replacement = PortableLinear(
                    state["weight_bits"],
                    state["scale_exponent"],
                    state["bias"],
                    backend=resolved,
                    device=module.weight.device,
                    compile_quantizers=compile_quantizers,
                    output_dtype=output_dtype,
                )
            layer_parameters, layer_source, layer_packed = _state_size(state)
            parameters += layer_parameters
            source_bytes += layer_source
            packed_bytes += layer_packed
            _replace_module(model, name, replacement)

    return InferenceReport(
        recipe=_recipe_for_scope(linear_scope),
        backend=resolved,
        layer_count=len(selected),
        linear_parameters=parameters,
        source_bytes=source_bytes,
        packed_bytes=packed_bytes,
    )


def export_portable_checkpoint(
    model: nn.Module,
    path: str | Path,
    *,
    linear_scope: PortableLinearScope = "validated",
) -> InferenceReport:
    """Write a hardware-neutral uint8/int8 overlay for a portable linear scope."""
    selected = _selected_linears(model, linear_scope)
    if not selected:
        raise ValueError("No RoMaV2 linear layers were found in the requested scope")

    layers: dict[str, dict[str, Tensor | None]] = {}
    parameters = source_bytes = packed_bytes = 0
    with torch.no_grad():
        for name, module in selected:
            state = (
                module.portable_state()
                if isinstance(module, PortableLinear)
                else _pack_linear(module)
            )
            layers[name] = state
            layer_parameters, layer_source, layer_packed = _state_size(state)
            parameters += layer_parameters
            source_bytes += layer_source
            packed_bytes += layer_packed

    recipe = _recipe_for_scope(linear_scope)
    payload = {
        "format": PORTABLE_FORMAT,
        "version": PORTABLE_VERSION,
        "recipe": recipe,
        "layers": layers,
    }
    checkpoint_path = Path(path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, checkpoint_path)
    return InferenceReport(
        recipe=recipe,
        backend=None,
        layer_count=len(layers),
        linear_parameters=parameters,
        source_bytes=source_bytes,
        packed_bytes=packed_bytes,
    )


def load_portable_checkpoint(
    model: nn.Module,
    path: str | Path,
    *,
    backend: InferenceBackend = "auto",
    compile_quantizers: bool = True,
    strict: bool = True,
) -> InferenceReport:
    """Apply a portable quantized overlay to an initialized RoMaV2 model."""
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if (
        payload.get("format") != PORTABLE_FORMAT
        or payload.get("version") != PORTABLE_VERSION
    ):
        raise ValueError("Unsupported portable checkpoint format or version")
    linear_scope = _scope_for_recipe(payload.get("recipe"))

    layers = payload.get("layers")
    if not isinstance(layers, dict) or not layers:
        raise ValueError("Portable checkpoint contains no layers")

    modules = dict(model.named_modules())
    expected = {
        name
        for name, module in modules.items()
        if is_layer_in_scope(name, linear_scope)
        and isinstance(module, (nn.Linear, PortableLinear))
    }
    checkpoint_names = set(layers)
    if strict and checkpoint_names != expected:
        missing = sorted(expected - checkpoint_names)
        unexpected = sorted(checkpoint_names - expected)
        raise ValueError(
            f"Portable checkpoint layer mismatch; missing={missing}, unexpected={unexpected}"
        )

    names = sorted(checkpoint_names & expected)
    if not names:
        raise ValueError("Portable checkpoint has no layers compatible with this model")
    first = modules[names[0]]
    device = (
        first._weight.device
        if isinstance(first, PortableLinear)
        else first.weight.device
    )
    resolved = resolve_inference_backend(backend, device)

    parameters = source_bytes = packed_bytes = 0
    with torch.no_grad():
        for name in names:
            state = layers[name]
            output_dtype = (
                torch.float32
                if linear_scope == "full" and name.startswith("refiners.")
                else None
            )
            replacement = PortableLinear(
                state["weight_bits"],
                state["scale_exponent"],
                state.get("bias"),
                backend=resolved,
                device=device,
                compile_quantizers=compile_quantizers,
                output_dtype=output_dtype,
            )
            old = modules[name]
            expected_shape = (old.out_features, old.in_features)
            if tuple(replacement.weight_bits.shape) != expected_shape:
                raise ValueError(
                    f"Layer {name} has checkpoint shape {tuple(replacement.weight_bits.shape)}, "
                    f"expected {expected_shape}"
                )
            layer_parameters, layer_source, layer_packed = _state_size(state)
            parameters += layer_parameters
            source_bytes += layer_source
            packed_bytes += layer_packed
            _replace_module(model, name, replacement)

    return InferenceReport(
        recipe=_recipe_for_scope(linear_scope),
        backend=resolved,
        layer_count=len(names),
        linear_parameters=parameters,
        source_bytes=source_bytes,
        packed_bytes=packed_bytes,
    )
