# Vendored from https://github.com/Parskatt/RoMaV2 (romav2/full_fp8_runtime.py @ 95c9968) as the reference implementation of FlashIM's parity tests.
# Copyright (c) 2025 Johan Edstedt. SPDX-License-Identifier: MIT (see LICENSE in this directory).
# Changes: none besides this header.

from __future__ import annotations

from collections.abc import Callable, Iterable
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
from pathlib import Path
from threading import RLock
from typing import Any, Self

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.nn.attention import SDPBackend, sdpa_kernel
from torch.utils._python_dispatch import TorchDispatchMode, _disable_current_modes

from romav2.full_fp8 import (
    QuantizedTensor,
    QuantizedWeight,
    calibrate_block32_scale,
    fp8_gemm,
    fp8_gemm_fp32,
    quantize_block32,
)

_FP8 = torch.float8_e4m3fn
_FP8_MAX = 448.0
_META_ATTR = "_romav2_full_fp8_meta"
_CONVOLUTION_OPS = {
    torch.ops.aten.convolution.default,
    torch.ops.aten.conv2d.default,
    torch.ops.aten.conv_transpose2d.input,
}
_ALIAS_OPS = {
    torch.ops.aten.alias.default,
    torch.ops.aten.detach.default,
    torch.ops.aten.view.default,
    torch.ops.aten._unsafe_view.default,
    torch.ops.aten.reshape.default,
    torch.ops.aten.expand.default,
    torch.ops.aten.permute.default,
    torch.ops.aten.transpose.int,
    torch.ops.aten.slice.Tensor,
    torch.ops.aten.select.int,
    torch.ops.aten.unsqueeze.default,
    torch.ops.aten.squeeze.dim,
    torch.ops.aten.squeeze.default,
    torch.ops.aten.flatten.using_ints,
}
_TRANSPOSED_WEIGHT_ATTR = "_romav2_full_fp8_transposed_weight"
_AUTOCAST_PATCH_LOCK = RLock()


@contextmanager
def _without_nested_autocast():
    original_autocast = torch.autocast

    def disabled_autocast(device_type, *args, **kwargs):
        kwargs["enabled"] = False
        return original_autocast(device_type, *args, **kwargs)

    with _AUTOCAST_PATCH_LOCK:
        torch.autocast = disabled_autocast
        try:
            yield
        finally:
            torch.autocast = original_autocast


@dataclass(frozen=True)
class ModuleCalibration:
    input_scale_exponent: Tensor
    output_scale_exponent: Tensor

    def cpu(self) -> ModuleCalibration:
        return ModuleCalibration(
            self.input_scale_exponent.detach().to(device="cpu", dtype=torch.int8),
            self.output_scale_exponent.detach().to(device="cpu", dtype=torch.int8),
        )


@dataclass(frozen=True)
class FullFP8Calibration:
    linear: dict[str, ModuleCalibration]
    convolution: dict[str, ModuleCalibration]

    def save(self, path: str | Path) -> None:
        payload = {
            "format": "romav2.full-fp8-calibration",
            "version": 1,
            "linear": {
                name: {
                    "input_scale_exponent": value.input_scale_exponent.cpu(),
                    "output_scale_exponent": value.output_scale_exponent.cpu(),
                }
                for name, value in self.linear.items()
            },
            "convolution": {
                name: {
                    "input_scale_exponent": value.input_scale_exponent.cpu(),
                    "output_scale_exponent": value.output_scale_exponent.cpu(),
                }
                for name, value in self.convolution.items()
            },
        }
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(payload, path)

    @classmethod
    def load(cls, path: str | Path) -> FullFP8Calibration:
        payload = torch.load(path, map_location="cpu", weights_only=True)
        if payload.get("format") != "romav2.full-fp8-calibration" or payload.get(
            "version"
        ) != 1:
            raise ValueError(f"Unsupported full-FP8 calibration file: {path}")

        def load_group(group: dict[str, dict[str, Tensor]]) -> dict[str, ModuleCalibration]:
            return {
                name: ModuleCalibration(
                    value["input_scale_exponent"].to(torch.int8),
                    value["output_scale_exponent"].to(torch.int8),
                )
                for name, value in group.items()
            }

        return cls(load_group(payload["linear"]), load_group(payload["convolution"]))


class FullFP8Calibrator:
    """Collect static block-32 input/output scales for every learned linear and convolution."""

    def __init__(self, model: nn.Module) -> None:
        self.model = model
        self._input: dict[str, Tensor] = {}
        self._output: dict[str, Tensor] = {}
        self._linear_names: set[str] = set()
        self._conv_names: set[str] = set()
        self._handles: list[Any] = []

    @staticmethod
    def _channel_matrix(value: Tensor, module: nn.Module) -> Tensor:
        if isinstance(module, nn.Linear):
            return value
        if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d)):
            if value.ndim != 4:
                raise ValueError(f"Expected NCHW convolution tensor, got {tuple(value.shape)}")
            return value.permute(0, 2, 3, 1)
        raise TypeError(type(module).__name__)

    @staticmethod
    def _merge(current: Tensor | None, update: Tensor) -> Tensor:
        update = update.detach().to(device="cpu", dtype=torch.int8)
        return update if current is None else torch.maximum(current, update)

    def _record(self, name: str, module: nn.Module, value: Tensor, *, output: bool) -> None:
        if not value.is_floating_point() or value.numel() == 0:
            return
        matrix = self._channel_matrix(value.detach(), module)
        scale = calibrate_block32_scale(matrix)
        target = self._output if output else self._input
        target[name] = self._merge(target.get(name), scale)

    def __enter__(self) -> Self:
        if self._handles:
            raise RuntimeError("FullFP8Calibrator is already active")

        for name, module in self.model.named_modules():
            if isinstance(module, nn.Linear):
                self._linear_names.add(name)
            elif isinstance(module, (nn.Conv2d, nn.ConvTranspose2d)):
                self._conv_names.add(name)
            else:
                continue

            def pre_hook(mod: nn.Module, args: tuple[Any, ...], module_name: str = name):
                if args and isinstance(args[0], Tensor):
                    self._record(module_name, mod, args[0], output=False)

            def post_hook(
                mod: nn.Module,
                args: tuple[Any, ...],
                result: Any,
                module_name: str = name,
            ):
                if isinstance(result, Tensor):
                    self._record(module_name, mod, result, output=True)

            self._handles.append(module.register_forward_pre_hook(pre_hook))
            self._handles.append(module.register_forward_hook(post_hook))
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()

    def result(self, *, strict: bool = True) -> FullFP8Calibration:
        missing = sorted(
            name
            for name in self._linear_names | self._conv_names
            if name not in self._input or name not in self._output
        )
        if missing and strict:
            raise RuntimeError(
                "Calibration did not execute every weighted module: " + ", ".join(missing)
            )

        def build(names: set[str]) -> dict[str, ModuleCalibration]:
            return {
                name: ModuleCalibration(self._input[name], self._output[name])
                for name in sorted(names)
                if name in self._input and name in self._output
            }

        return FullFP8Calibration(build(self._linear_names), build(self._conv_names))


@dataclass(frozen=True)
class _TensorMeta:
    kind: str
    scale_exponent: Tensor
    logical_shape: tuple[int, ...]
    module_name: str | None = None
    packed_weight: QuantizedWeight | None = None
    source_values: Tensor | None = None
    source_meta: _TensorMeta | None = None
    view_func: Any | None = None
    view_args: tuple[Any, ...] = ()
    view_kwargs: dict[str, Any] | None = None


@dataclass(frozen=True)
class FullFP8PackingReport:
    parameters: int
    parameter_bytes_before: int
    parameter_bytes_after: int
    linear_weights: int
    convolution_weights: int

    @property
    def compression_ratio(self) -> float:
        return self.parameter_bytes_before / self.parameter_bytes_after


@dataclass
class _PackedState:
    parameter_meta: dict[int, _TensorMeta]
    storage_meta: dict[int, _TensorMeta]
    weight_fingerprints: dict[tuple[tuple[int, ...], bytes], _TensorMeta]
    calibration: FullFP8Calibration
    report: FullFP8PackingReport

def _scale_exponent(value: Tensor) -> Tensor:
    if value.numel() == 0:
        return torch.tensor(-127, device=value.device, dtype=torch.int8)
    amax = torch.nan_to_num(
        value.detach().float().abs().amax(), nan=0.0, posinf=_FP8_MAX, neginf=_FP8_MAX
    )
    exponent = torch.ceil(torch.log2(amax / _FP8_MAX))
    exponent = torch.where(amax == 0, torch.full_like(exponent, -127), exponent)
    return exponent.clamp(-127, 127).to(torch.int8)


def _attach_meta(value: Tensor, meta: _TensorMeta) -> Tensor:
    setattr(value, _META_ATTR, meta)
    return value


def _quantize_per_tensor(value: Tensor, exponent: Tensor | None = None) -> Tensor:
    if exponent is None:
        exponent = _scale_exponent(value)
    exponent = exponent.to(device=value.device, dtype=torch.int8)
    inverse_scale = torch.exp2(-exponent.float())
    quantized = (value.float() * inverse_scale).clamp(-_FP8_MAX, _FP8_MAX).to(_FP8)
    return _attach_meta(
        quantized,
        _TensorMeta("per_tensor", exponent, tuple(value.shape)),
    )


def _quantize_axis0(value: Tensor) -> tuple[Tensor, Tensor]:
    rows = value.float().reshape(value.shape[0], -1)
    amax = torch.nan_to_num(rows.abs().amax(dim=1), nan=0.0, posinf=_FP8_MAX)
    exponent = torch.ceil(torch.log2(amax / _FP8_MAX))
    exponent = torch.where(amax == 0, torch.full_like(exponent, -127), exponent)
    exponent = exponent.clamp(-127, 127).to(torch.int8)
    broadcast = torch.exp2(-exponent.float()).reshape(
        value.shape[0], *((1,) * (value.ndim - 1))
    )
    quantized = (value.float() * broadcast).clamp(-_FP8_MAX, _FP8_MAX).to(_FP8)
    return quantized, exponent


def _quantized_tensor_to_raw(value: QuantizedTensor) -> Tensor:
    logical_channels = value.logical_shape[-1]
    raw = value.values[..., :logical_channels]
    return _attach_meta(
        raw,
        _TensorMeta(
            "block32_last",
            value.scale_exponent,
            value.logical_shape,
        ),
    )


def _quantize_channel_block32(value: Tensor, exponent: Tensor) -> Tensor:
    nhwc = value.permute(0, 2, 3, 1)
    quantized = quantize_block32(nhwc, exponent)
    raw = quantized.values[..., : value.shape[1]].permute(0, 3, 1, 2)
    return _attach_meta(
        raw,
        _TensorMeta("channel_block32", exponent, tuple(value.shape)),
    )


def _meta_for(value: Tensor, state: _PackedState) -> _TensorMeta | None:
    metadata = state.parameter_meta.get(id(value))
    if metadata is None and value.numel() > 0:
        metadata = state.storage_meta.get(value.untyped_storage().data_ptr())
    if metadata is not None:
        return metadata
    return getattr(value, _META_ATTR, None)


def _dequantize_with_meta(value: Tensor, meta: _TensorMeta) -> Tensor:
    if meta.kind == "view":
        if (
            meta.source_values is None
            or meta.source_meta is None
            or meta.view_func is None
        ):
            raise RuntimeError("Incomplete FP8 view metadata")
        source_dense = _dequantize_with_meta(meta.source_values, meta.source_meta)
        source_scale = torch.where(
            meta.source_values.float() == 0,
            torch.ones_like(source_dense),
            source_dense / meta.source_values.float(),
        )
        with _disable_current_modes():
            scale = meta.view_func(
                source_scale,
                *meta.view_args,
                **(meta.view_kwargs or {}),
            )
        return value.float() * scale
    if meta.kind == "per_tensor":
        return value.float() * torch.exp2(meta.scale_exponent.float())
    if meta.kind == "axis0":
        scale = torch.exp2(meta.scale_exponent.float()).reshape(
            value.shape[0], *((1,) * (value.ndim - 1))
        )
        return value.float() * scale
    if meta.kind == "linear_weight":
        if meta.packed_weight is None:
            raise RuntimeError("Missing packed linear weight")
        return meta.packed_weight.dequantize()
    if meta.kind == "block32_last":
        logical_channels = value.shape[-1]
        padded_channels = meta.scale_exponent.numel() * 32
        padded = F.pad(value.float(), (0, padded_channels - logical_channels))
        blocks = padded.reshape(-1, padded_channels // 32, 32)
        scales = torch.exp2(meta.scale_exponent.float())[None, :, None]
        return (blocks * scales).reshape(*value.shape[:-1], padded_channels)[
            ..., :logical_channels
        ]
    if meta.kind == "channel_block32":
        nhwc = value.permute(0, 2, 3, 1)
        logical_channels = nhwc.shape[-1]
        padded_channels = meta.scale_exponent.numel() * 32
        padded = F.pad(nhwc.float(), (0, padded_channels - logical_channels))
        blocks = padded.reshape(-1, padded_channels // 32, 32)
        scales = torch.exp2(meta.scale_exponent.float())[None, :, None]
        dense = (blocks * scales).reshape(*nhwc.shape[:-1], padded_channels)[
            ..., :logical_channels
        ]
        return dense.permute(0, 3, 1, 2)
    raise RuntimeError(f"Unknown full-FP8 tensor metadata kind: {meta.kind}")


def _dequantize(value: Tensor, state: _PackedState) -> Tensor:
    meta = _meta_for(value, state)
    return value if meta is None else _dequantize_with_meta(value, meta)


def _tree_map(value: Any, function: Callable[[Any], Any]) -> Any:
    if isinstance(value, tuple):
        return tuple(_tree_map(item, function) for item in value)
    if isinstance(value, list):
        return [_tree_map(item, function) for item in value]
    if isinstance(value, dict):
        return {key: _tree_map(item, function) for key, item in value.items()}
    return function(value)


def _contains_quantized(value: Any, state: _PackedState) -> bool:
    if isinstance(value, Tensor):
        return value.dtype == _FP8 or _meta_for(value, state) is not None
    if isinstance(value, (tuple, list)):
        return any(_contains_quantized(item, state) for item in value)
    if isinstance(value, dict):
        return any(_contains_quantized(item, state) for item in value.values())
    return False


def _pack_parameter_per_tensor(parameter: nn.Parameter) -> tuple[nn.Parameter, _TensorMeta]:
    exponent = _scale_exponent(parameter)
    quantized = _quantize_per_tensor(parameter.detach(), exponent)
    packed = nn.Parameter(quantized, requires_grad=False)
    meta = _TensorMeta("per_tensor", exponent, tuple(parameter.shape))
    return packed, meta


def _weight_fingerprint(value: Tensor) -> tuple[tuple[int, ...], bytes]:
    bits = value.detach().view(torch.uint8).reshape(-1)
    if bits.numel() > 32:
        bits = torch.cat((bits[:16], bits[-16:]))
    return tuple(value.shape), bytes(bits.cpu().tolist())


def _pack_model(model: nn.Module, calibration: FullFP8Calibration) -> _PackedState:
    if model.training:
        raise ValueError("Literal full-FP8 packing requires model.eval()")
    parameter_meta: dict[int, _TensorMeta] = {}
    storage_meta: dict[int, _TensorMeta] = {}
    weight_fingerprints: dict[tuple[tuple[int, ...], bytes], _TensorMeta] = {}
    parameters = before = after = linear_weights = convolution_weights = 0

    for module_name, module in model.named_modules():
        for parameter_name, parameter in list(module.named_parameters(recurse=False)):
            if parameter is None or not parameter.is_floating_point():
                continue
            parameters += parameter.numel()
            before += parameter.numel() * parameter.element_size()

            if parameter_name == "weight" and isinstance(module, nn.Linear):
                packed_weight = QuantizedWeight.from_float(parameter.detach())
                logical = packed_weight.values[
                    : parameter.shape[0], : parameter.shape[1]
                ].contiguous()
                replacement = nn.Parameter(logical, requires_grad=False)
                meta = _TensorMeta(
                    "linear_weight",
                    packed_weight.scale_exponent,
                    tuple(parameter.shape),
                    module_name,
                    packed_weight,
                )
                linear_weights += 1
                after += packed_weight.values.numel() + packed_weight.scale_exponent.numel()
            elif parameter_name == "weight" and isinstance(
                module, (nn.Conv2d, nn.ConvTranspose2d)
            ):
                quantized, exponent = _quantize_axis0(parameter.detach())
                replacement = nn.Parameter(quantized, requires_grad=False)
                packed_weight = None
                kind = "axis0"
                if isinstance(module, nn.Conv2d) and module.kernel_size == (1, 1):
                    packed_weight = QuantizedWeight.from_float(
                        parameter.detach().reshape(parameter.shape[0], -1)
                    )
                    kind = "linear_weight"
                meta = _TensorMeta(
                    kind,
                    exponent,
                    tuple(parameter.shape),
                    module_name,
                    packed_weight,
                )
                convolution_weights += 1
                after += quantized.numel() + exponent.numel()
            elif parameter_name == "bias" and isinstance(
                module, (nn.Linear, nn.Conv2d, nn.ConvTranspose2d)
            ):
                quantized = _quantize_per_tensor(parameter.detach())
                replacement = nn.Parameter(quantized, requires_grad=False)
                meta = getattr(quantized, _META_ATTR)
                _attach_meta(replacement, meta)
                after += quantized.numel() + meta.scale_exponent.numel()
            else:
                after += parameter.numel() * parameter.element_size()
                continue
            if parameter_name == "bias":
                del module._parameters[parameter_name]
                module.register_parameter("bias_fp8", replacement)
                dense_bias = _dequantize_with_meta(replacement, meta).to(
                    parameter.dtype
                )
                module.register_buffer("bias", dense_bias)
            else:
                setattr(module, parameter_name, replacement)
            parameter_meta[id(replacement)] = meta
            storage_meta[replacement.untyped_storage().data_ptr()] = meta
            if parameter_name == "weight":
                weight_fingerprints[_weight_fingerprint(replacement)] = meta

    # Non-weight buffers (including matcher positional frequencies) stay in
    # their source dtype; they are not inference GEMM operands.

    report = FullFP8PackingReport(
        parameters,
        before,
        after,
        linear_weights,
        convolution_weights,
    )
    return _PackedState(
        parameter_meta, storage_meta, weight_fingerprints, calibration, report
    )


class FullFP8DispatchMode(TorchDispatchMode):
    """Keep floating activations in E4M3 and run GEMM/BMM on FP8 tensor cores."""

    def __init__(self, state: _PackedState) -> None:
        super().__init__()
        self.state = state
        self.fp8_gemm_calls = 0
        self.fp8_bmm_calls = 0
        self.fp8_pointwise_conv_calls = 0
        self.fallback_calls = 0

    def _linear(self, args: tuple[Any, ...], kwargs: dict[str, Any]) -> Tensor | None:
        if len(args) < 2 or not isinstance(args[0], Tensor) or not isinstance(args[1], Tensor):
            return None
        activation = args[0]
        transposed_weight = getattr(args[1], _TRANSPOSED_WEIGHT_ATTR, None)
        if not isinstance(transposed_weight, _TensorMeta) or transposed_weight.packed_weight is None:
            return None
        module_name = transposed_weight.module_name
        if module_name is None or module_name not in self.state.calibration.linear:
            return None
        calibration = self.state.calibration.linear[module_name]
        dense_activation = _dequantize(activation, self.state)
        quantized_activation = quantize_block32(
            dense_activation, calibration.input_scale_exponent.to(activation.device)
        )
        bias = None
        if len(args) >= 3 and isinstance(args[2], Tensor):
            dense_bias = _dequantize(args[2], self.state).float()
            bias = F.pad(
                dense_bias,
                (0, transposed_weight.packed_weight.out_features - dense_bias.numel()),
            )
        output = fp8_gemm(
            quantized_activation,
            transposed_weight.packed_weight,
            calibration.output_scale_exponent.to(activation.device),
            bias=bias,
        )
        self.fp8_gemm_calls += 1
        return output.dequantize(dtype=torch.bfloat16)

    def _bmm(self, left: Tensor, right: Tensor) -> Tensor:
        left_dense = _dequantize(left, self.state)
        right_dense = _dequantize(right, self.state)
        if left_dense.ndim != 3 or right_dense.ndim != 3:
            raise ValueError("FP8 BMM expects rank-3 tensors")
        outputs = []
        for batch_index in range(left_dense.shape[0]):
            left_matrix = left_dense[batch_index]
            right_matrix = right_dense[batch_index]
            left_quantized = quantize_block32(
                left_matrix, calibrate_block32_scale(left_matrix)
            )
            right_weight = QuantizedWeight.from_float(right_matrix.transpose(0, 1))
            outputs.append(fp8_gemm_fp32(left_quantized, right_weight))
        self.fp8_bmm_calls += len(outputs)
        return torch.stack(outputs).to(torch.bfloat16)

    def _resolve_weight_meta(self, value: Tensor) -> _TensorMeta | None:
        metadata = _meta_for(value, self.state)
        if metadata is not None and metadata.packed_weight is not None:
            return metadata
        with _disable_current_modes():
            fingerprint = _weight_fingerprint(value)
        metadata = self.state.weight_fingerprints.get(fingerprint)
        if metadata is not None:
            self.state.storage_meta[value.untyped_storage().data_ptr()] = metadata
            _attach_meta(value, metadata)
        return metadata

    def _attention(
        self, args: tuple[Any, ...], kwargs: dict[str, Any]
    ) -> Tensor | None:
        if len(args) < 3 or not all(isinstance(value, Tensor) for value in args[:3]):
            return None
        dropout_p = args[4] if len(args) > 4 else kwargs.get("dropout_p", 0.0)
        is_causal = args[5] if len(args) > 5 else kwargs.get("is_causal", False)
        if dropout_p != 0.0 or kwargs.get("enable_gqa", False):
            return None

        query = _dequantize(args[0], self.state)
        key = _dequantize(args[1], self.state)
        value = _dequantize(args[2], self.state)
        if query.ndim != 4 or key.shape != query.shape or value.shape != query.shape:
            return None
        attn_mask = args[3] if len(args) > 3 else kwargs.get("attn_mask")
        if isinstance(attn_mask, Tensor):
            attn_mask = _dequantize(attn_mask, self.state)
        scale = kwargs.get("scale")
        if scale is None:
            scale = query.shape[-1] ** -0.5

        batch, heads, sequence, head_dim = query.shape
        batch_heads = batch * heads
        query = query.reshape(batch_heads, sequence, head_dim)
        key = key.reshape(batch_heads, sequence, head_dim)
        value = value.reshape(batch_heads, sequence, head_dim)
        outputs = []
        for batch_head in range(batch_heads):
            query_matrix = query[batch_head]
            query_quantized = quantize_block32(
                query_matrix, calibrate_block32_scale(query_matrix)
            )
            key_weight = QuantizedWeight.from_float(key[batch_head])
            logits = fp8_gemm_fp32(query_quantized, key_weight) * scale
            if is_causal:
                causal_mask = torch.ones_like(logits, dtype=torch.bool).tril()
                logits = logits.masked_fill(~causal_mask, float("-inf"))
            if isinstance(attn_mask, Tensor):
                mask = attn_mask.reshape(-1, sequence, sequence)
                mask = mask[min(batch_head, mask.shape[0] - 1)]
                logits = (
                    logits.masked_fill(~mask, float("-inf"))
                    if mask.dtype == torch.bool
                    else logits + mask
                )
            probabilities = torch.softmax(logits, dim=-1)
            probability_quantized = quantize_block32(
                probabilities, calibrate_block32_scale(probabilities)
            )
            value_weight = QuantizedWeight.from_float(value[batch_head].transpose(0, 1))
            outputs.append(fp8_gemm_fp32(probability_quantized, value_weight))
        self.fp8_bmm_calls += batch_heads * 2
        output = torch.stack(outputs).reshape(batch, heads, sequence, head_dim)
        return output.to(torch.bfloat16)

    def _direct_linear(self, args: tuple[Any, ...]) -> Tensor | None:
        if len(args) < 2 or not isinstance(args[0], Tensor) or not isinstance(args[1], Tensor):
            return None
        weight_meta = self._resolve_weight_meta(args[1])
        if (
            weight_meta is None
            or weight_meta.kind != "linear_weight"
            or weight_meta.packed_weight is None
            or weight_meta.module_name not in self.state.calibration.linear
        ):
            return None
        activation = args[0]
        dense = _dequantize(activation, self.state)
        matrix = dense.reshape(-1, dense.shape[-1])
        calibration = self.state.calibration.linear[weight_meta.module_name]
        quantized = quantize_block32(
            matrix, calibration.input_scale_exponent.to(matrix.device)
        )
        bias = None
        if len(args) >= 3 and isinstance(args[2], Tensor):
            dense_bias = _dequantize(args[2], self.state).float()
            bias = F.pad(
                dense_bias,
                (0, weight_meta.packed_weight.out_features - dense_bias.numel()),
            )
        output = fp8_gemm(
            quantized,
            weight_meta.packed_weight,
            calibration.output_scale_exponent.to(matrix.device),
            bias=bias,
        )
        self.fp8_gemm_calls += 1
        return output.dequantize(dtype=torch.bfloat16).reshape(
            *dense.shape[:-1], weight_meta.packed_weight.logical_shape[0]
        )

    def _pointwise_convolution(
        self,
        func,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
        weight_meta: _TensorMeta,
    ) -> Tensor | None:
        if weight_meta.packed_weight is None or weight_meta.module_name is None:
            return None
        if weight_meta.module_name not in self.state.calibration.convolution:
            return None
        if func is torch.ops.aten.convolution.default:
            if len(args) < 9:
                return None
            stride, padding, dilation, transposed, _output_padding, groups = args[3:9]
            if transposed:
                return None
        elif func is torch.ops.aten.conv2d.default:
            stride = args[3] if len(args) > 3 else kwargs.get("stride", (1, 1))
            padding = args[4] if len(args) > 4 else kwargs.get("padding", (0, 0))
            dilation = args[5] if len(args) > 5 else kwargs.get("dilation", (1, 1))
            groups = args[6] if len(args) > 6 else kwargs.get("groups", 1)
        else:
            return None
        if groups != 1:
            return None
        if tuple(stride) != (1, 1) or tuple(padding) != (0, 0) or tuple(dilation) != (1, 1):
            return None
        value = args[0]
        if not isinstance(value, Tensor):
            return None
        dense = _dequantize(value, self.state)
        matrix = dense.permute(0, 2, 3, 1).reshape(-1, dense.shape[1])
        calibration = self.state.calibration.convolution[weight_meta.module_name]
        quantized = quantize_block32(
            matrix, calibration.input_scale_exponent.to(matrix.device)
        )
        bias = None
        if isinstance(args[2], Tensor):
            dense_bias = _dequantize(args[2], self.state).float()
            bias = F.pad(
                dense_bias,
                (0, weight_meta.packed_weight.out_features - dense_bias.numel()),
            )
        output = fp8_gemm(
            quantized,
            weight_meta.packed_weight,
            calibration.output_scale_exponent.to(matrix.device),
            bias=bias,
        )
        self.fp8_pointwise_conv_calls += 1
        return output.dequantize(dtype=torch.bfloat16).reshape(
            dense.shape[0], dense.shape[2], dense.shape[3], -1
        ).permute(0, 3, 1, 2)
    def _alias(self, func, args: tuple[Any, ...], kwargs: dict[str, Any]):
        value = args[0]
        if not isinstance(value, Tensor):
            return None
        metadata = _meta_for(value, self.state)
        if metadata is None:
            return None
        with _disable_current_modes():
            result = func(*args, **kwargs)
        return _attach_meta(
            result,
            _TensorMeta(
                "view",
                metadata.scale_exponent,
                tuple(result.shape),
                source_values=value,
                source_meta=metadata,
                view_func=func,
                view_args=tuple(args[1:]),
                view_kwargs=dict(kwargs),
            ),
        )


    def __torch_dispatch__(
        self,
        func,
        types,
        args: tuple[Any, ...] = (),
        kwargs: dict[str, Any] | None = None,
    ):
        kwargs = kwargs or {}
        if func is torch.ops.aten.native_layer_norm.default and len(args) >= 5:
            value, normalized_shape, weight, bias, epsilon = args[:5]
            if isinstance(value, Tensor) and isinstance(weight, Tensor):
                value = _dequantize(value, self.state).to(weight.dtype)
                bias = (
                    _dequantize(bias, self.state).to(weight.dtype)
                    if isinstance(bias, Tensor)
                    else bias
                )
                with _disable_current_modes():
                    return func(value, normalized_shape, weight, bias, epsilon, *args[5:])
        if func is torch.ops.aten.layer_norm.default and len(args) >= 4:
            value, normalized_shape, weight, bias = args[:4]
            if isinstance(value, Tensor) and isinstance(weight, Tensor):
                normalized_args = list(args)
                normalized_args[0] = _dequantize(value, self.state).to(weight.dtype)
                normalized_args[2] = _dequantize(weight, self.state).to(weight.dtype)
                normalized_args[3] = (
                    _dequantize(bias, self.state).to(weight.dtype)
                    if isinstance(bias, Tensor)
                    else bias
                )
                with _disable_current_modes():
                    return func(*normalized_args, **kwargs)

        if func is torch.ops.aten._to_copy.default and args and isinstance(args[0], Tensor):
            metadata = _meta_for(args[0], self.state)
            requested_dtype = kwargs.get("dtype")
            if (
                metadata is not None
                and requested_dtype is not None
                and requested_dtype.is_floating_point
            ):
                with _disable_current_modes():
                    result = args[0].clone()
                return _attach_meta(result, metadata)

        if func is torch.ops.aten.t.default and args and isinstance(args[0], Tensor):
            metadata = _meta_for(args[0], self.state)
            if metadata is not None and metadata.kind == "linear_weight":
                with _disable_current_modes():
                    result = func(*args, **kwargs)
                setattr(result, _TRANSPOSED_WEIGHT_ATTR, metadata)
                return result

        if func in _ALIAS_OPS and args:
            result = self._alias(func, args, kwargs)
            if result is not None:
                return result

        if func is torch.ops.aten.linear.default:
            result = self._direct_linear(args)
            if result is not None:
                return result

        if func is torch.ops.aten.scaled_dot_product_attention.default:
            result = self._attention(args, kwargs)
            if result is not None:
                return result

        if func is torch.ops.aten.addmm.default:
            if len(args) >= 3:
                result = self._linear((args[1], args[2], args[0]), kwargs)
                if result is not None:
                    return result
        elif func is torch.ops.aten.mm.default:
            result = self._linear(args, kwargs)
            if result is not None:
                return result
        elif func is torch.ops.aten.bmm.default and len(args) >= 2:
            left, right = args[:2]
            if (
                isinstance(left, Tensor)
                and isinstance(right, Tensor)
                and left.ndim == right.ndim == 3
                and left.shape[0] == right.shape[0]
                and left.shape[-1] == right.shape[-2]
            ):
                return self._bmm(left, right)
        elif func in _CONVOLUTION_OPS and len(args) >= 2:
            weight = args[1]
            if isinstance(weight, Tensor):
                weight_meta = self._resolve_weight_meta(weight)
                if weight_meta is not None and weight_meta.kind == "linear_weight":
                    result = self._pointwise_convolution(func, args, kwargs, weight_meta)
                    if result is not None:
                        return result

        tainted = _contains_quantized(args, self.state) or _contains_quantized(
            kwargs, self.state
        )
        if not tainted:
            with _disable_current_modes():
                return func(*args, **kwargs)
        def dense(item: Any) -> Any:
            if not isinstance(item, Tensor):
                return item
            metadata = _meta_for(item, self.state)
            if metadata is not None:
                return _dequantize(item, self.state)
            if item.is_floating_point():
                return item.float()
            return item

        dense_args = _tree_map(args, dense)
        dense_kwargs = _tree_map(kwargs, dense)
        if func in _CONVOLUTION_OPS:
            dense_args = list(dense_args)
            dense_args[0] = dense_args[0].float()
            dense_args[1] = dense_args[1].float()
            if isinstance(dense_args[2], Tensor):
                dense_args[2] = dense_args[2].float()
            dense_args = tuple(dense_args)
        with torch.autocast(device_type="cuda", enabled=False), _disable_current_modes():
            result = func(*dense_args, **dense_kwargs)
        self.fallback_calls += 1
        if func in _CONVOLUTION_OPS and isinstance(args[0], Tensor):
            return result.to(args[0].dtype)
        return result.to(torch.bfloat16)



class FullFP8Model(nn.Module):
    """Inference wrapper for the literal E4M3 activation/weight execution path."""

    def __init__(self, model: nn.Module, calibration: FullFP8Calibration) -> None:
        super().__init__()
        if not torch.cuda.is_available() or torch.cuda.get_device_capability() < (12, 0):
            raise RuntimeError("Literal full-FP8 execution requires an SM120-or-newer CUDA GPU")
        self.model = model.eval()
        for module in self.model.modules():
            if isinstance(module, nn.ReLU):
                module.inplace = False
        self._input_dtype = next(
            (
                module.bias.dtype
                for module in self.model.modules()
                if isinstance(module, (nn.Conv2d, nn.Linear))
                and module.bias is not None
            ),
            next(
                (parameter.dtype for parameter in self.model.parameters()),
                torch.float32,
            ),
        )
        with torch.no_grad():
            for module in self.model.modules():
                if (
                    isinstance(module, (nn.Conv2d, nn.ConvTranspose2d))
                    and module.bias is not None
                    and module.bias.dtype != self._input_dtype
                ):
                    module.bias.data = module.bias.data.to(self._input_dtype)
        self._state = _pack_model(self.model, calibration)
        self.last_dispatch_stats: dict[str, int] = {}

    @property
    def packing_report(self) -> FullFP8PackingReport:
        return self._state.report

    def _execute(self, function: Callable[..., Any], *args, **kwargs):
        mode = FullFP8DispatchMode(self._state)
        attention_context = (
            sdpa_kernel(SDPBackend.MATH) if torch.cuda.is_available() else nullcontext()
        )
        def quantize_input(item: Any) -> Any:
            if isinstance(item, Tensor) and item.is_floating_point():
                return item.to(self._input_dtype)
            return item

        args = _tree_map(args, quantize_input)
        kwargs = _tree_map(kwargs, quantize_input)
        autocast_context = _without_nested_autocast()
        with autocast_context, attention_context, mode:
            result = function(*args, **kwargs)
        self.last_dispatch_stats = {
            "fp8_gemm_calls": mode.fp8_gemm_calls,
            "fp8_bmm_calls": mode.fp8_bmm_calls,
            "fp8_pointwise_conv_calls": mode.fp8_pointwise_conv_calls,
            "fallback_calls": mode.fallback_calls,
        }
        return _tree_map(
            result,
            lambda item: _dequantize(item, self._state)
            if isinstance(item, Tensor) and _meta_for(item, self._state) is not None
            else item,
        )
    def forward(self, *args, **kwargs):
        return self._execute(self.model, *args, **kwargs)

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
            "confidence_BA": confidence_BA.clone() if confidence_BA is not None else None,
            "overlap_BA": overlap_BA.clone() if overlap_BA is not None else None,
            "precision_BA": precision_BA.clone() if precision_BA is not None else None,
        }

    def sample(self, *args, **kwargs):
        return self.model.sample(*args, **kwargs)

    def to_pixel_coordinates(self, *args, **kwargs):
        return self.model.to_pixel_coordinates(*args, **kwargs)


def calibrate_full_fp8(
    model: nn.Module,
    samples: Iterable[tuple[tuple[Any, ...], dict[str, Any]]],
    *,
    call: Callable[..., Any] | None = None,
) -> FullFP8Calibration:
    """Run representative samples and return complete module-boundary scales."""
    function = model if call is None else call
    with torch.inference_mode(), FullFP8Calibrator(model) as calibrator:
        for args, kwargs in samples:
            function(*args, **kwargs)
    return calibrator.result()
