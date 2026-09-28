"""Quantized deformable convolution: INT8 GEMM over the gathered receptive-field patches, calibrated
per module. See ``flashim/ops/aliked.py`` for why the offset/gather stays FP32."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Self

import torch
from torch import Tensor, nn

from flashim import ops
from flashim.models.aliked.blocks import DeformableConv2d
from flashim.ops.aliked import gather_deform_patches
from flashim.quant.calibration import Int8CalibrationTable
from flashim.quant.methods.int8_linear import StaticInt8Linear


class QuantizedDeformableConv2d(nn.Module):
    """Drop-in replacement for :class:`DeformableConv2d`. ``offset_conv`` is unchanged; ``regular_conv`` +
    the patch gather become a :class:`StaticInt8Linear` GEMM over ``[B * oH * oW, C * K * K]``."""

    def __init__(self, offset_conv: nn.Conv2d, gemm: StaticInt8Linear, kernel_size: int, padding: int) -> None:
        super().__init__()
        self.offset_conv = offset_conv
        self.gemm = gemm
        self.kernel_size = kernel_size
        self.padding = padding

    @classmethod
    def from_deform_conv(cls, deform: DeformableConv2d, input_scale: Tensor) -> QuantizedDeformableConv2d:
        regular = deform.regular_conv
        out_channels, in_channels, k, _ = regular.weight.shape
        linear = nn.Linear(in_channels * k * k, out_channels, bias=regular.bias is not None)
        linear.weight = nn.Parameter(regular.weight.detach().reshape(out_channels, in_channels * k * k), requires_grad=False)
        if regular.bias is not None:
            linear.bias = nn.Parameter(regular.bias.detach(), requires_grad=False)
        gemm = StaticInt8Linear.from_linear(linear, input_scale)
        return cls(deform.offset_conv, gemm, kernel_size=k, padding=deform.padding)

    def forward(self, x: Tensor) -> Tensor:
        dtype = x.dtype
        max_offset = max(x.shape[-2:]) / 4.0
        offset = self.offset_conv(x).clamp(-max_offset, max_offset)
        quantized = ops.deform_patches_int8(x.float(), offset.float(), self.kernel_size, self.padding, self.gemm.input_scale)
        out = self.gemm.forward_prequantized(quantized)  # [B, oH, oW, out_channels], BF16
        return out.permute(0, 3, 1, 2).contiguous().to(dtype)

    def extra_repr(self) -> str:
        return f"in={self.gemm.in_features}, out={self.gemm.out_features}, k={self.kernel_size}, int8"


class DeformConvCalibrator:
    """Percentile calibration of each targeted :class:`DeformableConv2d`'s gathered-patches activation —
    the actual input to the quantized GEMM, not the block's raw input — matching
    :class:`~flashim.quant.calibration.Int8Calibrator`'s methodology (mean of per-call percentile / 127).
    """

    def __init__(self, model: nn.Module, names: Iterable[str], *, percentile: float = 0.9999, max_samples: int = 4_000_000) -> None:
        modules = dict(model.named_modules())
        self._targets: dict[str, DeformableConv2d] = {}
        for name in names:
            module = modules.get(name)
            if not isinstance(module, DeformableConv2d):
                raise TypeError(f"{name!r} is not a DeformableConv2d")
            self._targets[name] = module
        self.percentile, self.max_samples = percentile, max_samples
        self._calls: dict[str, list[float]] = {name: [] for name in self._targets}
        self._handles: list = []

    def __enter__(self) -> Self:
        for name, module in self._targets.items():

            def hook(mod: DeformableConv2d, args: tuple[Tensor, ...], name: str = name) -> None:
                x = args[0]
                max_offset = max(x.shape[-2:]) / 4.0
                offset = mod.offset_conv(x).clamp(-max_offset, max_offset)
                k = mod.regular_conv.weight.shape[-1]
                patches = gather_deform_patches(x.float(), offset.float(), k, mod.padding)
                flat = patches.reshape(-1).abs()
                if flat.numel() > self.max_samples:
                    flat = flat[torch.randperm(flat.numel(), device=flat.device)[: self.max_samples]]
                percentile_idx = max(1, int(flat.numel() * self.percentile))
                self._calls[name].append(float(flat.kthvalue(percentile_idx).values))

            self._handles.append(module.register_forward_pre_hook(hook))
        return self

    def __exit__(self, *exc: object) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()

    def table(self, metadata: dict[str, str] | None = None) -> Int8CalibrationTable:
        missing = sorted(name for name, calls in self._calls.items() if not calls)
        if missing:
            raise RuntimeError(f"No activations recorded for: {missing}")
        scales = {name: torch.tensor(sum(calls) / len(calls) / 127.0) for name, calls in self._calls.items()}
        meta = {"observer": "percentile", "percentile": str(self.percentile), **(metadata or {})}
        return Int8CalibrationTable(scales, meta)


def quantize_deformable_convs(model: nn.Module, calibration: Int8CalibrationTable) -> list[str]:
    """Replace every calibrated :class:`DeformableConv2d` (matched by module name) with a
    :class:`QuantizedDeformableConv2d`. Returns the replaced module paths."""
    from flashim.utils import set_submodule

    replaced = []
    for name, module in list(model.named_modules()):
        if isinstance(module, DeformableConv2d) and name in calibration:
            set_submodule(model, name, QuantizedDeformableConv2d.from_deform_conv(module, calibration[name]))
            replaced.append(name)
    return replaced
