"""Apply a :class:`QuantConfig` to a model: pack linears, then run fusion passes."""

from __future__ import annotations

from collections.abc import Iterable

from torch import nn

from flashim.platforms import Platform, platform_for
from flashim.quant.calibration import CalibrationTable
from flashim.quant.methods.linear import StaticMXFP8Linear
from flashim.quant.passes.refiner_block import FusedMXFP8RefinerBlock, is_refiner_block
from flashim.quant.passes.transformer_block import FP8TransformerBlock, is_transformer_block
from flashim.quant.scheme import PackingReport, QuantConfig


def _set_submodule(model: nn.Module, name: str, module: nn.Module) -> None:
    parent, _, child = name.rpartition(".")
    (model.get_submodule(parent) if parent else model)._modules[child] = module


def _nbytes(module: nn.Module) -> int:
    return sum(t.numel() * t.element_size() for t in [*module.parameters(), *module.buffers()])


def quantize_model(
    model: nn.Module,
    config: QuantConfig,
    calibration: CalibrationTable | None,
    *,
    protected: Iterable[str] = (),
    platform: Platform | None = None,
) -> PackingReport:
    """In place. Linears targeted by a static scheme become :class:`StaticMXFP8Linear`;
    transformer blocks whose attention projections were packed get fused boundaries;
    refiner blocks whose ``conv_pointwise`` is targeted by a dynamic scheme become
    :class:`FusedMXFP8RefinerBlock` (BatchNorm must already be folded).
    """
    platform = platform or platform_for(next(model.parameters()).device)
    names = [name for name, _ in model.named_modules()]
    config.check_protected(names, protected)
    report = PackingReport()
    modules = dict(model.named_modules())

    # 1) Static-activation linears.
    for name, module in modules.items():
        match = config.scheme_for(name)
        if match is None or not isinstance(module, nn.Linear):
            continue
        _, scheme = match
        if scheme.input_activations is None or scheme.input_activations.dynamic:
            continue
        if calibration is None or name not in calibration:
            raise RuntimeError(f"No calibration for static-activation linear {name!r}")
        packed = StaticMXFP8Linear.from_linear(module, calibration[name].input_exponent, platform)
        report.bytes_before += _nbytes(module)
        report.bytes_after += _nbytes(packed)
        _set_submodule(model, name, packed)
        report.linears[name] = packed.kernel.name

    # 2) Transformer-block boundary fusion (needs packed QKV).
    for name, module in list(model.named_modules()):
        if is_transformer_block(module) and isinstance(module.attn.qkv, StaticMXFP8Linear):
            _set_submodule(model, name, FP8TransformerBlock(module))
            report.fused_blocks.append(name)

    # 3) Refiner blocks with a dynamically quantized pointwise conv.
    for name, module in list(model.named_modules()):
        if not is_refiner_block(module):
            continue
        match = config.scheme_for(f"{name}.conv_pointwise")
        if match is None:
            continue
        scheme = match[1]
        if scheme.input_activations is None or not scheme.input_activations.dynamic:
            raise ValueError(f"Refiner pointwise {name!r} requires a dynamic-activation scheme")
        fused = FusedMXFP8RefinerBlock(module).to(next(module.parameters()).device)
        report.bytes_before += _nbytes(module)
        report.bytes_after += _nbytes(fused)
        _set_submodule(model, name, fused)
        report.fused_refiner_blocks.append(name)
    return report


def reset_call_counts(model: nn.Module) -> None:
    for module in model.modules():
        if hasattr(module, "call_count"):
            module.call_count = 0


def dispatch_stats(model: nn.Module) -> dict[str, int]:
    stats = {"fp8_linear_calls": 0, "fp8_transformer_block_calls": 0, "fused_refiner_block_calls": 0}
    for module in model.modules():
        if isinstance(module, StaticMXFP8Linear):
            stats["fp8_linear_calls"] += module.call_count
        elif isinstance(module, FP8TransformerBlock):
            stats["fp8_transformer_block_calls"] += module.call_count
        elif isinstance(module, FusedMXFP8RefinerBlock):
            stats["fused_refiner_block_calls"] += module.call_count
    return stats
