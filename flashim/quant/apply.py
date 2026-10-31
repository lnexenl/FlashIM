"""Apply a :class:`QuantConfig` to a model: pack linears, then run fusion passes."""

from __future__ import annotations

from collections.abc import Iterable

from torch import nn

from flashim.quant.calibration import CalibrationTable, Int8CalibrationTable
from flashim.quant.methods.conv2d import StaticInt8Conv2d, StaticMXFP8Conv2d
from flashim.quant.methods.int8_linear import StaticInt8Conv1x1, StaticInt8Linear
from flashim.quant.methods.linear import StaticMXFP8Linear
from flashim.quant.passes.deformable_encoder import QuantizedDeformableEncoder, is_deformable_encoder
from flashim.quant.passes.separable_conv_block import (
    Int8SeparableConvBlock,
    MXFP8SeparableConvBlock,
    is_separable_conv_block,
)
from flashim.quant.passes.transformer_block import QuantizedTransformerBlock, is_transformer_block
from flashim.quant.scheme import PackingReport, QuantConfig
from flashim.utils import nbytes, set_submodule


def quantize_model(
    model: nn.Module,
    config: QuantConfig,
    calibration: CalibrationTable | Int8CalibrationTable | None,
    *,
    protected: Iterable[str] = (),
) -> PackingReport:
    """In place. Linears targeted by a static scheme become :class:`StaticMXFP8Linear`
    (or :class:`StaticInt8Linear` for INT8 schemes, which need an :class:`Int8CalibrationTable`);
    transformer blocks whose attention projections were packed get fused boundaries;
    separable conv blocks whose ``conv_pointwise`` is targeted by a dynamic MXFP8 scheme become
    :class:`MXFP8SeparableConvBlock`, by a static INT8 scheme :class:`Int8SeparableConvBlock`
    (BatchNorm must already be folded). Other 1x1 convolutions targeted by an INT8 scheme become
    :class:`StaticInt8Conv1x1`. Kernels are chosen by op dispatch on the layers' device.
    """
    names = [name for name, _ in model.named_modules()]
    config.check_protected(names, protected)
    report = PackingReport()
    modules = dict(model.named_modules())

    # Separable blocks that are fused as a whole (step 3); a block with ``fuse_quantized`` off keeps its pointwise conv
    # in step 1, like any other 1x1.
    fused_blocks = {n for n, m in modules.items() if is_separable_conv_block(m) and m.fuse_quantized}
    block_pointwise = {f"{n}.{modules[n].pointwise_name}" for n in fused_blocks}

    # 1) Static-activation linears (and, for INT8, 1x1 convolutions outside separable blocks).
    for name, module in modules.items():
        match = config.scheme_for(name)
        if match is None:
            continue
        _, scheme = match
        is_conv1x1 = (
            isinstance(module, nn.Conv2d)
            and module.kernel_size == (1, 1)
            and module.groups == 1
            and module.stride == (1, 1)  # strided 1x1s take the generic im2col path below
            and module.padding == (0, 0)
            and name not in block_pointwise
        )
        is_conv = isinstance(module, nn.Conv2d) and module.groups == 1 and name not in block_pointwise
        if not (isinstance(module, nn.Linear) or is_conv1x1 and scheme.weights.type == "int" or is_conv):
            continue
        if scheme.input_activations is None or scheme.input_activations.dynamic:
            continue
        if calibration is None or name not in calibration:
            raise RuntimeError(f"No calibration for static-activation linear {name!r}")
        if scheme.weights.type == "int":
            if not isinstance(calibration, Int8CalibrationTable):
                raise TypeError(f"INT8 scheme for {name!r} needs an Int8CalibrationTable")
            if is_conv1x1:
                build = StaticInt8Conv1x1.from_conv
            elif is_conv:
                build = StaticInt8Conv2d.from_conv
            else:
                build = StaticInt8Linear.from_linear
            if name in calibration.smooth:
                if not isinstance(module, nn.Linear):
                    raise ValueError(f"SmoothQuant factors for {name!r}: only linears take them")
                packed = build(module, calibration[name], smooth=calibration.smooth[name])
            else:
                packed = build(module, calibration[name])
        else:
            if not isinstance(calibration, CalibrationTable):
                raise TypeError(f"MXFP8 scheme for {name!r} needs a CalibrationTable")
            build = StaticMXFP8Conv2d.from_conv if is_conv else StaticMXFP8Linear.from_linear
            packed = build(module, calibration[name].input_exponent)
        report.bytes_before += nbytes(module)
        report.bytes_after += nbytes(packed)
        set_submodule(model, name, packed)
        report.linears[name] = "int8_linear" if isinstance(packed, StaticInt8Linear) else packed.kernel_name
        if is_conv:
            report.convs.append(name)

    # 2) Transformer-block boundary fusion (needs packed QKV).
    for name, module in list(model.named_modules()):
        if is_transformer_block(module) and isinstance(module.attn.qkv, (StaticMXFP8Linear, StaticInt8Linear)):
            set_submodule(model, name, QuantizedTransformerBlock(module))
            report.fused_blocks.append(name)

    # 2b) Post-norm deformable-attention encoders with packed value/output/FFN linears.
    for name, module in list(model.named_modules()):
        if is_deformable_encoder(module):
            set_submodule(model, name, QuantizedDeformableEncoder(module))
            report.fused_blocks.append(name)

    # 3) Separable conv blocks with a dynamically quantized pointwise conv.
    for name, module in list(model.named_modules()):
        if name not in fused_blocks:
            continue
        match = config.scheme_for(f"{name}.{module.pointwise_name}")
        if match is None:
            continue
        scheme = match[1]
        if scheme.weights.type == "int":
            pointwise = f"{name}.{module.pointwise_name}"
            if scheme.input_activations is None or scheme.input_activations.dynamic:
                raise ValueError(f"INT8 pointwise conv {pointwise!r} requires a static-activation scheme")
            if not isinstance(calibration, Int8CalibrationTable) or pointwise not in calibration:
                raise RuntimeError(f"No INT8 calibration for {pointwise!r}")
            fused = Int8SeparableConvBlock(module, calibration[pointwise]).to(next(module.parameters()).device)
        elif scheme.input_activations is None or not scheme.input_activations.dynamic:
            raise ValueError(f"Pointwise conv {name!r} requires a dynamic-activation scheme")
        else:
            fused = MXFP8SeparableConvBlock(module).to(next(module.parameters()).device)
        report.bytes_before += nbytes(module)
        report.bytes_after += nbytes(fused)
        set_submodule(model, name, fused)
        report.fused_conv_blocks.append(name)
    return report

