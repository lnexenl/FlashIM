"""FlexGlue's spec: the sparse-to-dense matcher with its fp32 / bf16 / int8 / fp8 presets."""

from __future__ import annotations

import dataclasses

import torch

from flashim.build import BuildContext, EngineSpec, ModelSpec, Pass, dtype
from flashim.config import RunConfig
from flashim.models.flexglue import quant
from flashim.models.flexglue.config import FlexGlueConfig
from flashim.models.flexglue.descriptor import ConvNeXtBlock
from flashim.models.flexglue.model import FlexGlue
from flashim.platforms import Feature
from flashim.quant.methods.int8_linear import StaticInt8Linear
from flashim.quant.methods.linear import StaticMXFP8Linear, enable_fused_bias


def _load(config: RunConfig) -> FlexGlue:
    arch = dataclasses.replace(FlexGlueConfig(), **config.arch)
    arch.validate()
    return FlexGlue.from_pretrained(arch, checkpoint=config.checkpoint, device=config.device)


def _set_precision(model: FlexGlue, config: RunConfig) -> None:
    if config.precision.compute_dtype != "fp32":
        model.set_compute_dtype(dtype(config.precision.compute_dtype))


def _set_attention(model: FlexGlue, impl: str) -> None:
    for layer in model.transformers:
        layer.self_attn.sdpa_impl = layer.cross_attn.sdpa_impl = impl


def _fold_convnext_gamma(model: FlexGlue, ctx: BuildContext) -> int | str:
    """Fold ConvNeXt ``gamma`` into ``pwconv2`` before quantization (skipped in fp32 for exact reference parity)."""
    if ctx.config.precision.compute_dtype == "fp32":
        return "skipped: fp32"
    return sum(m.fold_gamma() for m in model.descriptor.backbone.modules() if isinstance(m, ConvNeXtBlock))


def _channels_last(model: FlexGlue, ctx: BuildContext) -> str:
    """NHWC ConvNeXt backbone + LinearConvHead in bf16 / int8 / fp8: eliminates 72 NCHW<->NHWC copies per pair and
    runs 7x7 depthwise convs via cuDNN NHWC."""
    if ctx.config.precision.compute_dtype == "fp32":
        return "skipped: fp32"
    model.descriptor.backbone.to(memory_format=torch.channels_last)
    model.descriptor.head.to(memory_format=torch.channels_last)
    model.descriptor.channels_last = True
    return "descriptor"


def _fuse_convnext_int8(model: FlexGlue, ctx: BuildContext) -> int | str:
    """Fuse ConvNeXtBlock LN + quantize, GELU + quantize, and pwconv2 residual add when pwconv1/pwconv2 are static
    INT8 (int8 and fp8 presets)."""
    count = 0
    for m in model.descriptor.backbone.modules():
        if (
            isinstance(m, ConvNeXtBlock)
            and m.gamma is None
            and isinstance(m.pwconv1, StaticInt8Linear)
            and isinstance(m.pwconv2, StaticInt8Linear)
        ):
            m.fused_int8 = True
            count += 1
    return count if count else "skipped: no static INT8 ConvNeXt blocks"


def _fuse_transformer_glue(model: FlexGlue, ctx: BuildContext) -> int | str:
    """fp8 / int8: fuse quantized transformer glue (MXFP8 cuBLAS bias epilogue, attention-output head quantize,
    message_norm+cat+ffn.0 quantize, ffn LN+GELU+quantize, INT8 residual epilogue, shared to_qk/to_v quantize)."""
    qkind = ctx.config.precision.quant
    if qkind not in ("fp8", "int8"):
        return f"skipped: quant={qkind!r}"
    if qkind == "fp8":
        enable_fused_bias(model)
    count = 0
    for layer in model.transformers:
        sa, ca = layer.self_attn, layer.cross_attn
        sa.fused_glue = True
        ca.fused_glue = True
        if isinstance(ca.to_qk, StaticMXFP8Linear) and isinstance(ca.to_v, StaticMXFP8Linear):
            ca.shared_qkv_scale = bool(torch.equal(ca.to_qk.input_exponent, ca.to_v.input_exponent))
        elif isinstance(ca.to_qk, StaticInt8Linear) and isinstance(ca.to_v, StaticInt8Linear):
            ca.shared_qkv_scale = bool(torch.equal(ca.to_qk.input_scale, ca.to_v.input_scale))
        count += 1
    return count


def _cuda_graph(model: FlexGlue, ctx: BuildContext) -> str:
    """Eager engine: replay the fixed-shape DINOv3-ConvNeXt descriptor from a CUDA graph (bit-identical; removes
    ~1.5 ms of ConvNeXt Python/Triton launch overhead on fp8/int8). Transformer layers are not graphed because
    SageAttention launches on the legacy default stream and corrupts CUDA graph replay."""
    if ctx.config.engine.kind != "eager":
        return f"skipped: engine {ctx.config.engine.kind}"
    if not ctx.platform.supports(Feature.CUDA_GRAPHS):
        return f"skipped: no CUDA graphs on {ctx.platform.describe()}"
    model.descriptor_graph = True
    return "descriptor"


def _sage_attention(model: FlexGlue, ctx: BuildContext) -> str:
    """int8 / fp8: SageAttention in every transformer layer, the impl :data:`quant.SAGE_ATTENTION` gives the
    platform (accuracy and timings there). Torch SDPA elsewhere; a pinned ``optimize.attention`` wins."""
    from flashim import ops
    from flashim.log import get_logger

    if ctx.config.optimize.attention != "sdpa":
        return f"skipped: optimize.attention={ctx.config.optimize.attention!r}"
    impl = quant.SAGE_ATTENTION.get(ctx.config.precision.quant or "", {}).get(ctx.platform.capability)
    if impl is None:
        return f"skipped: torch SDPA on {ctx.platform.describe()}"
    if not any(i.name == impl for i in ops.sdpa.impls):
        get_logger("build").warning("flexglue: %r needs the sageattention package; attention runs torch SDPA", impl)
        return f"skipped: {impl!r} not registered (sageattention not installed)"
    _set_attention(model, impl)
    return impl


def _validate(config: RunConfig) -> None:
    if config.optimize.attention != "sdpa" and config.precision.compute_dtype == "fp32":
        raise ValueError(f"attention={config.optimize.attention!r} needs a bf16 / int8 / fp8 preset (BF16 attention "
                         "inputs); in fp32 it would silently run torch SDPA")


def _regions(model: FlexGlue) -> list[str]:
    return [
        "descriptor",
        "input_proj",
        *(f"transformers.{i}" for i in range(len(model.transformers))),
        *(f"heads.{i}" for i in range(len(model.heads))),
    ]


_QUANTIZED = {name: p for name, p in quant.PRESETS.items() if p.config is not None}

SPEC = ModelSpec(
    name="flexglue",
    role="matcher",
    description="FlexGlue sparse-to-dense matcher (DINOv3-ConvNeXt descriptor, 9 transformer layers, per-layer "
    "correspondence heads with a 1.5-entmax readout; matches keypoints or samples the dense warp)",
    load=_load,
    presets={
        "fp32": {},
        "bf16": {
            "precision": {"compute_dtype": "bf16"},
            "optimize": {"passes": ["fold_convnext_gamma", "channels_last", "cuda_graph"]},
        },
        **{
            name: {
                "extends": "bf16",
                "precision": {"quant": name},
                "optimize": {
                    "passes": [
                        "fold_convnext_gamma",
                        "channels_last",
                        "fuse_convnext_int8",
                        "fuse_transformer_glue",
                        "sage_attention",
                        "cuda_graph",
                    ]
                },
            }
            for name in _QUANTIZED
        },
    },
    default_preset="fp8",
    quant_configs={name: p.config for name, p in _QUANTIZED.items()},
    calibrations={name: p.calibration for name, p in _QUANTIZED.items()},
    protected=quant.PROTECTED,
    passes={
        "fold_convnext_gamma": Pass(
            _fold_convnext_gamma, stage="pre_quant", exact=False,
            description="fold ConvNeXtBlock.gamma into pwconv2 before quantization",
        ),
        "channels_last": Pass(
            _channels_last, exact=False,
            description="NHWC ConvNeXt backbone + LinearConvHead (bf16 / int8 / fp8)",
        ),
        "fuse_convnext_int8": Pass(
            _fuse_convnext_int8, exact=False,
            description="fuse ConvNeXtBlock LN+quantize_int8, GELU+quantize_int8 and pwconv2 residual add",
        ),
        "fuse_transformer_glue": Pass(
            _fuse_transformer_glue, exact=False,
            description="fuse MXFP8/INT8 transformer bias, head quantize, message_norm+cat, FFN LN+GELU and residual",
        ),
        "sage_attention": Pass(
            _sage_attention, exact=False,
            description="int8 / fp8: SageAttention (INT8 QK; FP16 / FP8 PV) where measured faster than torch SDPA",
        ),
        "cuda_graph": Pass(
            _cuda_graph, exact=True,
            description="eager: replay fixed-shape descriptor from a CUDA graph (bit-identical)",
        ),
    },
    engines={"eager": EngineSpec(), "compile": EngineSpec(_regions, recompile_limit=64)},
    set_precision=_set_precision,
    set_attention=_set_attention,
    validate=_validate,
)
