"""FlexGlue's spec: the sparse-to-dense matcher with its fp32 / bf16 / int8 / fp8 presets."""

from __future__ import annotations

import dataclasses

from flashim.build import BuildContext, EngineSpec, ModelSpec, Pass, dtype
from flashim.config import RunConfig
from flashim.models.flexglue import quant
from flashim.models.flexglue.config import FlexGlueConfig
from flashim.models.flexglue.model import FlexGlue


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


def _hopper(platform) -> bool:
    return platform.device_type == "cuda" and platform.capability == (9, 0)


def _block128(model: FlexGlue, ctx: BuildContext) -> str:
    """fp8 on Hopper: every transformer MXFP8 linear whose shape tiles takes 1x128 activation / 128x128 weight scales
    (cuBLAS blockwise FP8, ``cublas-block128``): the input exponents become the max over groups of 4 blocks. Only on
    132-SM parts (H800 / H100 SXM): the GEMM time was measured there alone."""
    import torch

    from flashim.quant.methods.linear import StaticMXFP8Linear

    layers = [(n, m) for n, m in model.named_modules() if type(m) is StaticMXFP8Linear]
    if not layers:
        return "skipped: no MXFP8 linears"
    sms = torch.cuda.get_device_properties(layers[0][1].input_exponent.device).multi_processor_count
    if sms != 132:
        return f"skipped: {sms} SMs (measured on 132)"
    done = 0
    for _, layer in layers:
        if layer.weight_values.shape[0] % 128 == 0 and layer.weight_values.shape[1] % 128 == 0:
            layer.set_activation_block(128)
            done += 1
    return f"{done}/{len(layers)} linears"


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
        "bf16": {"precision": {"compute_dtype": "bf16"}},
        **{
            name: {"extends": "bf16", "precision": {"quant": name}, "optimize": {"passes": ["sage_attention"]}}
            for name in _QUANTIZED
        },
    },
    default_preset="fp8",
    quant_configs={name: p.config for name, p in _QUANTIZED.items()},
    calibrations={name: p.calibration for name, p in _QUANTIZED.items()},
    protected=quant.PROTECTED,
    passes={
        "sage_attention": Pass(
            _sage_attention, exact=False,
            description="int8 / fp8: SageAttention (INT8 QK; FP16 / FP8 PV) where measured faster than torch SDPA",
        ),
        "block128": Pass(
            _block128, requires=_hopper, exact=False,
            description="fp8 on Hopper: cuBLAS blockwise FP8 GEMMs (1x128 activation scales, 128x128 weight blocks)",
        ),
    },
    engines={"eager": EngineSpec(), "compile": EngineSpec(_regions, recompile_limit=64)},
    set_precision=_set_precision,
    set_attention=_set_attention,
    validate=_validate,
)
