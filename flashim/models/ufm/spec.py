"""UFM's :class:`~flashim.build.ModelSpec` (presets ``fp32`` / ``bf16`` / ``int8`` / ``fp8``)."""

from __future__ import annotations

import importlib.util

import torch
from torch import nn

from flashim.build import BuildContext, EngineSpec, ModelSpec, Pass
from flashim.build import dtype as resolve_dtype
from flashim.config import RunConfig
from flashim.models.ufm import quant
from flashim.models.ufm.config import UFMConfig
from flashim.models.ufm.model import UFM, DPTRegressionProcessor, FeatureFusionBlock
from flashim.platforms import Feature
from flashim.quant.methods.conv2d import StaticInt8Conv2d

COMPILE_REGIONS = ("encoder", "info_sharing")
COMPILE_FULL_REGIONS = ("encoder", "info_sharing", "head1.0", "uncertainty_head.0")

# SageAttention impl per architecture (QK INT8 per-warp, PV FP8): measured -2.2 ms on sm_120 with better flow EPE.
_SAGE_IMPLS = {(8, 9): "sage-fp8-warp", (9, 0): "sage-fp8-sm90-warp", (12, 0): "sage-fp8-warp"}


def _load(config: RunConfig) -> UFM:
    return UFM.from_pretrained(
        UFMConfig(**config.arch), checkpoint=config.checkpoint, device=config.device
    )


def _set_precision(model: UFM, config: RunConfig) -> None:
    compute = resolve_dtype(config.precision.compute_dtype)
    default_heads = "fp32" if config.precision.compute_dtype == "fp32" else "bf16"
    heads = resolve_dtype(config.optimize.options.get("heads_dtype", default_heads))
    model.set_compute_dtype(compute, heads_dtype=heads)


def _set_attention(model: nn.Module, impl: str) -> None:
    from flashim.layers.transformer import Attention
    from flashim.quant.passes.transformer_block import QuantizedAttention

    for module in model.modules():
        if isinstance(module, (Attention, QuantizedAttention)):
            module.sdpa_impl = impl


def _fold_layer_scale(model: UFM, ctx: BuildContext) -> list[str]:
    from flashim.quant.passes.transformer_block import fold_layer_scale

    return fold_layer_scale(model)


def _fused_block_epilogues(model: UFM, ctx: BuildContext) -> list[str] | str:
    from flashim.quant.passes.transformer_block import enable_fused_epilogues

    return enable_fused_epilogues(model) if ctx.config.precision.quant == "fp8" else "skipped: not fp8"


def _fused_dynamic_vit_blocks(model: UFM, ctx: BuildContext) -> list[str] | str:
    if ctx.config.precision.quant not in ("int8", "fp8"):
        return "skipped: not quantized"
    from flashim.quant.passes.dynamic_transformer_block import (
        DynamicInt8TransformerBlock,
        fuse_dynamic_blocks,
    )

    fused = fuse_dynamic_blocks(model)
    for mod in model.modules():
        if isinstance(mod, DynamicInt8TransformerBlock):
            mod.fuse_gelu_quantize = True
    return fused or "skipped: no dynamic INT8 transformer blocks"


def _sage_platform(platform) -> bool:
    return platform.capability in _SAGE_IMPLS and importlib.util.find_spec("sageattention") is not None


def _vit_sage_attention(model: UFM, ctx: BuildContext) -> dict | str:
    """SageAttention (QK INT8 per warp, PV FP8) on encoder blocks 2..23 and all 12 info_sharing blocks;
    encoder blocks 0..1 stay on torch SDPA (matches DeDoDe-G's sensitivity rule)."""
    if ctx.config.precision.quant not in ("int8", "fp8"):
        return "skipped: not quantized"
    from flashim import ops

    impl = _SAGE_IMPLS[ctx.platform.capability]
    if impl not in {i.name for i in ops.sdpa.impls}:
        return f"skipped: {impl} not registered"
    enc_blocks = model.encoder.model.blocks
    for i in range(2, len(enc_blocks)):
        enc_blocks[i].attn.sdpa_impl = impl
    for block in model.info_sharing.self_attention_blocks:
        block.attn.sdpa_impl = impl
    return {"impl": impl, "encoder_layers": f"2-{len(enc_blocks) - 1}", "info_layers": "all"}


def _channels_last_heads(model: UFM, ctx: BuildContext) -> int:
    model.channels_last_heads = True
    count = 0
    for head in (model.head1[0], model.uncertainty_head[0]):
        for m in head.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                m.weight.data = m.weight.data.to(memory_format=torch.channels_last)
                count += 1
    return count


def _fused_dpt_upsample(model: UFM, ctx: BuildContext) -> int | str:
    """Replace PyTorch NHWC bilinear upsample in DPT heads with Triton ``ops.upsample_bilinear``
    (fusing static INT8 output quantization when feeding a quantized conv); 4.8-10x faster on sm_120."""
    if ctx.config.precision.compute_dtype == "fp32":
        return "skipped: fp32"
    count = 0
    for head in (model.head1[0], model.uncertainty_head[0]):
        for m in head.modules():
            if isinstance(m, (FeatureFusionBlock, DPTRegressionProcessor)):
                m.fused_upsample = True
                count += 1
    return count


def _dpt_int8_chain(model: UFM, ctx: BuildContext) -> int | str:
    """Enable conv1 -> conv2 INT8 output chaining in DPT ResidualConvUnits (measured -0.22 ms on sm_120,
    bit-identical to separate quantize_int8)."""
    if ctx.config.precision.quant != "int8":
        return "skipped: not int8"
    count = 0
    for head in (model.head1[0], model.uncertainty_head[0]):
        for m in head.modules():
            if isinstance(m, StaticInt8Conv2d):
                m.chain_output = True
                count += 1
    return count


def _cuda_graph(model: UFM, ctx: BuildContext) -> int | str:
    """Opt-in CUDA graph capture per working resolution (safe for torch SDPA / BF16; NOT enabled with
    SageAttention because released SageAttention launches on the legacy default stream)."""
    if ctx.config.engine.kind != "eager":
        return f"skipped: engine {ctx.config.engine.kind}"
    if not ctx.platform.supports(Feature.CUDA_GRAPHS):
        return f"skipped: no CUDA graphs on {ctx.platform.describe()}"
    model.graphs = int(ctx.option("cuda_graphs", 4))
    return model.graphs


SPEC = ModelSpec(
    name="ufm",
    role="dense",
    description="UFM-Base dense matcher (DINOv2 ViT-L/14 + 12-layer global-attention ViT + dual DPT heads)",
    load=_load,
    presets={
        "fp32": {"precision": {"compute_dtype": "fp32"}},
        "bf16": {
            "precision": {"compute_dtype": "bf16"},
            "optimize": {"passes": ["channels_last_heads", "fused_dpt_upsample"]},
        },
        "int8": {
            "extends": "bf16",
            "precision": {"quant": "int8"},
            "optimize": {
                "passes+": [
                    "fold_layer_scale",
                    "fused_dynamic_vit_blocks",
                    "vit_sage_attention",
                ]
            },
        },
        "fp8": {
            "extends": "bf16",
            "precision": {"quant": "fp8"},
            "optimize": {
                "passes+": [
                    "fold_layer_scale",
                    "fused_block_epilogues",
                    "fused_dynamic_vit_blocks",
                    "vit_sage_attention",
                ]
            },
        },
    },
    default_preset="bf16",
    quant_configs=quant.QUANT_CONFIGS,
    calibrations=quant.CALIBRATIONS,
    protected=quant.PROTECTED,
    passes={
        "fold_layer_scale": Pass(
            _fold_layer_scale, stage="pre_quant", exact=False,
            description="ViT LayerScale folded into attn.proj / mlp.fc2",
        ),
        "fused_block_epilogues": Pass(
            _fused_block_epilogues, stage="post_quant", exact=False,
            description="fp8: encoder biases / residual adds fused into MXFP8 GEMM epilogues and quantize kernels",
        ),
        "fused_dynamic_vit_blocks": Pass(
            _fused_dynamic_vit_blocks, stage="post_quant", exact=False,
            description="Dynamic INT8 transformer blocks with fused LayerNorm+quantize, HND attention quantize, "
            "GELU+quantize, and residual epilogues",
        ),
        "vit_sage_attention": Pass(
            _vit_sage_attention, stage="post_quant", requires=_sage_platform, exact=False,
            description="SageAttention (QK INT8 per warp, PV FP8) on encoder blocks 2-23 and all info_sharing blocks",
        ),
        "channels_last_heads": Pass(
            _channels_last_heads, stage="post_quant", exact=False,
            description="DPT head convolutions in channels-last memory format",
        ),
        "fused_dpt_upsample": Pass(
            _fused_dpt_upsample, stage="post_quant", exact=False,
            description="Triton bilinear upsampling (+ fused static INT8 quantize) in DPT heads",
        ),
        "dpt_int8_chain": Pass(
            _dpt_int8_chain, stage="post_quant",
            description="int8: conv1 -> conv2 INT8 quantization chaining in DPT ResidualConvUnits",
        ),
        "cuda_graph": Pass(
            _cuda_graph, stage="post_quant",
            description="Opt-in CUDA graph replay per working resolution (eager engine; torch SDPA only)",
        ),
    },
    engines={
        "eager": EngineSpec(),
        "compile": EngineSpec(COMPILE_REGIONS),
        "compile-full": EngineSpec(COMPILE_FULL_REGIONS),
    },
    set_precision=_set_precision,
    set_attention=_set_attention,
)
