"""DeDoDe's :class:`~flashim.build.ModelSpec` (presets ``fp32`` / ``bf16`` / ``int8`` / ``fp8``); descriptor: ``dedode`` = B, ``dedode-g`` = G. ``B`` (VGG19) or ``G`` (VGG19 + DINOv2 ViT-L/14)."""

from __future__ import annotations

from pathlib import Path

import torch
from torch import nn

from flashim import ops
from flashim.build import BuildContext, Pass
from flashim.config import RunConfig
from flashim.models.dedode import blocks, weights
from flashim.models.dedode.model import DEDODE_SAMPLER, DeDoDe
from flashim.models.sparse_spec import extractor_spec
from flashim.quant.passes.transformer_block import enable_fused_epilogues, fold_layer_scale

_VIT_PREFIX = "descriptor.encoder.dinov2.blocks."
_VIT = f"{_VIT_PREFIX}*"
# The ViT cannot take static INT8: measured on the dense descriptor map vs BF16 (2 MegaDepth images), INT8 ViT linears with
# one calibrated per-tensor scale give cosine mean 0.79 / min 0.01 (0.995 / 0.87 with the ViT excluded; attention and the
# MLP down-projection each break it). DINOv2's outliers are whole tokens ("massive activations"), so dedode-g's int8
# preset quantizes ``VIT_LINEARS`` with one scale per token computed at run time (``int8_dynamic``: ``DynamicInt8Linear``);
# ``VIT_IGNORE`` (BF16 ViT) is what DaD's int8 preset still uses. MXFP8's per-32-block scales are fine (descriptor
# cosine 0.994).
VIT_IGNORE = (f"{_VIT}.*",)
VIT_LINEARS = (f"{_VIT}.attn.qkv", f"{_VIT}.attn.proj", f"{_VIT}.mlp.fc1", f"{_VIT}.mlp.fc2")
# Quantized: the VGG convs after the first (3-channel) one, every ConvRefiner 1x1 (block1 and the per-block pointwise
# convs; the depthwise 5x5s stay BF16) and DINOv2-G's transformer linears. Protected: the first VGG conv, DINOv2's patch
# embedding, ``out_conv`` (logits / descriptors) and the ViT's layer norms are not targeted. The hidden blocks the
# ``fused_bf16_blocks`` pass replaces (32 / 64 / 128 channels) are gone before quantization, so their ``.3`` convs are
# not quantized; the 256 / 512 / 768-channel ones are.
QUANTIZED = (
    "detector.encoder.layers.[1-9]*", "descriptor.encoder.layers.[1-9]*", "descriptor.encoder.vgg.layers.[1-9]*",
    "*.decoder.layers.*.block1.0", "*.decoder.layers.*.block1.3", "*.decoder.layers.*.hidden_blocks.*.3",
    *VIT_LINEARS,
)
PROTECTED = (
    "detector.encoder.layers.0", "descriptor.encoder.layers.0", "descriptor.encoder.vgg.layers.0",
    "*.out_conv", "descriptor.encoder.dinov2.patch_embed.*",
)


def _fold_layer_scale(model: nn.Module, ctx: BuildContext) -> list[str] | str:
    """DINOv2's LayerScale folded into ``attn.proj`` / ``mlp.fc2`` before quantization, when the preset quantizes them
    (per-output-channel weight scales absorb it): no scale pass left to run, in ``fused_block_epilogues`` (fp8) or
    after the dynamic INT8 linears (dedode-g int8). Skipped where the ViT stays BF16 (DaD's int8: ``VIT_IGNORE``)."""
    quant = ctx.config.precision.quant
    if not quant or ctx.spec.quant_configs[quant].scheme_for(f"{_VIT_PREFIX}0.attn.proj") is None:
        return "skipped: ViT not quantized"
    return fold_layer_scale(model)


def _fused_dynamic_vit_blocks(model: nn.Module, ctx: BuildContext) -> list[str] | str:
    """int8: DINOv2 blocks with dynamic INT8 linears (dedode-g) with LayerNorm + per-row quantize in one kernel and the
    GELU / residual add in the GEMM epilogues (``DynamicInt8TransformerBlock``); bit-identical. Pieces without a kernel
    for the platform keep running as the unfused modules."""
    if ctx.config.precision.quant != "int8":
        return "skipped: not int8"
    from flashim.quant.passes.dynamic_transformer_block import fuse_dynamic_blocks

    return fuse_dynamic_blocks(model) or "skipped: no dynamic INT8 transformer blocks"


def _fused_block_epilogues(model: nn.Module, ctx: BuildContext) -> list[str] | str:
    """The ViT's biases and residual adds inside neighbouring kernels (``QuantizedTransformerBlock.fused_epilogues``):
    measured as separate elementwise passes of 8.7 ms per DaD fp8 extract (benchmarks/profile/timeline_audit.py)."""
    return enable_fused_epilogues(model) if ctx.config.precision.quant == "fp8" else "skipped: not fp8"


def fused_decoder_upsample(model: nn.Module, ctx: BuildContext) -> list[str] | str:
    """The decoders' between-scale bilinear upsampling fused into the next scale's concat / add
    (``DeDoDeNet.fused_upsample``); skipped for ``fp32`` (the reference numerics)."""
    if ctx.config.precision.compute_dtype == "fp32":
        return "skipped: fp32"
    nets = [name for name, module in model.named_modules() if isinstance(module, blocks.DeDoDeNet)]
    for name in nets:
        model.get_submodule(name).fused_upsample = True
    return nets


DECODER_UPSAMPLE_PASS = {
    "fused_decoder_upsample": Pass(
        fused_decoder_upsample, exact=False, description="decoder upsampling fused into the next concat / add"
    ),
}


_DECODERS = ("detector.decoder.", "descriptor.decoder.")


def fold_refiner_residual(model: nn.Module, ctx: BuildContext) -> list[str] | str:
    """int8: every ConvRefiner's residual ``/ 1.4`` folded into its ``out_conv`` weights (``ConvRefiner.fold_residual_scale``,
    before the BF16 cast). Measured int8 dedode extract at 1200 px (sm_120, interleaved): -2.0 ms; bf16 within noise
    (59.1 -> 58.3 ms), so bf16 / fp8 keep their numerics."""
    if ctx.config.precision.quant != "int8":
        return "skipped: not int8"
    return [name for name, module in model.named_modules()
            if isinstance(module, blocks.ConvRefiner) and module.fold_residual_scale()]


def int8_decoder_producers(model: nn.Module, ctx: BuildContext) -> list[str] | str:
    """int8: the ConvRefiner hidden blocks left unfused by ``fused_bf16_blocks`` (256 / 512 / 768 channels: cuDNN BF16
    depthwise, a ReLU pass, the INT8 quantize pass, the INT8 1x1 GEMM) with the depthwise conv as the INT8 producer
    (``Int8ProducerSeparableConvBlock``)."""
    if ctx.config.precision.quant != "int8":
        return "skipped: not int8"
    from flashim.quant.passes.separable_conv_block import fuse_int8_producers

    return fuse_int8_producers(model, _DECODERS)


def int8_refiner_input(model: nn.Module, ctx: BuildContext) -> list[str] | str:
    """int8: the refiner input (features + upsampled context) written as ``block1.0``'s INT8 input
    (``ConvRefiner.quantized_input``); bit-identical."""
    if ctx.config.precision.quant != "int8":
        return "skipped: not int8"
    from flashim.quant.methods.int8_linear import StaticInt8Conv1x1

    refiners = []
    for name, module in model.named_modules():
        if (isinstance(module, blocks.ConvRefiner) and isinstance(module.block1[0], StaticInt8Conv1x1)
                and isinstance(module.block1[1], nn.Identity) and isinstance(module.block1[2], nn.ReLU)):
            module.quantized_input = True
            refiners.append(name)
    return refiners


# ConvRefiner decoder fusions for the int8 preset (ported from the LoMa experiments, benchmarks/experimental/loma on
# feat/loma). dedode int8 extract at 1200 px, 4096 keypoints (sm_120, eager, interleaved best of 4 in one process): 42.0 ->
# 33.9 ms with all three; one at a time -3.1 (producers), -2.4 (refiner input), -2.0 ms (residual fold).
DECODER_FUSION_PASSES = {
    "fold_refiner_residual": Pass(
        fold_refiner_residual, stage="pre_quant", exact=False, description="refiner residual 1/1.4 folded into out_conv"
    ),
    "int8_decoder_producers": Pass(
        int8_decoder_producers, exact=False,
        description="int8: depthwise 5x5 + ReLU as the INT8 producer of the 1x1 (256 / 512 / 768-channel blocks)",
    ),
    "int8_refiner_input": Pass(
        int8_refiner_input, description="int8: upsample-concat writes block1.0's INT8 input"
    ),
}


def fused_pool_epilogue(model: nn.Module, ctx: BuildContext) -> list[str] | str:
    """int8: every ``INT8 conv -> ReLU -> MaxPool2d(2) -> INT8 conv`` of the VGG encoders as one kernel
    (``VGG.pooled_convs``, ``ops.int8_conv2d_maxpool``); bit-identical."""
    if ctx.config.precision.quant != "int8":
        return "skipped: not int8"
    found = []
    for name, module in model.named_modules():
        if isinstance(module, blocks.VGG):
            module.pooled_convs = module.find_pooled_convs()
            found += [f"{name}.layers.{i}" for i in sorted(module.pooled_convs)]
    return found


def fused_first_conv_epilogue(model: nn.Module, ctx: BuildContext) -> list[str] | str:
    """int8: the first VGG conv (BF16, 3 -> 64) feeding an INT8 conv: cuDNN without the bias, then bias, ReLU and the
    next conv's quantize as one pass (``VGG.first_conv_relu``, ``ops.bias_relu_quantize_int8``); bit-identical to the
    cuDNN path, which rounds the convolution to BF16 before adding the bias (a CPU BF16 conv adds it in FP32)."""
    if ctx.config.precision.quant != "int8":
        return "skipped: not int8"
    if ctx.platform.device_type != "cuda":
        return "skipped: the bias is added after the BF16 rounding of the convolution only by cuDNN"
    found = []
    for name, module in model.named_modules():
        if isinstance(module, blocks.VGG):
            module.first_conv_relu = module.find_first_conv_relu()
            if module.first_conv_relu is not None:
                found.append(f"{name}.layers.0")
    return found


# The VGG encoders' stem for the int8 preset (bit-identical, see ``blocks.VGG``).
STEM_EXACT_PASSES = {
    "fused_pool_epilogue": Pass(
        fused_pool_epilogue, description="int8: max pool + next conv's quantize in the INT8 conv's epilogue"
    ),
    "fused_first_conv_epilogue": Pass(
        fused_first_conv_epilogue, description="int8: first conv's bias + ReLU + quantize as one pass"
    ),
}


def tensor_core_first_conv(model: nn.Module, ctx: BuildContext) -> list[str] | str:
    """int8: the first VGG conv (BF16, 3 -> 64, 3x3) feeding an INT8 conv as one tensor-core kernel with the bias / ReLU /
    quantize epilogue (``ops.conv2d_small_cin_quantize_int8``) instead of cuDNN's SIMT kernel; FP32 summation order
    differs (measured 1 BF16 ulp on ~2e-7 of the outputs). Only where the op has a kernel (sm_89 / sm_90, the
    architectures it was measured on)."""
    if ctx.config.precision.quant != "int8":
        return "skipped: not int8"
    found = []
    for name, module in model.named_modules():
        if not isinstance(module, blocks.VGG) or (relu := module.find_first_conv_relu()) is None:
            continue
        conv = module.layers[0]
        if (conv.in_channels, conv.kernel_size, conv.stride, conv.padding, conv.dilation, conv.groups) != (
            3, (3, 3), (1, 1), (1, 1), (1, 1), 1
        ):
            continue
        bound = ops.conv2d_small_cin_quantize_int8.bind(conv.weight, conv.bias).warmup()
        if bound.primary == "reference":
            return f"skipped: no kernel on {ctx.platform.describe()}"
        module.tensor_core_first_conv = (relu, bound)
        found.append(f"{name}.layers.0")
    return found


def patch_embed_gemm(model: nn.Module, ctx: BuildContext) -> list[str] | str:
    """int8: DINOv2's patch convolution + class token + positional embedding as one GEMM into the token buffer
    (``ops.patch_embed_tokens``; FP32 summation order differs from cuDNN's convolution). Only where the op has a kernel
    (sm_89 / sm_90)."""
    if ctx.config.precision.quant != "int8":
        return "skipped: not int8"
    found = []
    for name, module in model.named_modules():
        if isinstance(module, blocks.DinoV2) and module.patch_embed.proj.weight.dtype == torch.bfloat16:
            proj = module.patch_embed.proj
            bound = ops.patch_embed_tokens.bind(proj.weight, proj.bias).warmup()
            if bound.primary == "reference":
                return f"skipped: no kernel on {ctx.platform.describe()}"
            module.patch_embed_op = bound
            found.append(name)
    return found


# Numerics-changing stem passes (int8, accuracy-gated on MegaDepth-1500, see the PR).
STEM_TC_PASSES = {
    "tensor_core_first_conv": Pass(
        tensor_core_first_conv, exact=False, description="int8: 3 -> 64 first conv on tensor cores (sm_89 / sm_90)"
    ),
}
PATCH_EMBED_PASS = {
    "patch_embed_gemm": Pass(
        patch_embed_gemm, exact=False, description="int8: DINOv2 patch embedding as one GEMM (sm_89 / sm_90)"
    ),
}


# For the DINOv2 descriptor (dedode-g, dad).
VIT_EPILOGUE_PASSES = {
    "fold_layer_scale": Pass(
        _fold_layer_scale, stage="pre_quant", exact=False,
        description="ViT LayerScale folded into proj / fc2 (when quantized)",
    ),
    "fused_dynamic_vit_blocks": Pass(
        _fused_dynamic_vit_blocks, description="int8: ViT LayerNorm + row quantize, GELU and residual in one kernel each",
    ),
    "fused_block_epilogues": Pass(
        _fused_block_epilogues, stage="post_quant", exact=False,
        description="ViT biases / residual adds fused into GEMM epilogues and quantize kernels (fp8)",
    ),
}


# SageAttention impl per architecture (``flashim.backends.external.sageattention``): QK INT8 with per-warp scales, PV FP8.
# sm_89 kernel on Ada, the WGMMA kernel on Hopper (the sm_89 kernel on an H20 is slower than cuDNN SDPA).
_SAGE_IMPLS = {(8, 9): "sage-fp8-warp", (9, 0): "sage-fp8-sm90-warp"}
VIT_SAGE_LAYERS = "2-23"


def parse_layers(spec: str, depth: int = 24) -> list[int]:
    """``"all"``, ``"2-23"``, ``"0,3,5"`` or ``"!0-1"`` (every layer but those) -> sorted 0-based block indices."""
    if spec.startswith("!"):
        excluded = set(parse_layers(spec[1:], depth))
        return [i for i in range(depth) if i not in excluded]
    if spec == "all":
        return list(range(depth))
    layers: set[int] = set()
    for item in filter(None, spec.split(",")):
        first, _, last = item.partition("-")
        layers.update(range(int(first), int(last or first) + 1))
    if not all(0 <= i < depth for i in layers):
        raise ValueError(f"vit_attention layers {spec!r} outside 0..{depth - 1}")
    return sorted(layers)


def _sage_platform(platform) -> bool:
    import importlib.util

    return platform.capability in _SAGE_IMPLS and importlib.util.find_spec("sageattention") is not None


def _vit_sage_attention(model: nn.Module, ctx: BuildContext) -> list[str] | str:
    """int8: SageAttention (QK INT8, PV FP8) on a subset of the DINOv2 blocks, the rest stay on torch SDPA. Option
    ``optimize.options.vit_attention = {"layers": "2-23", "impl": "auto"}`` (``layers``: see :func:`parse_layers`, ``""``
    = none; ``impl``: an ``ops.sdpa`` impl name, ``auto`` = :data:`_SAGE_IMPLS` of the platform). Layers 0 and 1 stay
    out by default. Relative L2 error of the ViT output against the same ViT with FP32 attention in every layer (10
    MegaDepth images at 1200 px; the INT8 ViT with torch SDPA sits at 7.1e-2): sage on all 24 layers 8.2e-2 (L20) / 8.6e-2
    (H20), on layers 2-23 7.5e-2 / 7.4e-2; layers 0 and 1 alone give 8.0e-2 and 7.7e-2 on L20, any single later layer
    7.1-7.5e-2. Per-warp QK scales measured faster than per-thread at the same error class."""
    if ctx.config.precision.quant != "int8":
        return "skipped: not int8"
    dinov2 = getattr(getattr(model.descriptor, "encoder", None), "dinov2", None)
    if dinov2 is None:  # a VGG-only descriptor (LoMa-B128)
        return "skipped: no ViT"
    options = ctx.option("vit_attention", {})
    blocks_ = dinov2.blocks
    layers = parse_layers(options.get("layers", VIT_SAGE_LAYERS), len(blocks_))
    impl = options.get("impl", "auto")
    impl = _SAGE_IMPLS[ctx.platform.capability] if impl == "auto" else impl
    for i in layers:
        blocks_[i].attn.sdpa_impl = impl
    return {"impl": impl, "layers": layers}


# dedode-g only (dad shares the ViT but keeps its BF16 numerics).
VIT_ATTENTION_PASSES = {
    "vit_sage_attention": Pass(
        _vit_sage_attention, requires=_sage_platform, exact=False,
        description="int8: SageAttention (QK INT8 per warp, PV FP8) on DINOv2 blocks 2-23 (sm_89, sm_90)",
    ),
}


def _loader(variant: str):
    def load(config: RunConfig) -> DeDoDe:
        return DeDoDe.from_pretrained(
            blocks.DETECTOR_L, blocks.DESCRIPTORS[variant], DEDODE_SAMPLER,
            detector_state=weights.load("detector-L"),
            descriptor_state=weights.load(f"descriptor-{variant}"),
            dinov2_state=weights.load("dinov2") if variant == "G" else None,
            device=config.device,
        )

    return load


def _spec(name: str, variant: str, description: str, int8_dynamic: tuple[str, ...] = (), vit: str | None = None):
    return extractor_spec(
        name=name,
        description=description,
        load=_loader(variant),
        calibration_dir=Path(__file__).parent / "calibration",
        quantized=QUANTIZED,
        protected=PROTECTED,
        int8_dynamic=int8_dynamic,
        compile_regions=("detector", "descriptor"),
        # BF16 extract at 1200 px, interleaved in one process (sm_120): dedode 128 -> 102 ms, dedode-g 184 -> 160 ms.
        channels_last=True,
        # ConvRefiner hidden blocks with 32 / 64 / 128 channels as one BF16 kernel each (26 blocks). BF16 extract at
        # 1200 px, pass off vs on in one process (sm_120): dedode 101.8 -> 77.8 ms, dedode-g 174.5 -> 151.0 ms.
        fused_bf16_blocks=_DECODERS,
        extra_passes={
            **DECODER_UPSAMPLE_PASS, **DECODER_FUSION_PASSES, **STEM_EXACT_PASSES, **STEM_TC_PASSES,
            **(VIT_EPILOGUE_PASSES | PATCH_EMBED_PASS if vit else {}),
            **(VIT_ATTENTION_PASSES if vit == "dedode-g" else {}),
        },
        structural_passes=(
            *DECODER_UPSAMPLE_PASS, *DECODER_FUSION_PASSES, *STEM_EXACT_PASSES, *STEM_TC_PASSES,
            *((*VIT_EPILOGUE_PASSES, *PATCH_EMBED_PASS) if vit else ()),
            *(VIT_ATTENTION_PASSES if vit == "dedode-g" else ()),
        ),
    )


# One spec per descriptor: the two descriptors' layers differ in names and shapes, so each has its own calibration table.
SPEC = _spec("dedode", "B", "DeDoDe v2 detector (L) + descriptor B (VGG19), dual-softmax matching")
SPEC_G = _spec(
    "dedode-g", "G", "DeDoDe v2 detector (L) + descriptor G (VGG19 + DINOv2 ViT-L/14), dual-softmax matching",
    int8_dynamic=VIT_LINEARS, vit="dedode-g",
)
