"""RoMaV2's :class:`~flashim.build.ModelSpec`.

Presets (``bf16`` / ``int8`` / ``fp8``) are generated from :data:`flashim.models.romav2.quant.PRESETS` (whose docstring
holds the accuracy evidence), so the quantization config and the refiner flags of a preset keep one definition. RoMaV2
always runs mixed BF16 internally (DINOv3 weights BF16, autocast), so every preset's ``compute_dtype`` is ``bf16`` and
``fp32`` builds ``bf16``. :data:`ORACLE_OVERLAYS` are the configs bit-identical to RoMaV2's FastFP8 runtime, for the
parity tests and the oracle speed comparison (``flashim.load("romav2", "bf16", config=ORACLE_OVERLAYS["fp8-p8"])``).
"""

from __future__ import annotations

import os

import torch
from torch import nn

from flashim.build import BuildContext, EngineSpec, ModelSpec, Pass
from flashim.config import RunConfig
from flashim.models.romav2 import quant
from flashim.models.romav2.config import RoMaV2Config
from flashim.models.romav2.model import RoMaV2

# RoMaV2's validated regions: compiled numerics are covered by its P8 AUC results.
COMPILE_REGIONS = ("f", "matcher.mv_vit")
# compile-full: also VGG, the DPT head and every refiner's conv stack (+ VGG BN fold).
COMPILE_FULL_REGIONS = (
    *COMPILE_REGIONS,
    "refiner_features",
    "matcher.head",
    *(f"refiners.{stride}.{part}" for stride in (4, 2, 1) for part in ("block1", "hidden_blocks")),
)


def _load(config: RunConfig) -> RoMaV2:
    return RoMaV2.from_pretrained(
        RoMaV2Config(**config.arch), checkpoint=config.checkpoint, device=config.device
    )


def _fold_refiners(model: RoMaV2, ctx: BuildContext) -> list[str]:
    """Refiner conv weights channels-last (and refiners build their inputs channels-last), BatchNorm folded."""
    from flashim.quant.passes.separable_conv_block import fold_block_batchnorm

    for refiner in model.refiners.values():
        refiner.channels_last = True
        for child in refiner.modules():
            if isinstance(child, nn.Conv2d):
                child.weight.data = child.weight.data.to(memory_format=torch.channels_last)
    return fold_block_batchnorm(model.refiners)


def _fold_fine_features(model: RoMaV2, ctx: BuildContext) -> int:
    """VGG19-BN fine features: BatchNorm folded into the convs, channels-last weights (what ``compile-full`` sets up
    for its compiled VGG). Not exact: the fold changes rounding."""
    return model.refiner_features.fold_batchnorm(channels_last=True)


def _fused_bf16(stride: int):
    def apply(model: RoMaV2, ctx: BuildContext) -> list[str]:
        from flashim.quant.passes.separable_conv_block import fuse_bf16_blocks

        fused = fuse_bf16_blocks(model, (f"refiners.{stride}.",))
        if model.quant_report is not None:
            model.quant_report.fused_conv_blocks.extend(fused)
        return fused

    return apply


def _fused_refiner_io(model: RoMaV2, ctx: BuildContext) -> None:
    for refiner in model.refiners.values():
        refiner.fused_input = True
        refiner.enable_fused_heads()


def _fused_mx_blocks(model: RoMaV2, ctx: BuildContext) -> int:
    from flashim.quant.passes.separable_conv_block import MXFP8SeparableConvBlock

    blocks = [m for m in model.modules() if isinstance(m, MXFP8SeparableConvBlock)]
    for block in blocks:
        block.fuse_pointwise = True
    return len(blocks)


def _fused_rope(model: RoMaV2, ctx: BuildContext) -> int:
    """DINOv3 and MV-ViT RoPE through ``ops.rope_qk`` dispatch (one fused kernel for q and k) instead of the pinned
    reference math. On quantized blocks (BF16 tables) the Triton kernel is bit-identical to reference; on unquantized
    DINOv3 (FP32 tables) FMA contraction may move elements by one BF16 ulp. Measured on sm_120 (DINOv3 + MV-ViT fp8,
    mega1500): ~7 ms saved in eager."""
    pinned = [module for module in model.modules() if getattr(module, "rope_impl", None) == "reference"]
    for module in pinned:
        module.rope_impl = None
    return len(pinned)


def _fold_layer_scale(model: RoMaV2, ctx: BuildContext) -> list[str]:
    """Fold DINOv3 LayerScale into ``attn.proj`` / ``mlp.fc2`` before quantization (absorbed into per-channel weight
    scales). Enables ``fused_block_epilogues`` (fp8) and ``int8_linear(residual=)`` on ``fc2`` (int8)."""
    from flashim.quant.passes.transformer_block import fold_layer_scale

    return fold_layer_scale(model)


def _fused_block_epilogues(model: RoMaV2, ctx: BuildContext) -> list[str]:
    """Fuse MXFP8 transformer block biases and residual adds into cuBLAS / LayerNorm / GELU kernels
    (``QuantizedTransformerBlock.fused_epilogues``) and standalone ``StaticMXFP8Linear`` biases into cuBLAS."""
    from flashim.quant.methods.linear import enable_fused_bias
    from flashim.quant.passes.transformer_block import enable_fused_epilogues

    enabled = enable_fused_epilogues(model)
    enable_fused_bias(model)
    return enabled


def _batched_directions(model: RoMaV2, ctx: BuildContext) -> None:
    model.batched_directions = True


def _batched_matcher(model: RoMaV2, ctx: BuildContext) -> int:
    """Share ``logits_ba = logits_ab.T`` and batch AB+BA through a channels-last DPTHead (eliminates one 2500^2x1024
    FP32 similarity GEMM and NCHW<->NHWC copies inside DPTHead)."""
    model.matcher.batched_directions = True
    model.matcher.head.channels_last = True
    converted = 0
    for child in model.matcher.head.modules():
        if isinstance(child, (nn.Conv2d, nn.ConvTranspose2d)):
            child.weight.data = child.weight.data.to(memory_format=torch.channels_last)
            converted += 1
    return converted


SAGE_IMPLS = {(8, 9): "sage-fp8", (9, 0): "sage-fp8-sm90", (12, 0): "sage-fp8"}


def _sage_platform(platform) -> bool:
    import importlib.util

    return platform.capability in SAGE_IMPLS and importlib.util.find_spec("sageattention") is not None


def _sage_attention(model: RoMaV2, ctx: BuildContext) -> dict | str:
    """Quantized presets: SageAttention (QK INT8, PV FP8) on DINOv3 and MV-ViT blocks; ``optimize.attention`` overrides
    it. Measured on sm_120 (mega1500, eager): fp8 144.5 -> 135.7 ms (-8.8 ms), int8 117.7 -> 110.5 ms (-7.2 ms)."""
    if ctx.config.precision.quant is None:
        return "skipped: not quantized"
    from flashim import ops

    impl = SAGE_IMPLS[ctx.platform.capability]
    if impl not in {i.name for i in ops.sdpa.impls}:
        return f"skipped: {impl} not registered"
    _set_attention(model, impl)
    return {"impl": impl}


def _set_attention(model: nn.Module, impl: str) -> None:
    from flashim.quant.passes.transformer_block import QuantizedAttention

    for module in model.modules():
        if isinstance(module, QuantizedAttention):
            module.sdpa_impl = impl


def _compile_full_setup(model: RoMaV2, ctx: BuildContext) -> None:
    model.refiner_features.fold_batchnorm(channels_last=True)


def _apply_inference(model: RoMaV2, inference: dict) -> None:
    model.fast_sampling = inference.get("fast_sampling", os.environ.get("FLASHIM_FAST_SAMPLING", "1") != "0")


def _validate(config: RunConfig) -> None:
    if config.optimize.attention != "sdpa" and config.precision.quant is None:
        raise ValueError(f"attention={config.optimize.attention!r} needs a quantized preset (fused transformer blocks)")
    passes = config.optimize.passes
    if "batched_directions" in passes and "fused_refiner_io" not in passes:
        raise ValueError("batched_directions requires fused_refiner_io")


def _overlay(preset: quant.Preset) -> dict:
    passes = ["fold_refiners"] if preset.fold_refiners else []
    passes += ["fold_fine_features"] if preset.fold_fine_features else []
    passes += ["fold_layer_scale"] if preset.fold_layer_scale else []
    passes += [f"fused_bf16_s{stride}" for stride in preset.fused_bf16_refiners]
    passes += [name for name, on in (
        ("fused_refiner_io", preset.fused_refiner_io),
        ("fused_mx_blocks", preset.fused_mx_blocks),
        ("batched_directions", preset.batched_directions),
        ("batched_matcher", preset.batched_matcher),
        ("fused_rope", preset.fused_rope),
        ("fused_block_epilogues", preset.fused_block_epilogues),
        ("sage_attention", preset.sage_attention),
    ) if on]
    return {
        "precision": {"compute_dtype": "bf16", "quant": preset.name if preset.config is not None else None},
        "optimize": {"passes": passes},
    }


_QUANTIZED = {name: p for name, p in {**quant.PRESETS, **quant.ORACLE_PRESETS}.items() if p.config is not None}
ORACLE_OVERLAYS = {name: _overlay(p) for name, p in quant.ORACLE_PRESETS.items()}

SPEC = ModelSpec(
    name="romav2",
    role="dense",
    description="RoMa v2 dense matcher (DINOv3 ViT-L/16 + MV-ViT + DPT + conv refiners)",
    load=_load,
    presets={name: _overlay(p) for name, p in quant.PRESETS.items()},
    default_preset="fp8",
    quant_configs={name: p.config for name, p in _QUANTIZED.items()},
    calibrations={name: p.calibration for name, p in _QUANTIZED.items()},
    protected=quant.PROTECTED,
    passes={
        "fold_refiners": Pass(_fold_refiners, stage="pre_quant", description="channels-last refiners, BN folded"),
        "fold_fine_features": Pass(
            _fold_fine_features, stage="pre_quant", exact=False, description="channels-last VGG fine features, BN folded"
        ),
        "fold_layer_scale": Pass(
            _fold_layer_scale, stage="pre_quant", exact=False, description="ViT LayerScale folded into proj / fc2"
        ),
        **{
            f"fused_bf16_s{s}": Pass(_fused_bf16(s), exact=False, description=f"fused BF16 stride-{s} refiner blocks")
            for s in (4, 2, 1)
        },
        "fused_refiner_io": Pass(_fused_refiner_io, exact=False, description="fused refiner input + output heads"),
        "fused_mx_blocks": Pass(_fused_mx_blocks, exact=False, description="dw + quantize + pw MXFP8 in one kernel"),
        "batched_directions": Pass(_batched_directions, description="both match directions as one batch"),
        "batched_matcher": Pass(
            _batched_matcher, exact=False,
            description="share logits_ba = logits_ab^T and batch AB+BA through channels-last DPTHead",
        ),
        "fused_rope": Pass(_fused_rope, exact=False, description="DINOv3 + MV-ViT RoPE as one fused kernel"),
        "fused_block_epilogues": Pass(
            _fused_block_epilogues, stage="post_quant", exact=False,
            description="MXFP8 ViT biases / residual adds fused into GEMM / LayerNorm / GELU kernels",
        ),
        "sage_attention": Pass(
            _sage_attention, requires=_sage_platform, exact=False,
            description="quantized ViT attention on SageAttention (QK INT8, PV FP8)",
        ),
    },
    engines={
        "eager": EngineSpec(),
        "compile": EngineSpec(COMPILE_REGIONS),
        "compile-full": EngineSpec(COMPILE_FULL_REGIONS, setup=_compile_full_setup),
    },
    set_attention=_set_attention,
    apply_inference=_apply_inference,
    validate=_validate,
)

