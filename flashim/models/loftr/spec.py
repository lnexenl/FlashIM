"""LoFTR's :class:`~flashim.build.ModelSpec` (presets ``fp32`` / ``bf16`` / ``int8`` / ``fp8``); variant ``outdoor``
(MegaDepth, default), ``indoor`` (ScanNet), or ``minima`` (MINIMA modality-invariant weights, also registered directly
as ``minima-loftr``).

``minima-loftr`` optimization passes (RTX 5070 Ti, sm_120, 840 px eager):

- ``fused_linear_attention``: Triton ELU+1 linear attention (196 -> 31.4 us coarse, 167 -> 21 us fine; 66-74% DRAM SOL).
- ``fused_qkv``: merges ``q/k/v_proj`` -> ``qkv_proj`` (self) and ``k/v_proj`` -> ``kv_proj`` (cross), plus batches
  self-attention across both images (including unequal-length pairs).
- ``fused_encoder_epilogues``: single-pass ``ops.layernorm_cat*`` (78.5% SM/L1 SOL), ``mlp.0`` ReLU epilogue, and
  ``ops.layernorm_add`` (2-3 kernels -> 1 across all 10 layers).
- ``fused_fpn_upsample``: Triton ``ops.upsample_add(align_corners=True)`` (0.478 -> 0.112 ms, 87.5% DRAM SOL / 771 GB/s).
- ``bf16_coarse_similarity``: ``ops.matmul_nt_fp32`` cuBLAS BF16->FP32 GEMM (1.52 -> 1.00 ms on 8190x8085).
- ``batch_backbone``: runs ``backbone`` once with ``B=2`` for equal-resolution pairs.
- Quantization: ``int8`` uses dynamic per-token INT8 on all 60 transformer linears + static INT8 on backbone convs
  (improves MegaDepth agreement +0.85% over all-static INT8 and avoids static-scale clipping on cross-modal inputs);
  ``fp8`` uses static MXFP8 on both. ``dual_softmax_mutual`` two-pass was tested and discarded on sm_120 (1.051 vs
  1.017 ms for three-pass, which already runs at 77-92% DRAM SOL).
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

import torch
from torch import nn

from flashim.build import BuildContext, EngineSpec, ModelSpec, Pass
from flashim.config import RunConfig
from flashim.models.loftr.model import LocalFeatureTransformer, LoFTR, LoFTREncoderLayer
from flashim.models.sparse_spec import extractor_spec

CALIBRATION_DIR = Path(__file__).parent / "calibration"

# Quantized: the backbone's 3x3 / 1x1 convs after the 1-channel 7x7 stem and every transformer linear (q/k/v/merge/MLP,
# coarse and fine). Protected: the stem and the coarse -> fine feature projections (``fine_preprocess``), which feed the
# fine offsets directly.
_LINEARS = ("q_proj", "k_proj", "v_proj", "merge", "mlp.0", "mlp.2")
TRANSFORMER_LINEARS = tuple(
    f"{stage}.layers.*.{leaf}" for stage in ("loftr_coarse", "loftr_fine") for leaf in _LINEARS
)
QUANTIZED = (
    "backbone.layer[123].[01].conv[12]", "backbone.layer[123].[01].downsample.0", "backbone.layer[123]_outconv",
    "backbone.layer[12]_outconv2.[03]",
    *TRANSFORMER_LINEARS,
)
PROTECTED = ("backbone.conv1", "fine_preprocess.*")


def _compile_regions(model: nn.Module) -> tuple[str, ...]:
    """Skip ``backbone`` when quantized convs are present (Dynamo 2.11 cannot trace ``@triton.autotune`` inside
    ``StaticInt8Conv2d`` / ``StaticMXFP8Conv2d``)."""
    from flashim.quant.methods.conv2d import StaticInt8Conv2d, StaticMXFP8Conv2d

    if any(isinstance(m, (StaticInt8Conv2d, StaticMXFP8Conv2d)) for m in model.backbone.modules()):
        return ()
    return ("backbone",)


def _loader(default_variant: str):
    def load(config: RunConfig) -> LoFTR:
        return LoFTR.from_pretrained(
            variant=config.variant or default_variant, checkpoint=config.checkpoint, device=config.device, **config.arch
        )

    return load


def _channels_last(model: nn.Module, ctx: BuildContext) -> str:
    """NHWC backbone weights for the cuDNN convolutions: measured 24.0 -> 21.2 ms per 896x1200 image in BF16 and
    43.0 -> 39.7 ms in FP32 on sm_120 (the NCHW <-> NHWC conversions around every conv go away); MegaDepth-1500 AUC@5
    0.6657 vs 0.6638 without it (4 seeds, std 0.0005). Skipped for the
    quantized presets, whose backbone convs are im2col + GEMM modules on NCHW buffers."""
    if ctx.config.precision.quant:
        return "skipped: quantized backbone"
    model.backbone.to(memory_format=torch.channels_last)
    return "backbone"


def _bf16_linear_attention(model: nn.Module, ctx: BuildContext) -> int | str:
    """BF16 operands for the linear-attention GEMMs of every transformer layer (see ``linear_attention``): measured
    75.4 -> 71.2 ms per 1200 px pair; MegaDepth-1500 AUC@5 0.6650 vs 0.6657 without it (4 seeds, std 0.0005). Skipped
    for ``fp32`` (the reference numerics)."""
    if ctx.config.precision.compute_dtype == "fp32":
        return "skipped: fp32"
    layers = [m for m in model.modules() if isinstance(m, LoFTREncoderLayer)]
    for layer in layers:
        layer.attention_gemm_dtype = torch.bfloat16
    return len(layers)


def _fold_similarity_temperature(model: nn.Module, ctx: BuildContext) -> str:
    """Coarse similarity ``(a / 0.1) b^T`` instead of ``(a b^T) / 0.1``: the division was an elementwise pass over the
    16.8k x 16.5k FP32 matrix at 1200 px (2.9 ms per pair on sm_120, benchmarks/profile/timeline_audit.py). Same math,
    different rounding of the logits; skipped for ``fp32`` (the reference numerics)."""
    if ctx.config.precision.compute_dtype == "fp32":
        return "skipped: fp32"
    model.fold_temperature = True
    return "coarse matching"


def _fused_linear_attention(model: nn.Module, ctx: BuildContext) -> int | str:
    """Route every ``LoFTREncoderLayer`` through fused Triton ``ops.linear_attention`` (4.1-15.5x faster than
    unfused einsum/bmm on sm_120: 196 -> 31.4 us coarse, 167 -> 21 us fine)."""
    if ctx.config.precision.compute_dtype == "fp32":
        return "skipped: fp32"
    layers = [m for m in model.modules() if isinstance(m, LoFTREncoderLayer)]
    for layer in layers:
        layer.use_fused_linear_attention = True
    return len(layers)


def _bf16_coarse_similarity(model: nn.Module, ctx: BuildContext) -> str:
    """Coarse similarity via ``ops.matmul_nt_fp32`` (cuBLAS BF16 tensor cores -> FP32 output with fused ``alpha``):
    measured 1.52 -> 1.00 ms on 8190x8085 on sm_120. Skipped for ``fp32``."""
    if ctx.config.precision.compute_dtype == "fp32":
        return "skipped: fp32"
    model.bf16_similarity = True
    return "coarse matching"


def _merge_linears(mods: tuple[nn.Module, ...]) -> nn.Module | None:
    """Concatenate bias-free linears sharing the same input tensor into one wider linear."""
    from flashim.quant.methods.int8_linear import DynamicInt8Linear, StaticInt8Linear
    from flashim.quant.methods.linear import StaticMXFP8Linear
    from flashim.quant.tensor import QuantizedWeight

    if any(getattr(m, "bias", None) is not None for m in mods):
        return None
    if all(type(m) is nn.Linear for m in mods):
        weight = torch.cat([m.weight.detach() for m in mods], dim=0)
        merged = nn.Linear(weight.shape[1], weight.shape[0], bias=False, device=weight.device, dtype=weight.dtype)
        merged.weight = nn.Parameter(weight, requires_grad=False)
        return merged
    if all(type(m) is DynamicInt8Linear for m in mods):
        return DynamicInt8Linear(
            torch.cat([m.weight_int8 for m in mods], dim=0),
            torch.cat([m.weight_scale for m in mods], dim=0),
            None,
        )
    if all(type(m) is StaticInt8Linear for m in mods):
        s0 = mods[0].input_scale
        if any(m.channel_scale is not None or not torch.equal(m.input_scale, s0) for m in mods):
            return None
        return StaticInt8Linear(
            torch.cat([m.weight_int8 for m in mods], dim=0),
            torch.cat([m.weight_scale for m in mods], dim=0),
            None,
            s0,
        )
    if all(type(m) is StaticMXFP8Linear for m in mods):
        e0 = mods[0].input_exponent
        if any(not torch.equal(m.input_exponent, e0) for m in mods):
            return None
        w_vals = torch.cat([m.weight_values.view(torch.float8_e4m3fn) for m in mods], dim=0)
        w_exps = torch.cat([m.weight_exponent for m in mods], dim=0)
        qw = QuantizedWeight(w_vals, w_exps, (sum(m.out_features for m in mods), mods[0].in_features))
        return StaticMXFP8Linear(qw, None, e0, impl=mods[0].impl)
    return None


def _fused_qkv(model: nn.Module, ctx: BuildContext) -> int | str:
    """Fuse ``q/k/v_proj`` -> ``qkv_proj`` (self) and ``k/v_proj`` -> ``kv_proj`` (cross), plus batch self-attention
    pairs across both images."""
    if ctx.config.precision.compute_dtype == "fp32":
        return "skipped: fp32"
    fused = 0
    for tr in (model.loftr_coarse, model.loftr_fine):
        if not isinstance(tr, LocalFeatureTransformer):
            continue
        tr.batch_self = True
        for layer, name in zip(tr.layers, tr.layer_names):
            if name == "self" and layer.qkv_proj is None:
                merged = _merge_linears((layer.q_proj, layer.k_proj, layer.v_proj))
                if merged is not None:
                    layer.qkv_proj = merged
                    layer.q_proj = layer.k_proj = layer.v_proj = None
                    fused += 1
            elif name == "cross" and layer.kv_proj is None:
                merged = _merge_linears((layer.k_proj, layer.v_proj))
                if merged is not None:
                    layer.kv_proj = merged
                    layer.k_proj = layer.v_proj = None
                    fused += 1
    return fused


def _fused_encoder_epilogues(model: nn.Module, ctx: BuildContext) -> int | str:
    """Fuse ``cat([x, norm1(msg)])`` (+ quantize), ``mlp.0`` ReLU epilogue, and ``x + norm2(out)``."""
    if ctx.config.precision.compute_dtype == "fp32":
        return "skipped: fp32"
    layers = [m for m in model.modules() if isinstance(m, LoFTREncoderLayer)]
    for layer in layers:
        layer.fused_epilogues = True
    return len(layers)


def _fused_fpn_upsample(model: nn.Module, ctx: BuildContext) -> str:
    """Fuse FPN bilinear upsample (``align_corners=True``) + residual add via ``ops.upsample_add``
    (measured 0.482 -> 0.113 ms at 128x360x420 and 0.193 -> 0.020 ms at 256x180x210 on sm_120)."""
    if ctx.config.precision.compute_dtype == "fp32":
        return "skipped: fp32"
    model.backbone.fused_upsample = True
    return "backbone"


def _batch_backbone(model: nn.Module, ctx: BuildContext) -> str:
    """Run ``backbone`` once with ``B=2`` when both prepared images share the same resolution."""
    if ctx.config.precision.compute_dtype == "fp32":
        return "skipped: fp32"
    model.batch_backbone = True
    return "backbone"


def _spec(
    name: str,
    default_variant: str,
    description: str,
    *,
    optimized: bool = False,
    int8_dynamic: tuple[str, ...] = (),
) -> ModelSpec:
    extra_passes = {
        "channels_last": Pass(_channels_last, description="NHWC backbone convolutions (bf16 / fp32)", exact=False),
        "bf16_linear_attention": Pass(
            _bf16_linear_attention, description="linear-attention GEMMs on BF16 tensor cores (not fp32)", exact=False
        ),
        "fold_similarity_temperature": Pass(
            _fold_similarity_temperature, description="coarse-similarity temperature folded into an operand", exact=False
        ),
        "fused_linear_attention": Pass(
            _fused_linear_attention, description="fused Triton ELU+1 linear attention (not fp32)", exact=False
        ),
        "bf16_coarse_similarity": Pass(
            _bf16_coarse_similarity, description="coarse similarity via cuBLAS BF16->FP32 matmul_nt_fp32 (not fp32)", exact=False
        ),
        "fused_qkv": Pass(
            _fused_qkv, description="fused QKV (self) / KV (cross) projections + batched self-attention (not fp32)", exact=False
        ),
        "fused_encoder_epilogues": Pass(
            _fused_encoder_epilogues, description="fused LayerNorm+cat(+quantize), ReLU epilogue, and LayerNorm+add (not fp32)", exact=False
        ),
        "fused_fpn_upsample": Pass(
            _fused_fpn_upsample, description="fused Triton FPN bilinear upsample + add (not fp32)", exact=False
        ),
        "batch_backbone": Pass(
            _batch_backbone, description="batch equal-resolution image pairs through the FPN backbone (not fp32)", exact=False
        ),
    }
    structural: tuple[str, ...] = (
        ("channels_last", "fused_fpn_upsample", "batch_backbone", "fused_linear_attention",
         "fused_qkv", "fused_encoder_epilogues", "bf16_coarse_similarity")
        if optimized
        else ("channels_last", "bf16_linear_attention", "fold_similarity_temperature")
    )
    spec = extractor_spec(
        name=name,
        description=description,
        load=_loader(default_variant),
        calibration_dir=CALIBRATION_DIR,
        quantized=QUANTIZED,
        protected=PROTECTED,
        int8_dynamic=int8_dynamic,
        compile_regions=("backbone",),
        role="dense",
        extra_passes=extra_passes,
        structural_passes=structural,
    )
    presets = spec.presets
    if default_variant != "outdoor":
        presets = {k: {"variant": default_variant, **v} for k, v in presets.items()}
    calibrations = {
        "fp8": CALIBRATION_DIR / ("loftr_fp8.safetensors" if default_variant == "outdoor" else f"loftr_{default_variant}_fp8.safetensors"),
        "int8": CALIBRATION_DIR / ("loftr_int8.safetensors" if default_variant == "outdoor" else f"loftr_{default_variant}_int8.safetensors"),
        "outdoor/fp8": CALIBRATION_DIR / "loftr_fp8.safetensors",
        "outdoor/int8": CALIBRATION_DIR / "loftr_int8.safetensors",
        "minima/fp8": CALIBRATION_DIR / "loftr_minima_fp8.safetensors",
        "minima/int8": CALIBRATION_DIR / "loftr_minima_int8.safetensors",
    }
    engines = {"eager": EngineSpec(), "compile": EngineSpec(_compile_regions, recompile_limit=64)}
    return dataclasses.replace(spec, presets=presets, calibrations=calibrations, engines=engines)


SPEC = _spec("loftr", "outdoor", "LoFTR detector-free matcher (variant outdoor | indoor | minima); match(a, b) only")
SPEC_MINIMA = _spec(
    "minima-loftr",
    "minima",
    "MINIMA-LoFTR modality-invariant detector-free matcher (LoFTR architecture, MINIMA weights); match(a, b) only",
    optimized=True,
    int8_dynamic=TRANSFORMER_LINEARS,
)
