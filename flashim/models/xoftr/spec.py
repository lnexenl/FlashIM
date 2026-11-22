"""XoFTR's :class:`~flashim.build.ModelSpec` (presets ``fp32`` / ``bf16`` / ``int8`` / ``fp8``)."""

from __future__ import annotations

from pathlib import Path

import torch
from torch import nn

from flashim.build import BuildContext, Pass
from flashim.config import RunConfig
from flashim.models.loftr.model import LoFTREncoderLayer
from flashim.models.sparse_spec import extractor_spec
from flashim.models.xoftr.model import (
    CrossBidirectionalAttention,
    WindowAttention,
    WindowCrossAttention,
    WindowSelfAttention,
    XoFTR,
)
from flashim.models.xoftr.quant import PROTECTED, QUANTIZED


def _load(config: RunConfig) -> XoFTR:
    return XoFTR.from_pretrained(checkpoint=config.checkpoint, device=config.device, **config.arch)


def _channels_last(model: nn.Module, ctx: BuildContext) -> str:
    """NHWC weights for unquantized cuDNN convolutions. When the backbone is quantized (``int8`` / ``fp8``), still
    converts the unquantized stem, ``layer1.0`` and fine 1x1 convs (measured -0.42 ms on 480x640, and gives
    ``ops.gather_windows`` coalesced channel reads)."""
    if ctx.config.precision.quant:
        model.backbone.conv1.to(memory_format=torch.channels_last)
        model.backbone.layer1[0].to(memory_format=torch.channels_last)
        model.fine_process.out_conv_m.to(memory_format=torch.channels_last)
        model.fine_process.out_conv_f.to(memory_format=torch.channels_last)
        return "stem+layer1.0+fine_out"
    model.backbone.to(memory_format=torch.channels_last)
    model.fine_process.out_conv_m.to(memory_format=torch.channels_last)
    model.fine_process.out_conv_f.to(memory_format=torch.channels_last)
    return "backbone+fine_out"


def _bf16_linear_attention(model: nn.Module, ctx: BuildContext) -> int | str:
    """BF16 operands for the coarse linear-attention GEMMs (skipped for ``fp32`` reference numerics)."""
    if ctx.config.precision.compute_dtype == "fp32":
        return "skipped: fp32"
    layers = [m for m in model.modules() if isinstance(m, LoFTREncoderLayer)]
    for layer in layers:
        layer.attention_gemm_dtype = torch.bfloat16
    return len(layers)


def _fold_similarity_temperature(model: nn.Module, ctx: BuildContext) -> str:
    """Fold ``1 / temperature`` into the left operand of coarse similarity (skipped for ``fp32``)."""
    if ctx.config.precision.compute_dtype == "fp32":
        return "skipped: fp32"
    model.fold_temperature = True
    return "coarse matching"


def _low_precision_similarity(model: nn.Module, ctx: BuildContext) -> str:
    """Coarse similarity via ``ops.matmul_nt_fp32`` on BF16 tensor cores with FP32 output (skipped for ``fp32``)."""
    if ctx.config.precision.compute_dtype == "fp32":
        return "skipped: fp32"
    model.coarse_matching.low_precision_similarity = True
    return "coarse matching"


def _fused_coarse_transformer(model: nn.Module, ctx: BuildContext) -> int:
    """Pack QKV (self) / KV (cross) weights in ``loftr_coarse`` and dispatch through ``ops.elu_linear_attention``."""
    packed = 0
    for layer, name in zip(model.loftr_coarse.layers, model.loftr_coarse.layer_names):
        if name == "self" and all(isinstance(p, nn.Linear) for p in (layer.q_proj, layer.k_proj, layer.v_proj)):
            layer._qkv_w = torch.cat(
                (layer.q_proj.weight.detach(), layer.k_proj.weight.detach(), layer.v_proj.weight.detach()), dim=0
            ).contiguous()
            packed += 1
        elif name == "cross" and all(isinstance(p, nn.Linear) for p in (layer.k_proj, layer.v_proj)):
            layer._kv_w = torch.cat((layer.k_proj.weight.detach(), layer.v_proj.weight.detach()), dim=0).contiguous()
            packed += 1
    model.fused_coarse = True
    return packed


def _pack_window_qkv(model: nn.Module, ctx: BuildContext) -> int:
    """Pack Q+KV (self) and QK+V (cross) weights inside ``fine_process`` window attentions (exact)."""
    count = 0
    for m in model.fine_process.modules():
        if isinstance(m, (WindowAttention, CrossBidirectionalAttention)):
            m.pack_qkv()
            count += 1
    return count


def _fold_window_gamma(model: nn.Module, ctx: BuildContext) -> int | str:
    """Fold ``gamma`` LayerScale parameters into the preceding LayerNorm's affine weight and bias (skipped for ``fp32``)."""
    if ctx.config.precision.compute_dtype == "fp32":
        return "skipped: fp32"
    count = 0
    for m in model.fine_process.modules():
        if isinstance(m, (WindowSelfAttention, WindowCrossAttention)):
            m.fold_gamma()
            count += 1
    return count


SPEC = extractor_spec(
    name="xoftr",
    description="XoFTR cross-modal (visible-thermal) detector-free matcher; match(a, b) only",
    load=_load,
    calibration_dir=Path(__file__).parent / "calibration",
    quantized=QUANTIZED,
    protected=PROTECTED,
    compile_regions=("backbone",),
    role="dense",
    extra_passes={
        "channels_last": Pass(_channels_last, description="NHWC unquantized convolutions", exact=False),
        "bf16_linear_attention": Pass(
            _bf16_linear_attention, description="linear-attention GEMMs on BF16 tensor cores (not fp32)", exact=False
        ),
        "fold_similarity_temperature": Pass(
            _fold_similarity_temperature, description="coarse-similarity temperature folded into an operand", exact=False
        ),
        "low_precision_similarity": Pass(
            _low_precision_similarity,
            description="coarse similarity via ops.matmul_nt_fp32 (BF16 tensor cores, FP32 output)",
            exact=False,
        ),
        "fused_coarse_transformer": Pass(
            _fused_coarse_transformer,
            description="packed QKV/KV + batched self-attn + ops.elu_linear_attention in loftr_coarse",
            exact=False,
        ),
        "pack_window_qkv": Pass(
            _pack_window_qkv, description="pack Q+KV and QK+V projections in fine_process window attentions", exact=True
        ),
        "fold_window_gamma": Pass(
            _fold_window_gamma, description="fold fine_process LayerScale gamma into LayerNorm weight/bias", exact=False
        ),
    },
    structural_passes=(
        "channels_last",
        "bf16_linear_attention",
        "fold_similarity_temperature",
        "low_precision_similarity",
        "fused_coarse_transformer",
        "pack_window_qkv",
        "fold_window_gamma",
    ),
)
