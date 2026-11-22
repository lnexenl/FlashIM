"""MASt3R's :class:`~flashim.build.ModelSpec` (presets ``fp32`` / ``bf16`` / ``int8`` / ``fp8``)."""

from __future__ import annotations

from collections.abc import Mapping

import torch
from torch import nn

from flashim.build import BuildContext, EngineSpec, ModelSpec, Pass, dtype
from flashim.config import RunConfig
from flashim.models.mast3r import quant
from flashim.models.mast3r.config import MASt3RConfig
from flashim.models.mast3r.model import MASt3R
from flashim.platforms import Feature
from flashim.quant.calibration import Int8CalibrationTable
from flashim.quant.methods.int8_linear import DynamicInt8Linear
from flashim.quant.passes.transformer_block import enable_fused_epilogues
from flashim.utils import set_submodule


def _load(config: RunConfig) -> MASt3R:
    return MASt3R.from_pretrained(
        MASt3RConfig(**config.arch), checkpoint=config.checkpoint, device=config.device
    )


def _set_precision(model: nn.Module, config: RunConfig) -> None:
    if config.precision.compute_dtype != "fp32":
        model.set_compute_dtype(dtype(config.precision.compute_dtype))


def _apply_inference(model: MASt3R, inference: Mapping) -> None:
    for key in ("subsample", "border", "square_ok"):
        if key in inference:
            setattr(model, key, inference[key])


def _fold_smoothquant(model: nn.Module, ctx: BuildContext) -> int | str:
    """Fold calibrated SmoothQuant scales into preceding LayerNorm and V-projection weights before quantization
    (encoder + decoder for ``int8``, decoder only for hybrid ``fp8``)."""
    qname = ctx.config.precision.quant
    if qname not in ("int8", "fp8"):
        return f"skipped: quant={qname!r}"
    path = ctx.config.precision.calibration if qname == "int8" and ctx.config.precision.calibration else quant.INT8_CALIBRATION
    if not path.is_file():
        return "skipped: missing int8 smooth table"
    table = Int8CalibrationTable.load(path)
    return quant.fold_smoothquant(
        model, table.smooth, include_encoder=(qname == "int8"), include_decoder=True
    )


def _apply_decoder_smooth_fp8(model: nn.Module, ctx: BuildContext) -> int | str:
    """Hybrid ``fp8``: apply the remaining unfolded SmoothQuant factors (``decoder_embed`` and decoder
    ``mlp.fc2``) to the decoder's dynamic INT8 linears."""
    if ctx.config.precision.quant != "fp8":
        return "skipped: not fp8"
    if not quant.INT8_CALIBRATION.is_file():
        return "skipped: missing int8 smooth table"
    table = Int8CalibrationTable.load(quant.INT8_CALIBRATION)
    applied = 0
    for name in ("decoder_embed", *(f"{b}.{i}.mlp.fc2" for b in ("dec_blocks", "dec_blocks2") for i in range(len(model.dec_blocks)))):
        if name not in table.smooth:
            continue
        mod = model.get_submodule(name)
        if not isinstance(mod, nn.Linear):
            continue
        set_submodule(model, name, DynamicInt8Linear.from_linear(mod, smooth=table.smooth[name]))
        applied += 1
    return applied


def _fused_dynamic_vit_blocks(model: nn.Module, ctx: BuildContext) -> list[str] | str:
    """int8: fuse LayerNorm + per-row INT8 quantization, GELU epilogue and residual add into each encoder
    ``TransformerBlock`` (:class:`DynamicInt8TransformerBlock`); bit-identical."""
    if ctx.config.precision.quant != "int8":
        return "skipped: not int8"
    from flashim.quant.passes.dynamic_transformer_block import fuse_dynamic_blocks

    return fuse_dynamic_blocks(model) or "skipped: no dynamic INT8 transformer blocks"


def _fused_dynamic_dec_blocks(model: nn.Module, ctx: BuildContext) -> list[str] | str:
    """int8 / fp8: pack cross-attention K/V projections and fuse LayerNorm+quantize, head-transpose+quantize,
    and residual/GELU epilogues into each ``DecoderBlock``; bit-identical."""
    if ctx.config.precision.quant not in ("int8", "fp8"):
        return f"skipped: quant={ctx.config.precision.quant!r}"
    return quant.fuse_dynamic_decoder_blocks(model) or "skipped: no fusable decoder blocks"


def _fused_head_mlp(model: nn.Module, ctx: BuildContext) -> list[str] | str:
    """fp8: fuse ``fc1`` bias + GELU + ``fc2`` static MXFP8 quantization on the two local-feature MLPs."""
    if ctx.config.precision.quant != "fp8":
        return "skipped: not fp8"
    return quant.fuse_mxfp8_head_mlps(model) or "skipped: no fusable MXFP8 head MLPs"


def _fp16_nn(model: MASt3R, ctx: BuildContext) -> bool | str:
    """Non-fp32 presets: run ``fast_reciprocal_nn`` on FP16 descriptors (10x faster via tensor cores)."""
    if ctx.config.precision.compute_dtype == "fp32":
        return "skipped: fp32"
    model.fp16_nn = True
    return True


def _cuda_graph(model: MASt3R, ctx: BuildContext) -> int | str:
    """Eager engine: replay static-shape forward / descriptor passes from per-shape CUDA graphs (bit-identical)."""
    if ctx.config.engine.kind != "eager":
        return f"skipped: engine {ctx.config.engine.kind}"
    if not ctx.platform.supports(Feature.CUDA_GRAPHS):
        return f"skipped: no CUDA graphs on {ctx.platform.describe()}"
    model.graphs = int(ctx.option("cuda_graphs", 4))
    return model.graphs


def _fused_block_epilogues(model: nn.Module, ctx: BuildContext) -> list[str] | str:
    """fp8: fuse biases and residual adds into GEMM epilogues and quantize kernels on encoder
    ``QuantizedTransformerBlock``s."""
    if ctx.config.precision.quant != "fp8":
        return "skipped: not fp8"
    return enable_fused_epilogues(model) or "skipped: no fusable MXFP8 blocks"


def _fused_rope(model: nn.Module, ctx: BuildContext) -> int | str:
    """Switch every CroCo 2D RoPE call from pinned reference math to ``ops.rope_qk_2d`` dispatch (one fused
    Triton kernel for q and k in self-attention / matched-shape cross-attention; bit-identical on CUDA)."""
    if ctx.config.precision.compute_dtype == "fp32":
        return "skipped: fp32"
    count = 0
    for mod in model.modules():
        if hasattr(mod, "rope_impl") and mod.rope_impl == "reference":
            mod.rope_impl = None
            count += 1
    return count


def _channels_last_dpt(model: nn.Module, ctx: BuildContext) -> list[str] | str:
    """NHWC weights for the two DPT heads in reduced precision so cuDNN avoids NCHW<->NHWC layout conversions."""
    if ctx.config.precision.compute_dtype == "fp32":
        return "skipped: fp32"
    converted = []
    for name in ("downstream_head1", "downstream_head2"):
        head = getattr(model, name)
        head.dpt.to(memory_format=torch.channels_last)
        converted.append(f"{name}.dpt")
    return converted


PASSES = {
    "fold_smoothquant": Pass(
        _fold_smoothquant,
        stage="pre_quant",
        exact=False,
        description="fold SmoothQuant channel scales into LayerNorm and V-projection weights",
    ),
    "apply_decoder_smooth_fp8": Pass(
        _apply_decoder_smooth_fp8,
        stage="pre_quant",
        exact=False,
        description="fp8: pre-pack decoder_embed and decoder mlp.fc2 as smoothed DynamicInt8Linear",
    ),
    "fused_dynamic_vit_blocks": Pass(
        _fused_dynamic_vit_blocks,
        description="int8: encoder ViT LayerNorm+quantize, GELU and residual epilogues",
    ),
    "fused_dynamic_dec_blocks": Pass(
        _fused_dynamic_dec_blocks,
        description="int8/fp8: decoder packed KV, LayerNorm+quantize, head-transpose+quantize and residual epilogues",
    ),
    "fused_head_mlp": Pass(
        _fused_head_mlp,
        exact=False,
        description="fp8: fuse fc1 bias + GELU + fc2 MXFP8 quantization in local-feature MLPs",
    ),
    "fused_block_epilogues": Pass(
        _fused_block_epilogues,
        stage="post_quant",
        exact=False,
        description="fp8: encoder ViT biases / residual adds fused into GEMM epilogues and quantize kernels",
    ),
    "fused_rope": Pass(
        _fused_rope,
        description="CroCo 2D RoPE via ops.rope_qk_2d dispatch (fused Triton kernel)",
    ),
    "channels_last_dpt": Pass(
        _channels_last_dpt,
        exact=False,
        description="NHWC convolutions for the two DPT heads (non-fp32)",
    ),
    "fp16_nn": Pass(
        _fp16_nn,
        exact=False,
        description="FP16 tensor-core descriptors in fast_reciprocal_nn (non-fp32)",
    ),
    "cuda_graph": Pass(
        _cuda_graph,
        description="replay static-shape forward / descriptor passes from per-shape CUDA graphs (eager engine)",
    ),
}

PRESETS = {
    "fp32": {},
    "bf16": {
        "precision": {"compute_dtype": "bf16"},
        "optimize": {"passes": ["fused_rope", "channels_last_dpt", "fp16_nn"]},
    },
    "int8": {
        "precision": {"compute_dtype": "bf16", "quant": "int8"},
        "optimize": {
            "passes": [
                "fold_smoothquant",
                "fused_rope",
                "fused_dynamic_vit_blocks",
                "fused_dynamic_dec_blocks",
                "channels_last_dpt",
                "fp16_nn",
                "cuda_graph",
            ]
        },
    },
    "fp8": {
        "precision": {"compute_dtype": "bf16", "quant": "fp8"},
        "optimize": {
            "passes": [
                "fold_smoothquant",
                "apply_decoder_smooth_fp8",
                "fused_rope",
                "fused_block_epilogues",
                "fused_dynamic_dec_blocks",
                "fused_head_mlp",
                "channels_last_dpt",
                "fp16_nn",
                "cuda_graph",
            ]
        },
    },
}

SPEC = ModelSpec(
    name="mast3r",
    role="dense",
    description="MASt3R 3D-grounded matcher (CroCo ViT-L encoder, ViT-B cross-attention decoder, CatMLP+DPT heads)",
    load=_load,
    presets=PRESETS,
    default_preset="bf16",
    quant_configs={"int8": quant.INT8_CONFIG, "fp8": quant.FP8_CONFIG},
    calibrations={"int8": quant.INT8_CALIBRATION, "fp8": quant.FP8_CALIBRATION},
    protected=quant.PROTECTED,
    passes=PASSES,
    engines={
        "eager": EngineSpec(),
        "compile": EngineSpec(
            lambda m: (
                *(f"enc_blocks.{i}" for i in range(len(m.enc_blocks))),
                *(f"dec_blocks.{i}" for i in range(len(m.dec_blocks))),
                *(f"dec_blocks2.{i}" for i in range(len(m.dec_blocks2))),
                "downstream_head1",
                "downstream_head2",
            ),
            recompile_limit=64,
        ),
    },
    set_precision=_set_precision,
    apply_inference=_apply_inference,
)
