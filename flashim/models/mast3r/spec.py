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


def _rotate_residual_stream(model: nn.Module, ctx: BuildContext) -> dict[str, int] | str:
    """Fold exact LayerNorm-commuting orthogonal rotations (``Q_1024``, ``Q_768``, per-head ``H_64`` on ``V->proj``)
    into the weights before quantization and mark dynamic-INT8 ``mlp.fc2`` linears for online block-64 Hadamard
    quantization."""
    qname = ctx.config.precision.quant
    if qname not in ("int8", "fp8"):
        return f"skipped: quant={qname!r}"
    rotated = quant.rotate_mast3r(model)
    fc2_had = quant.attach_fc2_hadamard(model, include_encoder=(qname == "int8"))
    return {"rotated_tensors": rotated, "fc2_hadamard": fc2_had}


def _fused_dynamic_vit_blocks(model: nn.Module, ctx: BuildContext) -> list[str] | str:
    """int8 / fp8: fuse LayerNorm + per-row INT8 quantization, GELU epilogue (or hybrid MXFP8 ``gelu_quantize`` on
    ``fc2``) and residual add into each encoder ``TransformerBlock`` (:class:`DynamicInt8TransformerBlock`)."""
    if ctx.config.precision.quant not in ("int8", "fp8"):
        return f"skipped: quant={ctx.config.precision.quant!r}"
    from flashim.quant.passes.dynamic_transformer_block import fuse_dynamic_blocks

    return fuse_dynamic_blocks(model) or "skipped: no dynamic INT8 transformer blocks"


def _fused_dynamic_dec_blocks(model: nn.Module, ctx: BuildContext) -> list[str] | str:
    """int8 / fp8: pack cross-attention K/V projections and fuse LayerNorm+quantize, head-transpose+quantize,
    and residual/GELU epilogues into each ``DecoderBlock``; bit-identical."""
    if ctx.config.precision.quant not in ("int8", "fp8"):
        return f"skipped: quant={ctx.config.precision.quant!r}"
    return quant.fuse_dynamic_decoder_blocks(model) or "skipped: no fusable decoder blocks"


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
    "rotate_residual_stream": Pass(
        _rotate_residual_stream,
        stage="pre_quant",
        exact=False,
        description="fold LayerNorm-commuting orthogonal rotations (Q_1024/Q_768/H_64) and attach online H_64 on fc2",
    ),
    "fused_dynamic_vit_blocks": Pass(
        _fused_dynamic_vit_blocks,
        description="int8/fp8: encoder ViT LayerNorm+quantize, GELU/MXFP8-fc2 and residual epilogues",
    ),
    "fused_dynamic_dec_blocks": Pass(
        _fused_dynamic_dec_blocks,
        description="int8/fp8: decoder packed KV, LayerNorm+quantize, head-transpose+quantize and residual epilogues",
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
        # FP16 (10-bit mantissa) instead of BF16 (7-bit): MASt3R's CroCo residual stream reaches 344.38 at
        # enc_blocks.23 (190x below FP16 max 65504, zero non-finite outputs across 50 MegaDepth pairs), where BF16
        # ULP=2.0 degrades 2px match agreement by -0.209 and 3D point error by 10x vs FP16 at equal latency (quant.py).
        "precision": {"compute_dtype": "fp16"},
        "optimize": {"passes": ["fused_rope", "channels_last_dpt", "fp16_nn"]},
    },
    "int8": {
        "precision": {"compute_dtype": "fp16", "quant": "int8"},
        "optimize": {
            "passes": [
                "rotate_residual_stream",
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
        "precision": {"compute_dtype": "fp16", "quant": "fp8"},
        "optimize": {
            "passes": [
                "rotate_residual_stream",
                "fused_rope",
                "fused_dynamic_vit_blocks",
                "fused_dynamic_dec_blocks",
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
    calibrations={"fp8": quant.FP8_CALIBRATION},
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
