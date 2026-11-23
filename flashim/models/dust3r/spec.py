"""DUSt3R's :class:`~flashim.build.ModelSpec` (presets ``fp32`` / ``bf16`` / ``int8`` / ``fp8``)."""

from __future__ import annotations

from collections.abc import Mapping

import torch
from torch import nn

from flashim import ops
from flashim.build import BuildContext, EngineSpec, ModelSpec, Pass, dtype
from flashim.config import RunConfig
from flashim.models.dust3r import quant
from flashim.models.dust3r.config import DUSt3RConfig
from flashim.models.dust3r.model import DUSt3R
from flashim.models.mast3r.croco import FeatureFusionBlock, Interpolate
from flashim.platforms import Feature


def _load(config: RunConfig) -> DUSt3R:
    return DUSt3R.from_pretrained(
        DUSt3RConfig(**config.arch), checkpoint=config.checkpoint, device=config.device
    )


def _set_precision(model: nn.Module, config: RunConfig) -> None:
    if config.precision.compute_dtype != "fp32":
        model.set_compute_dtype(dtype(config.precision.compute_dtype))


def _apply_inference(model: DUSt3R, inference: Mapping) -> None:
    if "square_ok" in inference:
        model.square_ok = inference["square_ok"]


def _fc2_hadamard(model: nn.Module, ctx: BuildContext) -> dict[str, int] | str:
    """Attach online block-64 Sylvester Hadamard quantization (``ops.rotate_quantize_int8_rows``) to dynamic-INT8
    ``mlp.fc2`` linears (post-GELU channel outliers up to 268x; LayerNorm-folding residual rotation is omitted
    because it hurt 2px agreement on DUSt3R by -0.058)."""
    qname = ctx.config.precision.quant
    if qname not in ("int8", "fp8"):
        return f"skipped: quant={qname!r}"
    fc2_had = quant.attach_fc2_hadamard(model, include_encoder=(qname == "int8"))
    return {"fc2_hadamard": fc2_had}


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


def _cuda_graph(model: DUSt3R, ctx: BuildContext) -> int | str:
    """Eager engine: replay static-shape forward passes from per-shape CUDA graphs (bit-identical)."""
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


class _UpsampleBilinear2x(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return ops.upsample_bilinear2x(x)


class _FastFeatureFusionBlock(nn.Module):
    """Wraps a :class:`FeatureFusionBlock` (sharing its exact submodules) to dispatch 2x bilinear upsampling
    through :data:`flashim.ops.upsample_bilinear2x` (4.4x faster on NHWC FP16 without modifying ``mast3r/croco.py``)."""

    def __init__(self, block: FeatureFusionBlock) -> None:
        super().__init__()
        self.out_conv = block.out_conv
        self.resConfUnit1 = block.resConfUnit1
        self.resConfUnit2 = block.resConfUnit2

    def forward(self, *xs: torch.Tensor) -> torch.Tensor:
        output = xs[0]
        if len(xs) == 2:
            output = output + self.resConfUnit1(xs[1])
        output = self.resConfUnit2(output)
        output = ops.upsample_bilinear2x(output)
        return self.out_conv(output)


def _fast_dpt_upsample(model: nn.Module, ctx: BuildContext) -> int | str:
    """Replace DPT's 2x ``F.interpolate(..., align_corners=True)`` calls with :data:`flashim.ops.upsample_bilinear2x`
    (4.4-4.5x faster on NHWC FP16, saving ~1.1 ms per pair across all reduced-precision presets)."""
    if ctx.config.precision.compute_dtype == "fp32":
        return "skipped: fp32"
    count = 0
    for name in ("downstream_head1", "downstream_head2"):
        dpt = getattr(model, name).dpt
        for rname in ("refinenet1", "refinenet2", "refinenet3", "refinenet4"):
            blk = getattr(dpt.scratch, rname)
            if isinstance(blk, FeatureFusionBlock):
                setattr(dpt.scratch, rname, _FastFeatureFusionBlock(blk))
                count += 1
        if isinstance(dpt.head[1], Interpolate) and dpt.head[1].scale_factor == 2:
            dpt.head[1] = _UpsampleBilinear2x()
            count += 1
    return count


def _fp16_dpt(model: nn.Module, ctx: BuildContext) -> list[str] | str:
    """FP16 NHWC weights for the two DPT 3D-pointmap heads in reduced precision.

    BF16's 7-bit mantissa quantizes 3D coordinates coarser than 1-pixel spacing, collapsing ~35% of reciprocal 3D
    matches (2px agreement 0.091 vs 0.262, 4px 0.656 vs 0.929). FP16's 10-bit mantissa eliminates the staircase
    on tensor cores at +0.2 ms cost (5.1 ms faster than FP32 DPT) with zero non-finite outputs."""
    if ctx.config.precision.compute_dtype == "fp32":
        return "skipped: fp32"
    converted = []
    for name in ("downstream_head1", "downstream_head2"):
        head = getattr(model, name)
        head.dpt.to(dtype=torch.float16, memory_format=torch.channels_last)
        converted.append(f"{name}.dpt")
    return converted


PASSES = {
    "fc2_hadamard": Pass(
        _fc2_hadamard,
        stage="pre_quant",
        exact=False,
        description="attach online block-64 Sylvester Hadamard quantization on dynamic-INT8 mlp.fc2 linears",
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
    "fp16_dpt": Pass(
        _fp16_dpt,
        exact=False,
        description="FP16 NHWC convolutions for the two DPT 3D-pointmap heads (eliminates BF16 coordinate staircase)",
    ),
    "fast_dpt_upsample": Pass(
        _fast_dpt_upsample,
        description="replace DPT 2x bilinear upsampling with NHWC Triton ops.upsample_bilinear2x (4.4x faster)",
    ),
    "cuda_graph": Pass(
        _cuda_graph,
        description="replay static-shape forward passes from per-shape CUDA graphs (eager engine)",
    ),
}

PRESETS = {
    "fp32": {},
    "bf16": {
        "precision": {"compute_dtype": "bf16"},
        "optimize": {"passes": ["fused_rope", "fp16_dpt", "fast_dpt_upsample", "cuda_graph"]},
    },
    "int8": {
        "precision": {"compute_dtype": "bf16", "quant": "int8"},
        "optimize": {
            "passes": [
                "fc2_hadamard",
                "fused_rope",
                "fused_dynamic_vit_blocks",
                "fused_dynamic_dec_blocks",
                "fp16_dpt",
                "fast_dpt_upsample",
                "cuda_graph",
            ]
        },
    },
    "fp8": {
        "precision": {"compute_dtype": "bf16", "quant": "fp8"},
        "optimize": {
            "passes": [
                "fc2_hadamard",
                "fused_rope",
                "fused_dynamic_vit_blocks",
                "fused_dynamic_dec_blocks",
                "fp16_dpt",
                "fast_dpt_upsample",
                "cuda_graph",
            ]
        },
    },
}

SPEC = ModelSpec(
    name="dust3r",
    role="dense",
    description="DUSt3R stereo 3D pointmap matcher (CroCo ViT-L encoder, ViT-B cross-attention decoder, DPT heads)",
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
