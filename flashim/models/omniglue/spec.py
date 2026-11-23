"""OmniGlue specs: ``omniglue-extractor``, ``omniglue-matcher``, and the ``omniglue`` pipeline
(``SparsePipeline(omniglue-extractor, omniglue-matcher)``).

Measured end-to-end latency on RTX 5070 Ti (``sm_120``, ``sacre_coeur{1,2}.jpg``, eager mode, preloaded
GPU tensors):

- ``fp32``: ``90.82 -> 68.71 ms`` (extract ``34.95 -> 27.38 ms``, match ``12.44 -> 7.80 ms``)
- ``bf16``: ``52.35 -> 24.58 ms`` (**2.13x**; extract ``16.47 -> 9.80 ms``, match ``12.11 -> 4.24 ms``)
- ``int8`` (dynamic per-token INT8 ViT + GNN): ``20.66 ms`` (**2.53x** vs BF16 baseline, **4.40x** vs FP32;
  extract ``8.29 ms`` [``sp_v6`` ``4.73``, DINOv2 ``3.59``], match ``3.28 ms``)
- ``fp8`` (calibrated MXFP8 ViT + BF16 ``sp_v6``; matcher degrades to ``int8``): ``23.56 ms``
  (extract ``9.88 ms``, match ``3.32 ms``).
"""

from __future__ import annotations

import dataclasses

import torch
from torch import nn

from flashim.build import PRECISIONS, BuildContext, EngineSpec, ModelSpec, Pass, dtype
from flashim.config import RunConfig
from flashim.models.dedode.spec import PATCH_EMBED_PASS
from flashim.models.lightglue.spec import _compose
from flashim.models.omniglue import quant
from flashim.models.omniglue.config import OmniGlueConfig
from flashim.models.omniglue.extractor import OmniGlueExtractor
from flashim.models.omniglue.model import MultiHeadAttention, OmniGlueMatcher
from flashim.models.sparse_spec import extractor_spec
from flashim.platforms import Feature
from flashim.quant.passes.transformer_block import enable_fused_epilogues, fold_layer_scale

# -- Matcher spec ------------------------------------------------------------------------------------------


def _load_matcher(config: RunConfig) -> OmniGlueMatcher:
    arch = dataclasses.replace(OmniGlueConfig(), **config.arch)
    return OmniGlueMatcher.from_pretrained(arch, checkpoint=config.checkpoint, device=config.device)


def _set_matcher_precision(model: OmniGlueMatcher, config: RunConfig) -> None:
    if config.precision.compute_dtype != "fp32":
        model.set_compute_dtype(dtype(config.precision.compute_dtype))


def _set_matcher_attention(model: nn.Module, impl: str) -> None:
    for module in model.modules():
        if isinstance(module, MultiHeadAttention):
            module.sdpa_impl = impl


def _matcher_fold_bn(model: OmniGlueMatcher, ctx: BuildContext) -> list[str]:
    return model.fold_batchnorm()


def _matcher_pack_qk(model: OmniGlueMatcher, ctx: BuildContext) -> list[str]:
    return model.pack_qk()


def _matcher_low_precision_similarity(model: OmniGlueMatcher, ctx: BuildContext) -> bool | str:
    if ctx.config.precision.compute_dtype == "fp32":
        return "skipped: fp32"
    model.low_precision_similarity = True
    return True


def _matcher_cuda_graph(model: OmniGlueMatcher, ctx: BuildContext) -> int | str:
    if ctx.config.engine.kind != "eager":
        return f"skipped: engine {ctx.config.engine.kind}"
    if not ctx.platform.supports(Feature.CUDA_GRAPHS):
        return f"skipped: no CUDA graphs on {ctx.platform.describe()}"
    model.graphs = int(ctx.option("cuda_graphs", 4))
    return model.graphs


def _matcher_regions(model: OmniGlueMatcher) -> list[str]:
    return ["kenc", *(f"gnn.layers.{i}" for i in range(len(model.gnn.layers)))]


_MATCHER_PASSES = {
    "fold_batchnorm": Pass(_matcher_fold_bn, stage="pre_quant", description="exact Linear+BN1d fold"),
    "pack_qk": Pass(_matcher_pack_qk, stage="pre_quant", description="merge q_proj + k_proj into one qk_proj GEMM"),
    "low_precision_similarity": Pass(
        _matcher_low_precision_similarity,
        exact=False,
        description="similarity via ops.matmul_nt_fp32 on BF16 final_proj outputs (FP32 accumulation)",
    ),
    "cuda_graph": Pass(
        _matcher_cuda_graph,
        description="DINO mask + kenc + 18-layer GNN + Sinkhorn replayed from one CUDA graph per (N, M) shape",
    ),
}

_MATCHER_PRESETS: dict[str, dict] = {
    "fp32": {"optimize": {"passes": ["fold_batchnorm", "pack_qk"]}},
    "bf16": {
        "extends": "fp32",
        "precision": {"compute_dtype": "bf16"},
        "optimize": {"passes+": ["low_precision_similarity", "cuda_graph"]},
    },
    "int8": {
        "extends": "bf16",
        "precision": {"quant": "int8"},
    },
}

MATCHER_SPEC = ModelSpec(
    name="omniglue-matcher",
    role="matcher",
    description="OmniGlue matcher: OmniGlueFeatures x OmniGlueFeatures -> matches (9 self + 9 DINO-masked cross + Sinkhorn)",
    load=_load_matcher,
    presets=_MATCHER_PRESETS,
    default_preset="bf16",
    quant_configs=quant.MATCHER_QUANT_CONFIGS,
    calibrations={},  # dynamic per-token INT8 needs no calibration table
    protected=quant.MATCHER_PROTECTED,
    passes=_MATCHER_PASSES,
    engines={"eager": EngineSpec(), "compile": EngineSpec(_matcher_regions, recompile_limit=64)},
    set_precision=_set_matcher_precision,
    set_attention=_set_matcher_attention,
)

# -- Extractor spec (SuperPoint sp_v6 + DINOv2 ViT-B/14) ---------------------------------------------------


def _load_extractor(config: RunConfig) -> OmniGlueExtractor:
    arch_kwargs = dict(config.arch)
    sp_variant = arch_kwargs.pop("sp_variant", "sp_v6")
    arch = dataclasses.replace(OmniGlueConfig(), **arch_kwargs)
    return OmniGlueExtractor.from_pretrained(arch, sp_variant=sp_variant, device=config.device)


def _sp_channels_last(model: OmniGlueExtractor, ctx: BuildContext) -> str:
    if ctx.config.precision.compute_dtype == "fp32":
        return "skipped: fp32"
    model.superpoint.to(memory_format=torch.channels_last)
    return "superpoint"


def _sp_fused_relu_bn(model: OmniGlueExtractor, ctx: BuildContext) -> list[str] | str:
    if ctx.config.precision.compute_dtype == "fp32":
        return "skipped: fp32"
    return model.superpoint.fold_relu_batchnorm() or "skipped: no sp_v6 batchnorm"


def _fold_vit_layer_scale(model: nn.Module, ctx: BuildContext) -> list[str] | str:
    q = ctx.config.precision.quant
    if not q or ctx.spec.quant_configs[q].scheme_for("dinov2.blocks.0.attn.proj") is None:
        return "skipped: ViT not quantized"
    return fold_layer_scale(model)


def _fused_dynamic_vit(model: nn.Module, ctx: BuildContext) -> list[str] | str:
    if ctx.config.precision.quant != "int8":
        return "skipped: not int8"
    from flashim.quant.passes.dynamic_transformer_block import fuse_dynamic_blocks

    return fuse_dynamic_blocks(model) or "skipped: no dynamic INT8 transformer blocks"


def _fused_fp8_vit(model: nn.Module, ctx: BuildContext) -> list[str] | str:
    return enable_fused_epilogues(model) if ctx.config.precision.quant == "fp8" else "skipped: not fp8"


_EXTRACTOR_EXTRA_PASSES = {
    "sp_channels_last": Pass(
        _sp_channels_last,
        exact=False,
        description="SuperPoint sp_v6 conv weights in NHWC channels-last (saves ~2.7 ms at 1600 px on sm_120)",
    ),
    "fused_sp_relu_bn": Pass(
        _sp_fused_relu_bn,
        description="SuperPoint sp_v6 ReLU + BatchNorm2d fused into one ops.relu_affine Triton pass",
    ),
    "fold_layer_scale": Pass(
        _fold_vit_layer_scale,
        stage="pre_quant",
        exact=False,
        description="ViT LayerScale folded into proj / fc2 (when quantized)",
    ),
    "fused_dynamic_vit_blocks": Pass(
        _fused_dynamic_vit,
        description="int8: ViT LayerNorm + row quantize, GELU and residual in one kernel each",
    ),
    "fused_block_epilogues": Pass(
        _fused_fp8_vit,
        stage="post_quant",
        exact=False,
        description="ViT biases / residual adds fused into GEMM epilogues and quantize kernels (fp8)",
    ),
    **PATCH_EMBED_PASS,
}

_EXTRACTOR_BASE = extractor_spec(
    name="omniglue-extractor",
    description="OmniGlue extractor: SuperPoint sp_v6 keypoints/descriptors + DINOv2 ViT-B/14 guidance features",
    load=_load_extractor,
    calibration_dir=quant.CALIBRATION_DIR,
    quantized=quant.VIT_LINEARS,
    protected=quant.EXTRACTOR_PROTECTED,
    int8_dynamic=quant.VIT_LINEARS,
    compile_regions=("superpoint", "dinov2"),
    has_batchnorm=True,
    channels_last=False,  # only superpoint uses channels_last (via sp_channels_last); dinov2 is NLC
    extra_passes=_EXTRACTOR_EXTRA_PASSES,
    structural_passes=tuple(_EXTRACTOR_EXTRA_PASSES),
)

# int8 needs no calibration table (all quantized targets use per-token dynamic INT8); fp8 needs its table.
_EXTRACTOR_PRESETS = {
    k: v for k, v in _EXTRACTOR_BASE.presets.items()
    if k in ("fp32", "bf16", "int8") or (quant.CALIBRATION_DIR / f"omniglue-extractor_{k}.safetensors").exists()
}
EXTRACTOR_SPEC = dataclasses.replace(
    _EXTRACTOR_BASE,
    presets=_EXTRACTOR_PRESETS,
    quant_configs=quant.EXTRACTOR_QUANT_CONFIGS,
    calibrations={"fp8": quant.CALIBRATION_DIR / "omniglue-extractor_fp8.safetensors"},
)

# -- Pipeline spec -----------------------------------------------------------------------------------------

SPEC = ModelSpec(
    name="omniglue",
    role="pipeline",
    description="OmniGlue: SuperPoint sp_v6 + DINOv2 ViT-B/14 + OmniGlue matcher (CVPR 2024)",
    load=None,
    presets={
        preset: {
            "components": {
                "extractor": {"model": "omniglue-extractor", "preset": preset},
                "matcher": {"model": "omniglue-matcher", "preset": preset},
            }
        }
        for preset in PRECISIONS
    },
    default_preset="bf16",
    compose=_compose,
)
