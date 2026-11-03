"""``loma-matcher``: LoMa's matcher spec (variant b128 | b | l | g | r; presets ``fp32`` / ``bf16`` / ``int8`` /
``fp8``).

The quantized presets are LightGlue's (same module tree): ``bf16`` + every transformer linear as static MXFP8 / INT8,
positions, ``input_proj`` and the assignment heads protected; one calibration table per variant and format
(``calibration/loma_<variant>_<fmt>_mega.safetensors``, ``scripts/calibrate_lightglue.py --pipeline loma-<variant>``).
``int8`` also runs the ``rotate_residual_stream`` pass (Hadamard rotations folded into the weights,
:func:`~flashim.models.lightglue.quant.rotate_residual_stream`) and its tables smooth ``ffn.3``'s input
(``ROTATED_INT8_SMOOTHING``), then ``sage_attention`` (SageAttention, QK INT8 / PV FP8, on sm_89 / sm_90 with the
``sageattention`` package), ``fused_int8_glue`` (:mod:`flashim.models.lightglue.int8_glue`, bit-identical) and
``cuda_graph`` (one CUDA graph per keypoint-count pair, bit-identical). Without the graph the INT8 matcher of the
256-dim variants is launch-bound and slower than BF16 (L20 loma-b 12.0 vs 10.8 ms). Match of 4096 x 4096 keypoints,
int8 before -> after (eager, interleaved best of 3): L20 B 12.0 -> 6.8, L 20.1 -> 13.3, G 45.1 -> 31.8 ms; H20 12.3 ->
6.7, 17.3 -> 12.4, 37.7 -> 28.2 ms (bf16: L20 10.8 / 23.1 / 59.5, H20 10.1 / 19.3 / 47.5).
``bf16`` keeps positions, rotary math and the assignment softmaxes in FP32 (upstream's autocast runs the positional
projection in BF16 too). The reduced-precision presets run the ``low_precision_similarity`` pass: the last layer's
similarity on tensor cores from ``final_proj``'s BF16 outputs (FP32 accumulation and output) instead of an FP32 SGEMM.
Measured on loma-l's 4096^2 similarity + dual softmax (CUDA graph): L20 0.75 -> 0.35 ms, H20 0.89 -> 0.36 ms per match;
MegaDepth-1500 (recommended int8 config, 10 RANSAC seeds) AUC@5 paired against the FP32 SGEMM: L20 +0.00, H20 +0.01
(1499 / 1497 of 1500 pairs with identical match sets)."""

from __future__ import annotations

import dataclasses
from pathlib import Path

from flashim.build import EngineSpec, ModelSpec, Pass
from flashim.config import RunConfig
from flashim.models.lightglue import quant as lightglue_quant
from flashim.models.lightglue.spec import _regions, _set_attention, _set_precision
from flashim.models.loma.matcher import VARIANTS, DualSoftmaxAssignment, LoMaMatcher
from flashim.platforms import Feature

CALIBRATION_DIR = Path(__file__).parent / "calibration"
TRANSFORMER_LINEARS = lightglue_quant.TRANSFORMER_LINEARS
PROTECTED = lightglue_quant.PROTECTED
QUANT_CONFIGS = {name: lightglue_quant.PRESETS[name].config for name in ("int8", "fp8")}


def _load(config: RunConfig) -> LoMaMatcher:
    variant = config.variant or "b"
    if variant not in VARIANTS:
        raise ValueError(f"Unknown variant {variant!r}; choose from {sorted(VARIANTS)}")
    arch = dataclasses.replace(VARIANTS[variant], **config.arch)
    return LoMaMatcher.from_variant(variant, arch, checkpoint=config.checkpoint, device=config.device)


def _low_precision_similarity(model: LoMaMatcher, ctx) -> int:
    heads = [m for m in model.modules() if isinstance(m, DualSoftmaxAssignment)]
    for head in heads:
        head.low_precision_similarity = True
    return len(heads)


# SageAttention impl per architecture (``flashim.backends.external.sageattention``; per-thread QK INT8, PV FP8): the sm_89
# kernel on Ada, the WGMMA kernel on Hopper (the sm_89 kernel on an H20 is slower than cuDNN SDPA).
SAGE_IMPLS = {(8, 9): "sage-fp8", (9, 0): "sage-fp8-sm90"}


def _sage_platform(platform) -> bool:
    import importlib.util

    return platform.capability in SAGE_IMPLS and importlib.util.find_spec("sageattention") is not None


def _sage_attention(model: LoMaMatcher, ctx) -> dict | str:
    """int8: every self / cross attention on SageAttention (``SAGE_IMPLS``); ``optimize.attention`` overrides it."""
    if ctx.config.precision.quant != "int8":
        return "skipped: not int8"
    from flashim import ops

    impl = SAGE_IMPLS[ctx.platform.capability]
    if impl not in {i.name for i in ops.sdpa.impls}:
        return f"skipped: {impl} not registered (sageattention without its sm_90 kernels)"
    _set_attention(model, impl)
    return {"impl": impl}


def _cuda_graph(model: LoMaMatcher, ctx) -> int | str:
    """int8, eager engine: the transformer and the assignment head as one CUDA graph per keypoint-count pair
    (:attr:`LoMaMatcher.graphs`); bit-identical to eager. Needs every kernel on the current stream: SageAttention
    as released launches on the legacy default stream, the capture then fails and the matcher runs eagerly (warned)."""
    if ctx.config.precision.quant != "int8":
        return "skipped: not int8"
    if ctx.config.engine.kind != "eager":
        return f"skipped: engine {ctx.config.engine.kind}"
    if not ctx.platform.supports(Feature.CUDA_GRAPHS):
        return f"skipped: no CUDA graphs on {ctx.platform.describe()}"
    model.graphs = int(ctx.option("cuda_graphs", 4))
    return model.graphs


def _fused_int8_glue(model: LoMaMatcher, ctx) -> int | str:
    """int8: the transformer layers' glue fused (:mod:`flashim.models.lightglue.int8_glue`, bit-identical) where every
    op of that path has a kernel for the platform; the layers keep their own forwards otherwise."""
    if ctx.config.precision.quant != "int8":
        return "skipped: not int8"
    from flashim.models.lightglue import int8_glue

    if (reason := int8_glue.unsupported(model)) is not None:
        return f"skipped: {reason}"
    if missing := int8_glue.probe(model):
        return f"skipped: no kernel on {ctx.platform.describe()} for {', '.join(missing)}"
    model.int8_glue = True
    return len(model.transformers)


PASSES = {
    "low_precision_similarity": Pass(
        _low_precision_similarity, exact=False,
        description="the assignment similarity as ops.matmul_nt_fp32 of final_proj's BF16 outputs (tensor cores, "
        "FP32 accumulation and output) instead of an FP32 SGEMM",
    ),
    "sage_attention": Pass(
        _sage_attention, requires=_sage_platform, exact=False,
        description="int8: attention on SageAttention (QK INT8, PV FP8; sm_89 kernel / sm_90 WGMMA kernel)",
    ),
    "fused_int8_glue": Pass(
        _fused_int8_glue, description="int8: one residual stream for both images, quantize / RoPE / swap / LayerNorm "
        "+ GELU fused into the GEMMs and their producers",
    ),
    "cuda_graph": Pass(
        _cuda_graph, description="int8: transformer + assignment head replayed from one CUDA graph per keypoint-count pair",
    ),
    "rotate_residual_stream": Pass(
        lambda model, ctx: lightglue_quant.rotate_residual_stream(model), stage="pre_quant", exact=False,
        description="Hadamard rotations folded into the weights (int8: evens out the quantized inputs' channels)",
    ),
}

SPEC = ModelSpec(
    name="loma-matcher",
    role="matcher",
    description="LoMa matcher (LightGlue-style, DaD keypoints + LoMa descriptors): SparseFeatures x "
    "SparseFeatures -> matches (variant b128 | b | l | g | r)",
    load=_load,
    presets={
        "fp32": {},
        "bf16": {"precision": {"compute_dtype": "bf16"}, "optimize": {"passes": ["low_precision_similarity"]}},
        "fp8": {"extends": "bf16", "precision": {"quant": "fp8"}},
        "int8": {"extends": "bf16", "precision": {"quant": "int8"}, "optimize": {"passes+": ["rotate_residual_stream", "sage_attention", "fused_int8_glue", "cuda_graph"]}},
    },
    default_preset="bf16",
    quant_configs=QUANT_CONFIGS,
    calibrations={  # without a variant: B's
        **{name: CALIBRATION_DIR / f"loma_b_{name}_mega.safetensors" for name in QUANT_CONFIGS},
        **{f"{variant}/{name}": CALIBRATION_DIR / f"loma_{variant}_{name}_mega.safetensors"
           for variant in VARIANTS for name in QUANT_CONFIGS},
    },
    protected=PROTECTED,
    passes=PASSES,
    engines={"eager": EngineSpec(), "compile": EngineSpec(_regions, recompile_limit=64)},
    set_precision=_set_precision,
    set_attention=_set_attention,
)
