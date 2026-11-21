"""EfficientLoFTR's :class:`~flashim.build.ModelSpec` (presets ``fp32`` / ``bf16`` / ``int8`` / ``fp8``); variant
``full`` (dual-softmax coarse matching, default) or ``opt`` (skip-softmax, faster).

Note on memory layout: ``channels_last`` (NHWC) on ``backbone`` / ``fine_preprocess`` was measured slower than NCHW
on sm_120 in eager BF16 (20.75 vs 19.27 ms/pair for ``full``, 19.22 vs 17.80 ms/pair for ``opt`` at 640x832) due to
1-channel stem / bilinear-upsample / NCHW transformer transitions, so it is intentionally omitted.
"""

from __future__ import annotations

from dataclasses import replace

from torch import nn

from flashim.build import BuildContext, Pass
from flashim.config import RunConfig
from flashim.models.eloftr import quant
from flashim.models.eloftr.config import SETTINGS
from flashim.models.eloftr.model import AggregatedEncoderLayer, ELoFTR
from flashim.models.sparse_spec import extractor_spec


def _load(config: RunConfig) -> ELoFTR:
    return ELoFTR.from_pretrained(
        variant=config.variant or "full",
        checkpoint=config.checkpoint,
        device=config.device,
        **config.arch,
    )


def _validate(config: RunConfig) -> None:
    if config.variant is not None and config.variant not in SETTINGS:
        raise ValueError(f"eloftr: unknown variant {config.variant!r}; choose from {sorted(SETTINGS)}")


def _reparameterize(model: nn.Module, ctx: BuildContext) -> list[str]:
    """Fuse every RepVGG block's 3x3 + 1x1 + identity branches and BatchNorms into one 3x3 conv (exact in real
    arithmetic; upstream's ``reparameter``)."""
    return model.reparameterize()


def _fold_similarity_temperature(model: nn.Module, ctx: BuildContext) -> str:
    """Fold ``/ 0.1`` into the ``[L, C]`` coarse operand instead of an elementwise pass over ``[L, S]``. Skipped for
    ``fp32`` (reference numerics)."""
    if ctx.config.precision.compute_dtype == "fp32":
        return "skipped: fp32"
    model.fold_temperature = True
    return "coarse matching"


def _set_attention(model: nn.Module, impl: str) -> None:
    target = "reference" if impl == "sdpa" else impl
    for module in model.modules():
        if isinstance(module, AggregatedEncoderLayer):
            module.sdpa_impl = target


_BASE = extractor_spec(
    name="eloftr",
    description="EfficientLoFTR detector-free matcher (variant full | opt); match(a, b) only",
    load=_load,
    calibration_dir=quant.CALIBRATION_DIR,
    quantized=quant.QUANTIZED,
    protected=quant.PROTECTED,
    compile_regions=("backbone",),
    role="dense",
    extra_passes={
        "reparameterize": Pass(
            _reparameterize,
            stage="pre_quant",
            description="fuse RepVGG 3x3 + 1x1 + identity branches and BatchNorms into one conv",
        ),
        "fold_similarity_temperature": Pass(
            _fold_similarity_temperature, description="coarse-similarity temperature folded into an operand", exact=False
        ),
    },
    structural_passes=("reparameterize", "fold_similarity_temperature"),
)

SPEC = replace(_BASE, set_attention=_set_attention, validate=_validate)
