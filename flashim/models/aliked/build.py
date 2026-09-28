"""Build a ready-to-run ALIKED from a preset, a variant and an engine."""

from __future__ import annotations

from pathlib import Path

import torch

from flashim.engines.compile import compile_regions
from flashim.models.aliked.config import VARIANTS, ALIKEDConfig
from flashim.models.aliked.deform_quant import quantize_deformable_convs
from flashim.models.aliked.fuse import fold_batchnorm
from flashim.models.aliked.model import ALIKED
from flashim.models.aliked.quant import PRESETS
from flashim.quant.calibration import Int8CalibrationTable

ENGINES = ("eager", "compile")
COMPILE_REGIONS = ("block1", "block2", "block3", "block4", "desc_head")


def load_aliked(
    preset: str = "fp32",
    *,
    variant: str = "aliked-n16",
    engine: str = "eager",
    device: torch.device | str | None = None,
    checkpoint: str | Path | None = None,
    config: ALIKEDConfig | None = None,
    compile_mode: str | None = "max-autotune-no-cudagraphs",
    channels_last: bool = False,
    deform_int8_calibration: str | Path | Int8CalibrationTable | None = None,
) -> ALIKED:
    """Load ALIKED weights and apply a precision preset and an execution engine.

    ``variant``: one of ``aliked-t16`` / ``aliked-n16`` / ``aliked-n16rot`` / ``aliked-n32`` — selects the
    encoder width/descriptor-sampling defaults and the checkpoint search key (``config`` overrides them).
    Every preset also folds each block's eval BatchNorm into its convolution (exact, not an approximation;
    measured ~15-30% faster on an RTX 5070 Ti). ``channels_last`` defaults to ``False``: measured *slower*
    here (see :meth:`ALIKED.set_channels_last`) because ALIKED's channel counts are tiny (<=128) and
    ``deform_conv2d`` (block3/4) always returns contiguous NCHW regardless of its input's layout, so there is
    no channels-last algorithm for cuDNN to pick for most of the encoder — kept opt-in for other hardware/variants.
    ``deform_int8_calibration`` (a path or an already-loaded :class:`Int8CalibrationTable`, see
    :func:`flashim.models.aliked.deform_quant.DeformConvCalibrator`) quantizes ``block3``/``block4``'s
    deformable convs to INT8 when given. Worth it for batched workloads (measured 25-39% faster than FP32
    at batch>=4 on an RTX 5070 Ti, growing with batch); at ALIKED's typical single-image ``extract()`` it's
    within noise of FP32 (~2% either way) — see ``quant.py``'s docstring for the full story.
    ``engine="compile"`` compiles the encoder blocks and the descriptor head once per input resolution.
    """
    if preset not in PRESETS:
        raise ValueError(f"Unknown preset {preset!r}; choose from {sorted(PRESETS)}")
    if engine not in ENGINES:
        raise ValueError(f"Unknown engine {engine!r}; choose from {ENGINES}")
    if variant not in VARIANTS:
        raise ValueError(f"Unknown variant {variant!r}; choose from {sorted(VARIANTS)}")
    spec = PRESETS[preset]
    model = ALIKED.from_pretrained(config, variant=variant, checkpoint=checkpoint, device=device)
    model.preset = preset
    model.fused_batchnorm = fold_batchnorm(model)
    if spec.compute_dtype != torch.float32:
        model.set_compute_dtype(spec.compute_dtype)
    if channels_last:
        model.set_channels_last(True)
    if deform_int8_calibration is not None:
        table = deform_int8_calibration
        if not isinstance(table, Int8CalibrationTable):
            table = Int8CalibrationTable.load(table)
        model.quantized_deform_convs = quantize_deformable_convs(model, table)
    if engine == "compile":
        compile_regions(model, COMPILE_REGIONS, mode=compile_mode)
    model.engine = engine
    return model
