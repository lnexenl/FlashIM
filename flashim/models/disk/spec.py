"""DISK's :class:`~flashim.build.ModelSpec` (presets ``fp32`` / ``bf16`` / ``int8`` / ``fp8``)."""

from __future__ import annotations

from pathlib import Path

from flashim.build import BuildContext, Pass
from flashim.config import RunConfig
from flashim.models.disk.model import DISK, Conv, UpBlock
from flashim.models.sparse_spec import extractor_spec

# Quantized: the 5x5 convs of the four down blocks after the input conv and of the first three up blocks. Protected: the
# input conv (K = 75, image-facing) and both halves of the split head conv (descriptors, heatmap logits; strict NMS
# needs the latter at FP32 -- see DISK.split_head_conv).
QUANTIZED = ("unet.path_down.[1234].1.3", "unet.path_up.[012].conv.3")
PROTECTED = ("unet.path_down.0.1.3", "unet.path_up.3.conv.3.desc", "unet.path_up.3.conv.3.heat")


def _split(model: DISK, ctx: BuildContext) -> str:
    return model.split_head_conv()


def _fused_instance_norm_prelu(model: DISK, ctx: BuildContext) -> list[str] | str:
    """Replace ``nn.InstanceNorm2d`` + ``nn.PReLU`` prologues with fused Triton ``ops.instance_norm_prelu``
    (parallel Chan/Welford stats + normalize + PReLU + layout conversion). Skipped for ``fp32`` so the oracle parity
    preset matches PyTorch's ``batch_norm_collect_statistics`` rounding."""
    if ctx.config.precision.compute_dtype == "fp32":
        return "skipped: fp32"
    convs = [name for name, mod in model.named_modules() if isinstance(mod, Conv)]
    for name in convs:
        model.get_submodule(name).fused_prologue = True
    return convs


def _fused_upblock(model: DISK, ctx: BuildContext) -> list[str] | str:
    """Replace ``UpBlock.conv_input`` (``contiguous`` + ``F.interpolate`` + ``torch.cat``) with fused Triton
    ``ops.upsample_concat(..., coarse_first=True)``. Skipped for ``fp32`` (reference numerics)."""
    if ctx.config.precision.compute_dtype == "fp32":
        return "skipped: fp32"
    ups = [name for name, mod in model.named_modules() if isinstance(mod, UpBlock)]
    for name in ups:
        model.get_submodule(name).fused_upsample = True
    return ups


def _load(config: RunConfig) -> DISK:
    return DISK.from_pretrained(checkpoint=config.checkpoint, device=config.device, **config.arch)


SPEC = extractor_spec(
    name="disk",
    description="DISK detector/descriptor (U-Net of 5x5 convs, 128-d descriptors)",
    load=_load,
    calibration_dir=Path(__file__).parent / "calibration",
    quantized=QUANTIZED,
    protected=PROTECTED,
    compile_regions=("unet.path_down.0", "unet.path_down.1", "unet.path_down.2", "unet.path_down.3", "unet.path_down.4",
                     "unet.path_up.0", "unet.path_up.1", "unet.path_up.2"),
    has_batchnorm=False,
    extra_passes={
        # Splits the 129-channel output conv into two smaller ones (desc_dim + 1 -> desc_dim, 1), which alone is
        # measurably faster (DISK bf16 eager 1600px: 67.9ms -> 50.0ms) for the same FP32 precision on both halves --
        # this conv was the single largest kernel in DISK's forward (~70% of total time), running at the input
        # image's full resolution with the U-Net's highest channel count (benchmarks/profile/extractor_profile.py).
        # DISK.fp32_modules keeps both ``desc`` and ``heat`` FP32 for now, matching pre-split precision exactly.
        # Moving ``desc`` onto the model's compute dtype (bf16/lower) like every other conv is a further real
        # speedup (67.9ms -> 43.1ms) but changes descriptor precision that was previously incidentally FP32
        # -protected; an informal 100-pair MegaDepth check showed a ~0.01 AUC@5 drop (single seed, noisy) --
        # accepting it needs the full task-level AUC matrix (benchmarks/accuracy/matrix.sh), not this pass alone.
        "split_head_conv": Pass(
            _split, stage="pre_quant",
            description="head conv split by output channel: FP32 heatmap (1ch) / FP32 descriptors (128ch)",
        ),
        "fused_instance_norm_prelu": Pass(
            _fused_instance_norm_prelu, exact=False,
            description="fused Triton InstanceNorm2d + PReLU (+ layout conversion) for bf16/int8/fp8",
        ),
        "fused_upblock": Pass(
            _fused_upblock, exact=False,
            description="fused Triton upsample_concat in UpBlock for bf16/int8/fp8",
        ),
    },
    structural_passes=("split_head_conv", "fused_instance_norm_prelu", "fused_upblock"),
)
