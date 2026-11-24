"""SuperPoint's :class:`~flashim.build.ModelSpec` (presets ``fp32`` / ``bf16`` / ``int8`` / ``fp8``)."""

from __future__ import annotations

from pathlib import Path

from flashim import ops
from flashim.build import BuildContext, Pass
from flashim.config import RunConfig
from flashim.models.sparse_spec import extractor_spec
from flashim.models.superpoint.model import SuperPoint

# Quantized: every 3x3 trunk conv after the 1-channel stem and both 256-wide head convs. Protected (never
# quantized): ``conv1a`` (K = 9, image-facing) and the 65-way score head ``convPb`` (softmax logits) and
# ``convDb`` (final descriptor projection, normalized right after).
QUANTIZED = ("conv1b", "conv2a", "conv2b", "conv3a", "conv3b", "conv4a", "conv4b", "convPa", "convDa")
PROTECTED = ("conv1a", "convPb", "convDb")


def _fused_stem_conv(model: SuperPoint, ctx: BuildContext) -> str:
    """Replace cuDNN's slow 1->64 SIMT stem ``conv1a`` (+ bias + ReLU + optional INT8/MXFP8 quantize) with
    Triton BF16 tensor-core ``ops.conv3x3_small_cin_relu``. Skipped for ``fp32`` (reference numerics)."""
    if ctx.config.precision.compute_dtype == "fp32":
        return "skipped: fp32"
    bound = ops.conv3x3_small_cin_relu.bind(model.conv1a.weight, model.conv1a.bias).warmup()
    if bound.primary == "reference":
        return f"skipped: no kernel on {ctx.platform.describe()}"
    model.fused_stem_op = bound
    return "conv1a"


def _fused_pool_epilogue(model: SuperPoint, ctx: BuildContext) -> list[str] | str:
    """int8: fuse ``conv1b/2b/3b`` + ReLU + ``MaxPool2d(2)`` + next conv's INT8 quantize via
    ``ops.int8_conv2d_maxpool_only`` (no unpooled BF16 store); bit-identical."""
    if ctx.config.precision.quant != "int8":
        return "skipped: not int8"
    model.fused_pool_epilogue = True
    return ["conv1b", "conv2b", "conv3b"]


def _load(config: RunConfig) -> SuperPoint:
    return SuperPoint.from_pretrained(checkpoint=config.checkpoint, device=config.device, **config.arch)


SPEC = extractor_spec(
    name="superpoint",
    description="SuperPoint detector/descriptor (VGG encoder, 256-d descriptors)",
    load=_load,
    calibration_dir=Path(__file__).parent / "calibration",
    quantized=QUANTIZED,
    protected=PROTECTED,
    compile_regions=("conv1a", "conv1b", "conv2a", "conv2b", "conv3a", "conv3b", "conv4a", "conv4b"),
    has_batchnorm=False,
    channels_last=True,  # BF16 extract at 1200 px, interleaved in one process (sm_120): 7.8 -> 6.7 ms
    cuda_graph=True,
    extra_passes={
        "fused_stem_conv": Pass(
            _fused_stem_conv, exact=False,
            description="bf16/int8/fp8: 1->64 stem conv1a + ReLU (+ quantize) on BF16 tensor cores",
        ),
        "fused_pool_epilogue": Pass(
            _fused_pool_epilogue,
            description="int8: conv1b/2b/3b + ReLU + MaxPool2d(2) + next conv quantize without unpooled store",
        ),
    },
    structural_passes=("fused_stem_conv", "fused_pool_epilogue"),
)
