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
# FP8-only ignore: ``conv1b`` stays BF16 in ``fp8`` (``conv1a`` has >1000x channel-magnitude spread and ``conv1b``
# performs 2.26x DC cancellation, so 3-bit E4M3 weight quantization gives 8.77% rel-L2 error and drops isolated
# descriptor cosine to 0.952 vs >=0.991 for every other conv, causing a -0.0099 ScanNet AUC@5 drop on dev; 7-bit
# INT8 weights hold 3.69% rel-L2 and keep ``conv1b`` quantized with its fused pool-only epilogue).
QUANTIZED = ("conv1b", "conv2a", "conv2b", "conv3a", "conv3b", "conv4a", "conv4b", "convPa", "convDa")
PROTECTED = ("conv1a", "convPb", "convDb")
FP8_IGNORE = ("conv1b",)


def _fused_stem_conv(model: SuperPoint, ctx: BuildContext) -> str:
    """Replace cuDNN's slow 1->64 SIMT stem ``conv1a`` (+ bias + ReLU + optional INT8/MXFP8 quantize) with
    Triton ``ops.conv3x3_small_cin_relu`` (exact 9-tap row-major FP32 FMA for Cin=1, bit-identical to cuDNN at
    480x640). Skipped for ``fp32`` (reference numerics)."""
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
    fp8_ignore=FP8_IGNORE,
    compile_regions=("conv1a", "conv1b", "conv2a", "conv2b", "conv3a", "conv3b", "conv4a", "conv4b"),
    has_batchnorm=False,
    channels_last=True,  # BF16 extract at 1200 px, interleaved in one process (sm_120): 7.8 -> 6.7 ms
    cuda_graph=True,
    extra_passes={
        "fused_stem_conv": Pass(
            _fused_stem_conv, exact=False,
            description="bf16/int8/fp8: 1->64 stem conv1a + ReLU (+ quantize) via fused Triton 9-tap FMA",
        ),
        "fused_pool_epilogue": Pass(
            _fused_pool_epilogue,
            description="int8: conv1b/2b/3b + ReLU + MaxPool2d(2) + next conv quantize without unpooled store",
        ),
    },
    structural_passes=("fused_stem_conv", "fused_pool_epilogue"),
)
