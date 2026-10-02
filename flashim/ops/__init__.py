"""FlashIM ops: declarations (signature + torch reference); optimized impls live in :mod:`flashim.backends`.

Files group ops by what they compute (``attention``, ``conv``, ``correlation``, ``density``,
``gemm``, ``quantize``); ops only one model uses live in a file named after it (``romav2``, ``aliked``)
and carry its name as a prefix. Each backend package mirrors the same file names.
"""

from flashim.ops._registry import (
    OPS,
    BoundOp,
    DeviceSpec,
    Op,
    call_counts,
    configure,
    reset_call_counts,
    set_strict,
)
from flashim.ops.aliked import deform_patches_int8, multiscale_fuse
from flashim.ops.attention import deformable_attention, rope_qk, sdpa
from flashim.ops.conv import (
    depthwise5x5_relu_int8,
    depthwise5x5_relu_mxfp8,
    depthwise5x5_relu_pointwise,
    depthwise5x5_relu_pointwise_int8,
    depthwise5x5_relu_pointwise_mxfp8,
    im2col_gather,
    int8_conv2d,
    mxfp8_conv2d,
)
from flashim.ops.correlation import local_corr
from flashim.ops.density import kde_density
from flashim.ops.gemm import int8_linear, linear_fp32, mxfp8_gelu_mlp, mxfp8_linear, mxfp8_scaled_mm
from flashim.ops.quantize import (
    gelu_quantize,
    gelu_quantize_int8,
    layernorm_quantize,
    layernorm_quantize_int8,
    quantize_int8,
    quantize_mxfp8,
)
from flashim.ops.raco import raco_head5x5, raco_resblock
from flashim.ops.romav2 import romav2_refiner_heads, romav2_refiner_input

__all__ = [
    "OPS",
    "BoundOp",
    "DeviceSpec",
    "Op",
    "call_counts",
    "configure",
    "deform_patches_int8",
    "deformable_attention",
    "depthwise5x5_relu_int8",
    "depthwise5x5_relu_mxfp8",
    "depthwise5x5_relu_pointwise",
    "depthwise5x5_relu_pointwise_int8",
    "depthwise5x5_relu_pointwise_mxfp8",
    "gelu_quantize",
    "gelu_quantize_int8",
    "im2col_gather",
    "int8_conv2d",
    "int8_linear",
    "kde_density",
    "layernorm_quantize",
    "layernorm_quantize_int8",
    "linear_fp32",
    "local_corr",
    "multiscale_fuse",
    "mxfp8_conv2d",
    "mxfp8_gelu_mlp",
    "mxfp8_linear",
    "mxfp8_scaled_mm",
    "quantize_int8",
    "quantize_mxfp8",
    "raco_head5x5",
    "raco_resblock",
    "reset_call_counts",
    "romav2_refiner_heads",
    "romav2_refiner_input",
    "rope_qk",
    "sdpa",
    "set_strict",
]

# Last: backends register their impls into the ops above.
from flashim import backends

backends.load()
