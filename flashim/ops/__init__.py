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
from flashim.ops.attention import deformable_attention, rope_qk, rope_qk_interleaved, sdpa
from flashim.ops.conv import (
    conv2d_at_points,
    conv2d_single_output,
    conv2d_small_cin_quantize_int8,
    depthwise5x5_relu_int8,
    depthwise5x5_relu_mxfp8,
    depthwise5x5_relu_pointwise,
    depthwise5x5_relu_pointwise_int8,
    depthwise5x5_relu_pointwise_mxfp8,
    im2col_gather,
    int8_conv2d,
    int8_conv2d_maxpool,
    mxfp8_conv2d,
)
from flashim.ops.correlation import dual_softmax_mutual, local_corr
from flashim.ops.density import kde_density
from flashim.ops.flexglue import flexglue_entmax_readout
from flashim.ops.gemm import (
    int8_linear,
    int8_linear_quantize,
    linear_fp32,
    matmul_nt_fp32,
    mxfp8_gelu_mlp,
    mxfp8_linear,
    mxfp8_scaled_mm,
)
from flashim.ops.image import jpeg_fancy_upsample
from flashim.ops.lightglue import lightglue_qkv_rope, lightglue_swapped_linear
from flashim.ops.patchify import patch_embed_tokens
from flashim.ops.quantize import (
    bias_relu_quantize_int8,
    gelu_quantize,
    gelu_quantize_int8,
    layernorm_gelu_quantize_int8,
    layernorm_quantize,
    layernorm_quantize_int8,
    layernorm_quantize_int8_rows,
    quantize_int8,
    quantize_int8_heads,
    quantize_int8_multi,
    quantize_int8_rows,
    quantize_mxfp8,
)
from flashim.ops.raco import raco_head5x5, raco_resblock
from flashim.ops.resample import pil_bicubic_resize, upsample_add, upsample_concat, upsample_concat_int8
from flashim.ops.romav2 import romav2_refiner_heads, romav2_refiner_input

__all__ = [
    "OPS",
    "BoundOp",
    "DeviceSpec",
    "Op",
    "bias_relu_quantize_int8",
    "call_counts",
    "configure",
    "conv2d_at_points",
    "conv2d_single_output",
    "conv2d_small_cin_quantize_int8",
    "deform_patches_int8",
    "deformable_attention",
    "depthwise5x5_relu_int8",
    "depthwise5x5_relu_mxfp8",
    "depthwise5x5_relu_pointwise",
    "depthwise5x5_relu_pointwise_int8",
    "depthwise5x5_relu_pointwise_mxfp8",
    "dual_softmax_mutual",
    "flexglue_entmax_readout",
    "gelu_quantize",
    "gelu_quantize_int8",
    "im2col_gather",
    "int8_conv2d",
    "int8_conv2d_maxpool",
    "int8_linear",
    "int8_linear_quantize",
    "jpeg_fancy_upsample",
    "kde_density",
    "layernorm_gelu_quantize_int8",
    "layernorm_quantize",
    "layernorm_quantize_int8",
    "layernorm_quantize_int8_rows",
    "lightglue_qkv_rope",
    "lightglue_swapped_linear",
    "linear_fp32",
    "local_corr",
    "matmul_nt_fp32",
    "multiscale_fuse",
    "mxfp8_conv2d",
    "mxfp8_gelu_mlp",
    "mxfp8_linear",
    "mxfp8_scaled_mm",
    "patch_embed_tokens",
    "pil_bicubic_resize",
    "quantize_int8",
    "quantize_int8_heads",
    "quantize_int8_multi",
    "quantize_int8_rows",
    "quantize_mxfp8",
    "raco_head5x5",
    "raco_resblock",
    "reset_call_counts",
    "romav2_refiner_heads",
    "romav2_refiner_input",
    "rope_qk",
    "rope_qk_interleaved",
    "sdpa",
    "set_strict",
    "upsample_add",
    "upsample_concat",
    "upsample_concat_int8",
]

# Last: backends register their impls into the ops above.
from flashim import backends

backends.load()
