"""FlashIM ops: each has a torch reference plus optional per-platform implementations."""

from flashim.ops._registry import (
    OPS,
    DeviceSpec,
    Op,
    call_counts,
    configure,
    reset_call_counts,
    set_strict,
)
from flashim.ops.local_corr import local_corr
from flashim.ops.mx import (
    depthwise5x5_relu_mxfp8,
    depthwise5x5_relu_pointwise,
    fp8_linear,
    gelu_quantize,
    layernorm_quantize,
    mxfp8_scaled_mm,
)
from flashim.ops.refiner_heads import refiner_heads
from flashim.ops.refiner_input import refiner_input
from flashim.ops.refiner_proj import refiner_proj
from flashim.ops.rope import rope_qk

__all__ = [
    "OPS",
    "DeviceSpec",
    "Op",
    "call_counts",
    "configure",
    "depthwise5x5_relu_mxfp8",
    "depthwise5x5_relu_pointwise",
    "fp8_linear",
    "gelu_quantize",
    "layernorm_quantize",
    "local_corr",
    "mxfp8_scaled_mm",
    "refiner_heads",
    "refiner_input",
    "refiner_proj",
    "reset_call_counts",
    "rope_qk",
    "set_strict",
]
