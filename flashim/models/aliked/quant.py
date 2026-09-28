"""ALIKED precision presets.

- ``fp32``: the checkpoint's own precision (matches the reference implementation).
- ``bf16``: the encoder (``block1..4``), projections and the descriptor head (``SDDH``) in BF16; the score
  head and all NMS/soft-argmax/sampling math stay FP32.

No quantized preset by default: investigated and not shipped as one. ALIKED has no ``nn.Linear`` layers,
and the existing quantization infrastructure otherwise only targets 1x1 ``nn.Conv2d`` (INT8); ALIKED's 1x1
convs (``conv1..4``, the ``downsample``s, ``desc_head.sf_conv``) are all K/N <= 128 — RDD's ``fp8-backbone``
preset already measured that class of GEMM (small-K, memory-bound) *slower* to quantize than to leave in
BF16 (see ``flashim/models/rdd/quant.py``); ALIKED's are smaller still, so this wasn't re-measured.

``deform_quant.QuantizedDeformableConv2d`` *was* built and measured: an INT8 GEMM over the deformable
convs' gathered receptive-field patches (``ops.deform_patches_int8``, a fused Triton gather+quantize
kernel feeding the existing INT8 GEMM). Correctness is good (feature cosine > 0.999 vs FP32) and it is
fully calibratable (``deform_quant.DeformConvCalibrator``).

First cut was 10-35% *slower* than FP32 at every batch size — profiling (``torch.profiler``, not
assumption) showed the INT8 GEMM itself was cheap (~7us) and the *gather* kernel was the entire problem:
4x slower than torchvision's own gather kernel, because ``x`` is NCHW, so gathering one pixel's channels
strides across ``H*W``-sized planes (no coalescing at all). Converting ``x`` to channels-last *inside the
Triton kernel* (so a pixel's channels are contiguous) fixed most of it: on an RTX 5070 Ti, ``batch=1`` is
now ~20% slower (was 35%; whole-model ``extract()`` difference is noise-level, ~2%), and it's a clear
**win** at ``batch>=4`` (25-39% faster, growing with batch — FP32's `deform_conv2d` scales worse per-image
than the now-coalesced INT8 path). Kept opt-in (``load_aliked``'s ``deform_int8_calibration``): worth it for
batched workloads, not for ALIKED's typical single-image ``extract()``.

See ``fuse.fold_batchnorm`` and ``build.load_aliked`` for the optimizations that *did* measure faster
(BatchNorm folding: ~30% faster; ``channels_last``: measured slower, kept opt-in only).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class Preset:
    name: str
    compute_dtype: torch.dtype
    description: str = ""


PRESETS: dict[str, Preset] = {
    "fp32": Preset("fp32", torch.float32, description="checkpoint precision (reference numerics)"),
    "bf16": Preset("bf16", torch.bfloat16, description="BF16 encoder/descriptor head, FP32 score head"),
}
