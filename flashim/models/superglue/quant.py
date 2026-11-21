"""SuperGlue precision presets and measured quantization evidence.

Presets:

- ``fp32`` : checkpoint precision (exact linear + BatchNorm1d fold in FP32; matches upstream numerics).
- ``bf16`` : GNN layers (18 self/cross attention + MLP blocks) and ``final_proj`` in BF16; keypoint encoder
  (``kenc``), similarity matmul, ``bin_score`` and log-space Sinkhorn optimal transport stay FP32.

``int8`` and ``fp8`` are investigated and not shipped on ``superglue`` (asking for either falls back to ``bf16``
via :data:`flashim.build.NEAREST`). Measured on an RTX 5070 Ti (sm_120) over 20 MegaDepth-1500 pairs
(``benchmarks/accuracy/fidelity.py``, 1600 px, 1024 keypoints, ``kenc.*`` and ``final_proj`` protected; 108 GNN
linears of width 256 / 512 quantized):

- **Matcher-only** (same FP32 SuperPoint features, 1024 x 1024 eager best-of-3):
  - ``fp32`` : 11.59 ms/pair (reference)
  - ``bf16`` : 8.21 ms/pair (3.02 ms GNN + 5.19 ms FP32 Sinkhorn(100)), ``match_agree = 0.9322``, ``match_ratio = 0.9933``
  - static ``int8`` (per-tensor calibrated): 12.16 ms/pair (1.48x slower than ``bf16``), ``match_agree = 0.8374``, ``match_ratio = 0.9689``
  - dynamic ``int8`` (per-token): 12.27 ms/pair (1.49x slower than ``bf16``), ``match_agree = 0.8862``, ``match_ratio = 0.9700``
  - static ``fp8`` (MXFP8): 13.01 ms/pair (1.58x slower than ``bf16``), ``match_agree = 0.7554``, ``match_ratio = 0.9045``
- **End-to-end ``superpoint-superglue``** (``sacre_coeur`` at 1024 px, 816 x 1024 keypoints, eager best-of-3;
  20-pair MegaDepth-1500 fidelity at 1600 px / 1024 keypoints):
  - ``fp32`` (``fp32+fp32``): 31.60 ms/pair (matcher 12.86 ms)
  - ``bf16`` (``bf16+bf16``): 19.31 ms/pair (matcher 8.82 ms), ``kp_overlap = 0.9559``, ``desc_cos = 0.9984``, ``match_agree = 0.8693``
  - ``int8`` (shipped ``int8+bf16`` fallback): 16.87 ms/pair, ``kp_overlap = 0.9386``, ``match_agree = 0.8288``
    (vs experimental ``int8+int8``: 25.70 ms/pair, matcher 17.80 ms, ``match_agree = 0.7658``)
  - ``fp8`` (shipped ``fp8+bf16`` fallback): 16.61 ms/pair, ``kp_overlap = 0.8321``, ``match_agree = 0.7244``
    (vs experimental ``fp8+fp8``: 27.46 ms/pair, matcher 19.45 ms, ``match_agree = 0.6156``)

Quantizing the 108 small 256/512-wide GNN linears is launch-bound in eager mode (slower than ``bf16``, and even
slower than ``fp32`` when image keypoint counts differ) while 63% of the matcher's ``bf16`` time is spent in FP32
Sinkhorn iterations, and both static formats materially degrade match agreement.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

# Protected if a caller runs experimental quantization manually: keypoint coordinate/score MLP and final descriptor
# projection feeding the similarity matrix and Sinkhorn optimal transport.
PROTECTED = ("kenc.*", "final_proj")

GNN_LINEARS = tuple(
    f"gnn.layers.*.{leaf}" for leaf in ("attn.proj.0", "attn.proj.1", "attn.proj.2", "attn.merge", "mlp.0", "mlp.3")
)


@dataclass(frozen=True)
class Preset:
    name: str
    compute_dtype: torch.dtype
    description: str = ""


PRESETS: dict[str, Preset] = {
    "fp32": Preset("fp32", torch.float32, description="checkpoint precision (reference numerics)"),
    "bf16": Preset("bf16", torch.bfloat16, description="BF16 GNN, FP32 keypoint encoder and Sinkhorn"),
}
