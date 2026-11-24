"""SuperGlue quantization presets, Hadamard residual-stream rotation, and protected modules.

Measured outlier diagnosis and quantization evidence (RTX 5070 Ti sm_120, 20 MegaDepth pairs, 1600 px / 1024 kpts,
calibrated on 32 separate MegaDepth training pairs, same FP32 SuperPoint features):

- SuperGlue's 18-layer attentional GNN has no LayerNorm, producing strong fixed-channel outliers in the residual
  stream (``attn.proj`` ch_max/med median 7.6x / max 12.3x; ``mlp.0`` 8.0x / 15.2x with 33.0x weight ratios from
  folded BN1d; ``mlp.3`` behind ReLU 4.5x / 6.4x).
- Unrotated static INT8 drops matcher agreement to 0.8374 (vs BF16 0.9322) and runs slower than BF16 without fused
  glue + CUDA graphs (12.2 ms vs 8.21 ms pre-opt).
- Folding Sylvester Hadamard rotations into the GNN + ``final_proj`` (exact in FP32: 1.0000 agreement) plus
  SmoothQuant ``alpha = 0.75`` on ``mlp.3`` recovers static INT8 agreement to 0.9124 (ratio 0.9872; alpha 0.0:
  0.9028, alpha 0.5: 0.9099), and fused ``int8_glue`` + CUDA graphs brings 1024x1024 latency down to 1.805 ms
  (4.55x faster than pre-opt BF16 8.21 ms, 1.55x faster than optimized BF16 2.806 ms; bit-identical to unfused).
- End-to-end ``benchmarks/accuracy/fidelity.py --models superpoint-superglue --pairs-per-scene 4`` (MegaDepth-1500,
  1600 px / 1024 kpts): ``bf16`` kp_overlap=0.9559 / desc_cos=0.9984 / match_agree=0.8625 / ratio=0.9762;
  ``int8`` kp_overlap=0.9386 / desc_cos=0.9970 / match_agree=0.8135 / ratio=0.9607.
- Rotated dynamic per-token INT8 reaches 0.9300 agreement at 2.013 ms (kept as internal ``int8_dynamic`` config).
- Rotated static MXFP8 (``fp8``) is strictly dominated: slower than INT8 (2.228 ms vs 1.805 ms under CUDA graph)
  and much lower agreement (0.7647-0.7875 vs 0.9124; even excluding ``mlp.3`` gives 0.8009, MLP-only 0.7829,
  layers 0-8 0.8014) due to E4M3's 3-bit mantissa compounding across 18 unnormalized residual layers. ``fp8``
  therefore falls back to ``int8`` via ``build.NEAREST``.
- On the ``superpoint-superglue`` pipeline, pairing ``superpoint fp8`` with ``superglue int8`` dropped 1500-pair
  AUC@5 vs ``bf16`` by -0.0133 on MegaDepth-1500 and -0.0144 on ScanNet-1500 (vs ``int8``: -0.0028 / +0.0068).
  Per-layer SuperPoint diagnosis showed ``conv1b`` (64->64, unnormalized) has 41.6x block-32 activation channel
  outliers (ch_max/med 28.1x) and 10.2x block-32 weight ratios, dropping ``kp_overlap`` to 0.7635 on its own; both
  exact ReLU/MaxPool-commuting SmoothQuant (best kp_overlap 0.8296 at 4.31 ms) and skipping ``conv1b``/``conv2b``
  (0.841-0.892 at 5.59-6.03 ms, slower than BF16 5.59 ms) are strictly dominated by ``superpoint int8`` (0.9101
  kp_overlap, 0.9980 desc_cos, 4.41 ms at 1024 px / 1.99 ms at 640x480). ``superpoint-superglue`` therefore omits
  ``fp8`` as well (degrading ``fp8 -> int8`` via ``build.NEAREST``) without modifying shared ``superpoint``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
from torch import Tensor, nn

from flashim.quant.rotation import hadamard
from flashim.quant.scheme import (
    INT8_ACTIVATIONS,
    INT8_TOKEN_ACTIVATIONS,
    INT8_WEIGHTS,
    MXFP8,
    QuantConfig,
    QuantScheme,
)

CALIBRATION_DIR = Path(__file__).parent / "calibration"

# Never quantized: keypoint coordinate/score MLP and final descriptor projection feeding similarity + Sinkhorn OT.
PROTECTED = ("kenc.*", "final_proj")

# Targets both unpacked ``attn.proj.{0,1,2}`` and packed ``attn.qkv``.
GNN_LINEARS = tuple(
    f"gnn.layers.*.{leaf}"
    for leaf in ("attn.proj.0", "attn.proj.1", "attn.proj.2", "attn.qkv", "attn.merge", "mlp.0", "mlp.3")
)

# SmoothQuant alpha for the one linear behind ReLU that Hadamard rotation cannot reach.
ROTATED_INT8_SMOOTHING = {"gnn.layers.*.mlp.3": 0.75}


@dataclass(frozen=True)
class Preset:
    name: str
    config: QuantConfig | None
    compute_dtype: torch.dtype
    description: str = ""


PRESETS: dict[str, Preset] = {
    "fp32": Preset("fp32", None, torch.float32, description="checkpoint precision (reference numerics)"),
    "bf16": Preset("bf16", None, torch.bfloat16, description="BF16 GNN, FP32 keypoint encoder and Sinkhorn"),
    "int8": Preset(
        "int8",
        QuantConfig({"linear": QuantScheme(GNN_LINEARS, INT8_WEIGHTS, INT8_ACTIVATIONS)}, name="int8"),
        torch.bfloat16,
        description="bf16 + rotated / SmoothQuant static INT8 GNN linears",
    ),
    "int8_dynamic": Preset(
        "int8_dynamic",
        QuantConfig({"linear": QuantScheme(GNN_LINEARS, INT8_WEIGHTS, INT8_TOKEN_ACTIVATIONS)}, name="int8_dynamic"),
        torch.bfloat16,
        description="bf16 + rotated dynamic per-token INT8 GNN linears",
    ),
    "fp8": Preset(
        "fp8",
        QuantConfig({"linear": QuantScheme(GNN_LINEARS, MXFP8, MXFP8)}, name="fp8"),
        torch.bfloat16,
        description="bf16 + rotated static MXFP8 GNN linears",
    ),
}


@torch.no_grad()
def rotate_residual_stream(model: nn.Module) -> int:
    """Fold Sylvester Hadamard rotations into SuperGlue's GNN and ``final_proj`` weights (run after
    ``fold_batchnorm``). Exact in FP32 (1.0000 match agreement on 20 MegaDepth pairs); evens out the 7-15x
    channel outliers of SuperGlue's LayerNorm-free residual stream before INT8 / MXFP8 quantization.

    - Residual stream lives in ``x H_d`` (applied once via ``_rot_in`` on ``desc + kenc`` before layer 0).
    - Attention values and messages per head live in ``v H_head_dim`` (folded into ``proj[2]`` / ``qkv`` V rows and
      ``merge`` input columns); ``merge`` outputs and ``mlp.3`` outputs live in ``H_d``.
    - ``mlp.3`` sits behind ReLU and cannot be rotated; its inputs are smoothed via ``ROTATED_INT8_SMOOTHING``.
    """
    cfg = model.config
    dim, heads = cfg.descriptor_dim, cfg.num_heads
    device = model.device
    rot = hadamard(dim, device=device, dtype=torch.float64)
    per_head = torch.block_diag(*[hadamard(dim // heads, device=device, dtype=torch.float64)] * heads)
    cat_rot = torch.block_diag(rot, rot)
    rewritten: set[int] = set()

    def update(tensor: Tensor, value: Tensor) -> None:
        tensor.copy_(value.to(tensor.dtype))

    def read(linear: nn.Linear, r: Tensor) -> None:
        update(linear.weight, linear.weight.double() @ r)
        rewritten.add(id(linear))

    def write(linear: nn.Linear, r: Tensor, rows: slice = slice(None)) -> None:
        update(linear.weight[rows], r.T @ linear.weight[rows].double())
        if linear.bias is not None:
            update(linear.bias[rows], r.T @ linear.bias[rows].double())
        rewritten.add(id(linear))

    model.register_buffer("_rot_in", rot.to(torch.float32))
    for layer in model.gnn.layers:
        attn = layer.attn
        if attn.qkv is not None:
            read(attn.qkv, rot)
            write(attn.qkv, per_head, slice(2 * dim, 3 * dim))
        else:
            read(attn.proj[0], rot)
            read(attn.proj[1], rot)
            read(attn.proj[2], rot)
            write(attn.proj[2], per_head)
        read(attn.merge, per_head)
        write(attn.merge, rot)
        read(layer.mlp[0], cat_rot)
        write(layer.mlp[3], rot)
    read(model.final_proj, rot)
    return len(rewritten)
