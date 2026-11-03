"""LightGlue quantization presets and protected modules.

Presets:

- ``fp32`` : the checkpoint's precision (matches the reference implementation).
- ``bf16`` : transformer layers and assignment projections in BF16; positions, rotary math, the assignment
  softmaxes and the similarity matrix stay FP32.
- ``fp8``  : ``bf16`` + every transformer linear (QKV / output / FFN, self- and cross-attention) as static MXFP8
  (E4M3 values, E8M0 block-32 scales); attention itself stays BF16.
- ``int8`` : ``bf16`` + the same linears as static symmetric INT8 (per-channel weights, per-tensor activation scale).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
from torch import Tensor, nn

from flashim.quant.rotation import hadamard
from flashim.quant.scheme import INT8_ACTIVATIONS, INT8_WEIGHTS, MXFP8, QuantConfig, QuantScheme

CALIBRATION_DIR = Path(__file__).parent / "calibration"
FP8_CALIBRATION = CALIBRATION_DIR / "lightglue_fp8_mega.safetensors"
INT8_CALIBRATION = CALIBRATION_DIR / "lightglue_int8_mega.safetensors"

# Never quantized: the Fourier position encoding, the assignment heads (similarity + matchability), token
# confidences, and the input projection (``nn.Identity`` for the RDD pairing; a small 128 -> 256 entry
# projection for the ALIKED pairing, same rationale as RDD's own protected `input_proj.*`).
PROTECTED = ("posenc.*", "log_assignment.*", "token_confidence.*", "input_proj.*")

TRANSFORMER_LINEARS = tuple(
    f"transformers.*.{leaf}" for leaf in ("Wqkv", "out_proj", "to_qk", "to_v", "to_out", "ffn.0", "ffn.3")
)


@dataclass(frozen=True)
class Preset:
    name: str
    config: QuantConfig | None
    compute_dtype: torch.dtype
    description: str = ""
    calibration: Path = FP8_CALIBRATION

    @property
    def int8(self) -> bool:
        return self.config is not None and any(
            s.weights.type == "int" for s in self.config.config_groups.values()
        )


PRESETS: dict[str, Preset] = {
    "fp32": Preset("fp32", None, torch.float32, description="checkpoint precision (reference numerics)"),
    "bf16": Preset("bf16", None, torch.bfloat16, description="BF16 transformer, FP32 positions/assignment"),
    "fp8": Preset(
        "fp8",
        QuantConfig({"linear": QuantScheme(TRANSFORMER_LINEARS, MXFP8, MXFP8)}, name="fp8"),
        torch.bfloat16,
        description="bf16 + static MXFP8 transformer linears",
    ),
    "int8": Preset(
        "int8",
        QuantConfig({"linear": QuantScheme(TRANSFORMER_LINEARS, INT8_WEIGHTS, INT8_ACTIVATIONS)}, name="int8"),
        torch.bfloat16,
        description="bf16 + static INT8 transformer linears",
        calibration=INT8_CALIBRATION,
    ),
}


# -- INT8 with a rotated residual stream ---------------------------------------------------------------------------

# Inputs that the rotation cannot reach (behind the FFN's GELU), smoothed instead (SmoothQuant ``alpha``): the per-channel
# ``1 / s`` is applied in the input quantization, ``s`` is folded into the weight columns (calibration tables' ``smooth``).
ROTATED_INT8_SMOOTHING = {"transformers.*.ffn.3": 0.75}


@torch.no_grad()
def rotate_residual_stream(model: nn.Module) -> int:
    """Fold Hadamard rotations into a LightGlue-style matcher's weights; the function is unchanged (up to rounding),
    the inputs of every quantized linear but ``ffn.3`` are rotated. Returns the number of rewritten linears (an identity
    ``input_proj`` replaced included).

    The residual stream has no LayerNorm (it sits inside the FFN), so it can live in the basis ``x H_d``: the writers
    (``input_proj``, ``ffn.3``) produce it, the readers (``Wqkv``, ``to_qk``, ``to_v``, ``ffn.0``'s first half, the
    assignment heads) take ``W H_d``. The attention messages (``out_proj`` / ``to_out`` outputs, ``ffn.0``'s second
    half) are rotated by ``H_d`` too, and the attention outputs per head by ``H_head_dim``, folded into the value rows
    (``softmax(q k^T) v R = (softmax(q k^T) v) R``). An ``nn.Identity`` input projection becomes the ``H_d`` linear
    (protected, BF16).

    Measured with LoMa's INT8 preset (static per-tensor scales; ``results/loma/README.md``): with ``ffn.3`` smoothed
    (``ROTATED_INT8_SMOOTHING``), match agreement with the BF16 matcher rose from 0.92-0.95 to 0.95-0.97 on every
    variant (LoMa-L 0.922 -> 0.957, as the emulation predicted), MegaDepth-1500 AUC@5 vs BF16 from -0.10..-0.70 to
    +0.01..-0.42 (10 RANSAC seeds, paired; each variant +0.11 to +0.46). Per-token dynamic scales without rotation
    changed nothing: the limit is channel outliers (max / median channel amax 5-7, up to 40 on ``ffn.3``). Cost:
    ~1.5 % of the eager LoMa-B / L match time.
    """
    config = model.config
    dim, heads = config.dim, config.num_heads
    param = next(model.transformers.parameters())
    rot = hadamard(dim, device=param.device, dtype=torch.float64)
    per_head = torch.block_diag(*[hadamard(dim // heads, device=param.device, dtype=torch.float64)] * heads)
    rewritten: set[int] = set()

    def update(tensor: Tensor, value: Tensor) -> None:
        tensor.copy_(value.to(tensor.dtype))

    def read(linear: nn.Linear, r: Tensor) -> None:  # input in the rotated basis: W r
        update(linear.weight, linear.weight.double() @ r)
        rewritten.add(id(linear))

    def write(linear: nn.Linear, r: Tensor, rows: slice = slice(None)) -> None:  # output rows in the basis: r^T W
        update(linear.weight[rows], r.T @ linear.weight[rows].double())
        if linear.bias is not None:
            update(linear.bias[rows], r.T @ linear.bias[rows].double())
        rewritten.add(id(linear))

    if isinstance(model.input_proj, nn.Identity):
        model.input_proj = nn.Linear(dim, dim, bias=False, device=param.device, dtype=param.dtype)
        update(model.input_proj.weight, rot.T)
        rewritten.add(id(model.input_proj))
    else:
        write(model.input_proj, rot)
    for layer in model.transformers:
        attn, cross = layer.self_attn, layer.cross_attn
        read(attn.Wqkv, rot)
        write(attn.Wqkv, per_head, slice(2 * dim, 3 * dim))  # rows [q; k; v], heads contiguous in each
        read(attn.out_proj, per_head)
        write(attn.out_proj, rot)
        read(cross.to_qk, rot)
        read(cross.to_v, rot)
        write(cross.to_v, per_head)
        read(cross.to_out, per_head)
        write(cross.to_out, rot)
        for block in (attn, cross):
            read(block.ffn[0], torch.block_diag(rot, rot))  # cat(x, message)
            write(block.ffn[3], rot)
    for head in model.log_assignment:
        read(head.final_proj, rot)
        read(head.matchability, rot)
    for confidence in model.token_confidence:
        read(confidence.token[0], rot)
    return len(rewritten)
