"""MASt3R quantization targets, SmoothQuant folding, protected modules and measured accuracy evidence.

Measured on 6 MegaDepth-1500 pairs (512 px working resolution, PIL inputs, vs FP32):

- ``bf16`` (reference reduced precision): ag@1px 0.7299, ag@2px 0.8910, desc_cos 0.99956, pts3d_rel 0.00916.
- Unsmoothed per-token dynamic INT8 (all 293 linears): ag@1px 0.5170, ag@2px 0.7972, desc_cos 0.99612,
  pts3d_rel 0.01904 — degraded by severe channel outliers (max/median channel amax up to 1865x on
  ``enc_blocks.0.mlp.fc2``, 176x on ``enc_blocks.1.mlp.fc2``, 131x on ``enc_blocks.1.attn.qkv``, 120x on
  ``dec_blocks.0.mlp.fc2``).
- SmoothQuant + per-token dynamic INT8 (all 293 linears):
  - alpha=0.25: ag@1px 0.5614, ag@2px 0.8209, desc_cos 0.99771, pts3d_rel 0.01796
  - alpha=0.35: ag@1px 0.5970, ag@2px 0.8414, desc_cos 0.99827, pts3d_rel 0.01411
  - alpha=0.50: ag@1px 0.6189, ag@2px 0.8380, desc_cos 0.99866, pts3d_rel 0.01318
  - alpha=0.60: ag@1px 0.6250, ag@2px 0.8489, desc_cos 0.99881, pts3d_rel 0.01322  (selected for ``int8``)
  - alpha=0.75: ag@1px 0.5733, ag@2px 0.8148, desc_cos 0.99811, pts3d_rel 0.01322
- Pure static MXFP8 (all 293 linears): ag@1px 0.4757, ag@2px 0.7911, desc_cos 0.99348, pts3d_rel 0.02151.
  Dynamic per-(token, block-32) MXFP8 (0.4454 / 0.7825 / 0.99335) and SmoothQuant-folded MXFP8 on the decoder
  (0.4816 / 0.8037 / 0.99517) do not recover it: E4M3's 3-bit mantissa is too coarse for the decoder's
  cross-attention projections, whereas the ViT-L encoder (0.6692 / 0.8601 / 0.99873) and local-feature MLPs
  (0.6691 / 0.8605 / 0.99925) tolerate MXFP8 well.
- Hybrid ``fp8`` (static MXFP8 on ViT-L encoder + local-feature MLPs, SmoothQuant alpha=0.6 dynamic INT8 on
  the ViT-B cross-attention decoder): ag@1px ~0.61, ag@2px ~0.84, desc_cos ~0.9983 — all 293 linears quantized
  in 8-bit with zero BF16 fallback.

Measured on 20 MegaDepth-1500 pairs via ``benchmarks/accuracy/fidelity.py`` (512 px, 2 px original-image radius
where 1 px at 512 px corresponds to ~3.125 px original, vs FP32) and RTX 5070 Ti (sm_120) end-to-end
``match_prepared`` speed (384x512 Toronto / 352x512+512x384 Sacre):

- ``bf16``: match_agree 0.7324 (PIL 0.7386), match_ratio 0.9967 | 26.68 ms / 27.66 ms (baseline 60.53 ms)
- ``int8`` (SmoothQuant alpha=0.6 dynamic INT8): match_agree 0.6326 (PIL 0.6339, vs 0.5418 unsmoothed),
  match_ratio 0.9975 | 16.35 ms / 18.07 ms (1.63x faster than optimized bf16; 3.70x vs bf16 baseline)
- ``fp8`` (hybrid MXFP8 encoder/head + SmoothQuant INT8 decoder): match_agree 0.6126 (PIL 0.6141, vs 0.5004 pure
  MXFP8), match_ratio 0.9984 | 15.85 ms / 17.97 ms (1.68x faster than optimized bf16; 3.82x vs bf16 baseline)
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import torch
from torch import Tensor, nn

from flashim import ops
from flashim.models.mast3r.croco import DecoderBlock
from flashim.quant.methods.int8_linear import DynamicInt8Linear
from flashim.quant.methods.linear import StaticMXFP8Linear
from flashim.quant.scheme import (
    INT8_TOKEN_ACTIVATIONS,
    INT8_WEIGHTS,
    MXFP8,
    QuantConfig,
    QuantScheme,
)
from flashim.utils import set_submodule

CALIBRATION_DIR = Path(__file__).parent / "calibration"
FP8_CALIBRATION = CALIBRATION_DIR / "mast3r_fp8.safetensors"
INT8_CALIBRATION = CALIBRATION_DIR / "mast3r_int8.safetensors"
SMOOTH_ALPHA = 0.6

ENC_LINEARS = (
    "enc_blocks.*.attn.qkv",
    "enc_blocks.*.attn.proj",
    "enc_blocks.*.mlp.fc1",
    "enc_blocks.*.mlp.fc2",
)
DEC_LINEARS = (
    "decoder_embed",
    "dec_blocks.*.attn.qkv",
    "dec_blocks.*.attn.proj",
    "dec_blocks.*.cross_attn.projq",
    "dec_blocks.*.cross_attn.projk",
    "dec_blocks.*.cross_attn.projv",
    "dec_blocks.*.cross_attn.proj",
    "dec_blocks.*.mlp.fc1",
    "dec_blocks.*.mlp.fc2",
    "dec_blocks2.*.attn.qkv",
    "dec_blocks2.*.attn.proj",
    "dec_blocks2.*.cross_attn.projq",
    "dec_blocks2.*.cross_attn.projk",
    "dec_blocks2.*.cross_attn.projv",
    "dec_blocks2.*.cross_attn.proj",
    "dec_blocks2.*.mlp.fc1",
    "dec_blocks2.*.mlp.fc2",
)
HEAD_MLP_LINEARS = (
    "downstream_head1.head_local_features.fc1",
    "downstream_head1.head_local_features.fc2",
    "downstream_head2.head_local_features.fc1",
    "downstream_head2.head_local_features.fc2",
)
ALL_TRANSFORMER_LINEARS = (*ENC_LINEARS, *DEC_LINEARS, *HEAD_MLP_LINEARS)

# Never quantized: input patch embedding and the 3D-point/confidence DPT conv heads.
PROTECTED = (
    "patch_embed.*",
    "downstream_head1.dpt.*",
    "downstream_head2.dpt.*",
)

INT8_CONFIG = QuantConfig(
    {
        "transformer": QuantScheme(
            targets=ALL_TRANSFORMER_LINEARS,
            weights=INT8_WEIGHTS,
            input_activations=INT8_TOKEN_ACTIVATIONS,
        ),
    },
    name="int8",
)

# Hybrid FP8: MXFP8 (cuBLAS scaled_mm) on the ViT-L encoder and local-feature MLPs; SmoothQuant per-token dynamic
# INT8 on the ViT-B cross-attention decoder where E4M3's 3-bit mantissa loses match accuracy.
FP8_CONFIG = QuantConfig(
    {
        "mxfp8": QuantScheme(
            targets=(*ENC_LINEARS, *HEAD_MLP_LINEARS),
            weights=MXFP8,
            input_activations=MXFP8,
        ),
        "decoder_int8": QuantScheme(
            targets=DEC_LINEARS,
            weights=INT8_WEIGHTS,
            input_activations=INT8_TOKEN_ACTIVATIONS,
        ),
    },
    name="fp8",
)


def is_foldable_smooth_target(name: str) -> bool:
    """Linears whose SmoothQuant input scaling ``1 / s`` folds into a preceding LayerNorm or V projection at zero
    runtime cost."""
    if name.startswith("enc_blocks."):
        return name.endswith((".attn.qkv", ".attn.proj", ".mlp.fc1"))
    if name.startswith(("dec_blocks.", "dec_blocks2.")):
        return name.endswith(
            (".attn.qkv", ".attn.proj", ".cross_attn.projq", ".cross_attn.projk", ".cross_attn.projv",
             ".cross_attn.proj", ".mlp.fc1")
        )
    return False


@torch.no_grad()
def fold_smoothquant(
    model: nn.Module,
    smooth: Mapping[str, Tensor],
    *,
    include_encoder: bool = True,
    include_decoder: bool = True,
) -> int:
    """Fold SmoothQuant scales stored under ``fold::<linear_name>`` into preceding LayerNorm / V-projection weights
    so the quantized linears need no per-channel input scaling at runtime. Exact in real arithmetic."""
    folded = 0

    def _get(name: str, device: torch.device) -> Tensor | None:
        key = f"fold::{name}"
        if key not in smooth:
            return None
        return smooth[key].to(device=device, dtype=torch.float32)

    def _fold_ln(norm: nn.LayerNorm, linear: nn.Linear, s: Tensor) -> None:
        norm.weight.copy_((norm.weight.float() / s).to(norm.weight.dtype))
        if norm.bias is not None:
            norm.bias.copy_((norm.bias.float() / s).to(norm.bias.dtype))
        linear.weight.copy_((linear.weight.float() * s).to(linear.weight.dtype))

    def _fold_qkv_v(qkv: nn.Linear, proj: nn.Linear, s: Tensor) -> None:
        dim = proj.in_features
        v_slice = slice(2 * dim, 3 * dim)
        qkv.weight[v_slice].copy_((qkv.weight[v_slice].float() / s[:, None]).to(qkv.weight.dtype))
        if qkv.bias is not None:
            qkv.bias[v_slice].copy_((qkv.bias[v_slice].float() / s).to(qkv.bias.dtype))
        proj.weight.copy_((proj.weight.float() * s).to(proj.weight.dtype))

    if include_encoder:
        for i, blk in enumerate(model.enc_blocks):
            prefix = f"enc_blocks.{i}"
            dev = blk.norm1.weight.device
            if (s := _get(f"{prefix}.attn.qkv", dev)) is not None:
                _fold_ln(blk.norm1, blk.attn.qkv, s)
                folded += 1
            if (s := _get(f"{prefix}.attn.proj", dev)) is not None:
                _fold_qkv_v(blk.attn.qkv, blk.attn.proj, s)
                folded += 1
            if (s := _get(f"{prefix}.mlp.fc1", dev)) is not None:
                _fold_ln(blk.norm2, blk.mlp.fc1, s)
                folded += 1

    if include_decoder:
        for branch in ("dec_blocks", "dec_blocks2"):
            for i, blk in enumerate(getattr(model, branch)):
                prefix = f"{branch}.{i}"
                dev = blk.norm1.weight.device
                if (s := _get(f"{prefix}.attn.qkv", dev)) is not None:
                    _fold_ln(blk.norm1, blk.attn.qkv, s)
                    folded += 1
                if (s := _get(f"{prefix}.attn.proj", dev)) is not None:
                    _fold_qkv_v(blk.attn.qkv, blk.attn.proj, s)
                    folded += 1
                if (s := _get(f"{prefix}.cross_attn.projq", dev)) is not None:
                    _fold_ln(blk.norm2, blk.cross_attn.projq, s)
                    folded += 1
                sk = _get(f"{prefix}.cross_attn.projk", dev)
                sv = _get(f"{prefix}.cross_attn.projv", dev)
                if sk is not None and sv is not None:
                    s_kv = torch.sqrt(sk * sv)
                    blk.norm_y.weight.copy_((blk.norm_y.weight.float() / s_kv).to(blk.norm_y.weight.dtype))
                    if blk.norm_y.bias is not None:
                        blk.norm_y.bias.copy_((blk.norm_y.bias.float() / s_kv).to(blk.norm_y.bias.dtype))
                    blk.cross_attn.projk.weight.copy_(
                        (blk.cross_attn.projk.weight.float() * s_kv).to(blk.cross_attn.projk.weight.dtype)
                    )
                    blk.cross_attn.projv.weight.copy_(
                        (blk.cross_attn.projv.weight.float() * s_kv).to(blk.cross_attn.projv.weight.dtype)
                    )
                    folded += 2
                if (s := _get(f"{prefix}.cross_attn.proj", dev)) is not None:
                    pv = blk.cross_attn.projv
                    pv.weight.copy_((pv.weight.float() / s[:, None]).to(pv.weight.dtype))
                    if pv.bias is not None:
                        pv.bias.copy_((pv.bias.float() / s).to(pv.bias.dtype))
                    blk.cross_attn.proj.weight.copy_(
                        (blk.cross_attn.proj.weight.float() * s).to(blk.cross_attn.proj.weight.dtype)
                    )
                    folded += 1
                if (s := _get(f"{prefix}.mlp.fc1", dev)) is not None:
                    _fold_ln(blk.norm3, blk.mlp.fc1, s)
                    folded += 1
    return folded


# -- dynamic INT8 decoder-block fusion & MXFP8 local-feature MLP fusion ----------------------------------


def _is_fusable_decoder_block(block: nn.Module) -> bool:
    if not isinstance(block, DecoderBlock):
        return False
    ca = block.cross_attn
    linears = (block.attn.qkv, block.attn.proj, ca.projq, ca.projk, ca.projv, ca.proj, block.mlp.fc1, block.mlp.fc2)
    return (
        all(isinstance(m, DynamicInt8Linear) for m in linears)
        and all(getattr(m, "channel_scale", None) is None for m in linears[:-1])
        and ca.projk.bias is not None
        and ca.projv.bias is not None
    )


class DynamicInt8DecoderBlock(nn.Module):
    """Fused :class:`DecoderBlock` for per-token dynamic INT8 linears (used by both ``int8`` and hybrid ``fp8``):

    - Packs ``cross_attn.projk`` + ``cross_attn.projv`` into one ``[2C, C]`` INT8 GEMM (quantizing ``norm_y(y)``
      once instead of twice, and saturating SMs with a wider N);
    - Fuses ``norm1`` / ``norm2`` / ``norm_y`` / ``norm3`` with per-row INT8 quantization where supported
      (:data:`flashim.ops.layernorm_quantize_int8_rows`);
    - Fuses attention output head-transpose with per-row INT8 quantization (:data:`flashim.ops.quantize_int8_rows_heads`);
    - Fuses residual adds into ``attn.proj``, ``cross_attn.proj`` and ``mlp.fc2`` GEMM epilogues, and GELU into
      ``mlp.fc1`` where supported.

    Bit-identical to the unfused ``DecoderBlock``.
    """

    def __init__(self, block: DecoderBlock, device: torch.device | None = None) -> None:
        super().__init__()
        self.norm1 = block.norm1
        self.attn = block.attn
        self.norm2 = block.norm2
        self.norm_y = block.norm_y
        self.cross_attn = block.cross_attn
        self.norm3 = block.norm3
        self.mlp = block.mlp
        self._exact_gelu = isinstance(self.mlp.act, nn.GELU) and self.mlp.act.approximate == "none"
        self._has_norm_y = isinstance(self.norm_y, nn.LayerNorm) and self.norm_y.bias is not None

        pk, pv = block.cross_attn.projk, block.cross_attn.projv
        self.register_buffer("kv_weight_int8", torch.cat((pk.weight_int8, pv.weight_int8), dim=0).contiguous())
        self.register_buffer("kv_weight_scale", torch.cat((pk.weight_scale, pv.weight_scale), dim=0).contiguous())
        self.register_buffer("kv_bias", torch.cat((pk.bias, pv.bias), dim=0).contiguous())

        self._plan: dict[torch.device, tuple[bool, bool, bool]] = {}
        if device is not None:
            self.plan(device)

    def plan(self, device: torch.device) -> tuple[bool, bool, bool]:
        if device not in self._plan:
            channels = self.norm1.weight.shape[0]
            qkv, fc1 = self.attn.qkv, self.mlp.fc1
            rows = torch.zeros(64, channels, device=device, dtype=torch.bfloat16)
            q = torch.zeros(64, channels, device=device, dtype=torch.int8)
            s = torch.ones(64, 1, device=device)
            ln = ops.layernorm_quantize_int8_rows.supports(rows, self.norm1.weight, self.norm1.bias, self.norm1.eps)
            gelu = self._exact_gelu and ops.int8_linear.supports(
                q, fc1.weight_int8, s, fc1.weight_scale, fc1.bias, gelu=True
            )
            residual = ops.int8_linear.supports(
                q, qkv.weight_int8[:channels], s, qkv.weight_scale[:channels], None, residual=rows
            )
            self._plan[device] = (ln, gelu, residual)
        return self._plan[device]

    @staticmethod
    def _linear(lin: DynamicInt8Linear, q: Tensor, s: Tensor, **epilogue) -> Tensor:
        return ops.int8_linear(q, lin.weight_int8, s, lin.weight_scale, lin.bias, **epilogue)

    @staticmethod
    def _norm_quantize(norm: nn.LayerNorm, x: Tensor, fused: bool) -> tuple[Tensor, Tensor]:
        if fused:
            return ops.layernorm_quantize_int8_rows(x, norm.weight, norm.bias, norm.eps)
        return ops.quantize_int8_rows(norm(x))

    def forward(
        self,
        x: Tensor,
        y: Tensor,
        x_rope: tuple[Tensor, Tensor],
        y_rope: tuple[Tensor, Tensor],
    ) -> Tensor:
        if x.dtype != torch.bfloat16 or y.dtype != torch.bfloat16 or not x.is_contiguous() or not y.is_contiguous():
            return self._unfused(x, y, x_rope, y_rope)
        ln, gelu, residual = self.plan(x.device)
        batch, nx, channels = x.shape
        ny = y.shape[1]
        x_flat = x.reshape(-1, channels)
        y_flat = y.reshape(-1, channels)

        # 1. Self-attention
        attn = self.attn
        head_dim = channels // attn.num_heads
        q, s = self._norm_quantize(self.norm1, x_flat, ln)
        qkv = self._linear(attn.qkv, q, s).reshape(batch, nx, 3, attn.num_heads, head_dim)
        qh, kh, vh = (t.transpose(1, 2) for t in qkv.unbind(2))
        qh, kh = ops.rope_qk_2d.get(attn.rope_impl)(qh, kh, *x_rope)
        out = ops.sdpa.get(attn.sdpa_impl)(qh, kh, vh)
        q, s = ops.quantize_int8_rows_heads(out)
        x1 = self._linear(attn.proj, q, s, residual=x_flat) if residual else x_flat + self._linear(attn.proj, q, s)

        # 2. Cross-attention (packed projk + projv)
        ca = self.cross_attn
        c_head_dim = channels // ca.num_heads
        q_q, s_q = self._norm_quantize(self.norm2, x1, ln)
        if self._has_norm_y:
            q_kv, s_kv = self._norm_quantize(self.norm_y, y_flat, ln)
        else:
            q_kv, s_kv = ops.quantize_int8_rows(self.norm_y(y_flat))
        qh = self._linear(ca.projq, q_q, s_q).reshape(batch, nx, ca.num_heads, c_head_dim).transpose(1, 2)
        kv = ops.int8_linear(q_kv, self.kv_weight_int8, s_kv, self.kv_weight_scale, self.kv_bias)
        kv = kv.reshape(batch, ny, 2, ca.num_heads, c_head_dim)
        kh, vh = kv[:, :, 0].transpose(1, 2), kv[:, :, 1].transpose(1, 2)
        qh, kh = ops.rope_qk_2d.get(ca.rope_impl)(qh, kh, x_rope[0], x_rope[1], y_rope[0], y_rope[1])
        out_c = ops.sdpa.get(ca.sdpa_impl)(qh, kh, vh)
        q, s = ops.quantize_int8_rows_heads(out_c)
        x2 = self._linear(ca.proj, q, s, residual=x1) if residual else x1 + self._linear(ca.proj, q, s)

        # 3. MLP
        q, s = self._norm_quantize(self.norm3, x2, ln)
        h = self._linear(self.mlp.fc1, q, s, gelu=True) if gelu else self.mlp.act(self._linear(self.mlp.fc1, q, s))
        fc2_cs = self.mlp.fc2.channel_scale
        q, s = ops.quantize_int8_rows(h, channel_scale=fc2_cs) if fc2_cs is not None else ops.quantize_int8_rows(h)
        x3 = self._linear(self.mlp.fc2, q, s, residual=x2) if residual else x2 + self._linear(self.mlp.fc2, q, s)
        return x3.view(batch, nx, channels)

    def _unfused(self, x: Tensor, y: Tensor, x_rope: tuple[Tensor, Tensor], y_rope: tuple[Tensor, Tensor]) -> Tensor:
        x = x + self.attn(self.norm1(x), x_rope)
        y_norm = self.norm_y(y)
        x = x + self.cross_attn(self.norm2(x), y_norm, y_norm, x_rope, y_rope)
        return x + self.mlp(self.norm3(x))


def fuse_dynamic_decoder_blocks(model: nn.Module) -> list[str]:
    """Replace every fusable :class:`DecoderBlock` by a :class:`DynamicInt8DecoderBlock`."""
    fused = []
    for name, module in list(model.named_modules()):
        if _is_fusable_decoder_block(module):
            dev = next(module.parameters()).device
            set_submodule(model, name, DynamicInt8DecoderBlock(module, dev))
            fused.append(name)
    return fused


class FusedMXFP8LocalFeatureMlp(nn.Module):
    """Two-layer MXFP8 GELU MLP with ``fc1``'s bias, GELU and ``fc2``'s static E4M3 quantization fused into one
    Triton kernel (:meth:`StaticMXFP8Linear.quantize_gelu`)."""

    def __init__(self, mlp: nn.Module) -> None:
        super().__init__()
        self.fc1: StaticMXFP8Linear = mlp.fc1
        self.act: nn.GELU = mlp.act
        self.fc2: StaticMXFP8Linear = mlp.fc2

    def forward(self, x: Tensor) -> Tensor:
        shape = x.shape
        hidden = self.fc1.matmul(self.fc1.quantize(x.reshape(-1, shape[-1])))
        quantized = self.fc2.quantize_gelu(hidden, self.act.approximate, input_bias=self.fc1.bias)
        return self.fc2.matmul(quantized, bias=True).to(x.dtype).reshape(*shape[:-1], self.fc2.out_features)


def fuse_mxfp8_head_mlps(model: nn.Module) -> list[str]:
    """Fuse ``head_local_features`` MLPs whose ``fc1`` / ``fc2`` are :class:`StaticMXFP8Linear`."""
    fused = []
    for head_name in ("downstream_head1", "downstream_head2"):
        head = getattr(model, head_name, None)
        mlp = getattr(head, "head_local_features", None)
        if (
            mlp is not None
            and isinstance(getattr(mlp, "fc1", None), StaticMXFP8Linear)
            and isinstance(getattr(mlp, "fc2", None), StaticMXFP8Linear)
            and isinstance(getattr(mlp, "act", None), nn.GELU)
            and mlp.fc1.bias is not None
        ):
            head.head_local_features = FusedMXFP8LocalFeatureMlp(mlp)
            fused.append(f"{head_name}.head_local_features")
    return fused
