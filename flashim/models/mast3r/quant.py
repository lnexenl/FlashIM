"""MASt3R quantization targets, LayerNorm-commuting orthogonal rotation, block fusion and measured accuracy evidence.

Outlier diagnosis (on MegaDepth-1500 calibration activations and FP32 weights):

- **Channel & weight kurtosis outliers**: ``enc_blocks.0.mlp.fc2`` activation channel amax ratio 3860x,
  ``enc_blocks.1.norm1.weight`` ratio 69.7x; massive per-row weight kurtosis in cross-attention and local-feature
  head (``dec_blocks2.6.cross_attn.projk`` kurtosis 611.1, max/med 210.5; ``head1.fc2`` kurtosis 447.9, mean row
  max/med 29.4 across all 6400 output channels; ``enc_blocks.0.attn.proj`` kurtosis 348.3).
- **Solution**:
  1. Fold exact LayerNorm-commuting orthogonal rotations (:func:`~flashim.quant.rotation.layernorm_orthogonal`:
     ``Q_1024`` on the encoder residual stream, ``Q_768`` on the decoder residual stream, ``Q^T Q = I`` and
     ``Q 1 = 1`` so ``LN_0(x Q) = LN_0(x) Q`` identically after folding LayerNorm affine parameters into
     consuming linears and 1x1 DPT convs) plus per-head Sylvester Hadamard ``H_64`` on attention ``V -> proj`` at
     zero runtime cost. Slashes cross-attention weight kurtosis from 611.1 to 3.7 and lifts decoder W8A8 alone from
     0.6007 to 0.6982 (+0.0975).
  2. Apply an online block-64 Sylvester Hadamard transform (``H_64``) fused into per-row dynamic INT8 quantization
     (:data:`flashim.ops.rotate_quantize_int8_rows`, Triton BF16 tensor-core ``tl.dot``: 15.0 us on ``[1536, 4096]``
     vs 61.9 us for plain ``quantize_int8_rows``, 12.5 us vs 45.0 us on ``[768, 3072]``) on post-GELU ``mlp.fc2``
     layers where residual-stream rotation cannot cross GELU.
  3. Keep only the 4 final ``head_local_features`` linears (``fc1``/``fc2`` on each view) in BF16 after logging
     every algorithmic alternative on all 293 linears (all-293 INT8 peaked at 0.7137 / 0.7162 due to ``head.fc2``'s
     447.9 weight kurtosis projecting directly to 24-D normalized descriptors). All 24 encoder blocks and 24 decoder
     blocks (289 / 293 transformer linears, 98.6%) run quantized and fused.

Measured on 20 MegaDepth-1500 pairs (``benchmarks/accuracy/fidelity.py --models mast3r --pairs-per-scene 4``,
512 px working resolution, 2 px original-image radius vs FP32; PIL and default decoders) and RTX 5070 Ti (sm_120)
CUDA-graphed ``match_prepared`` speed (Toronto 384x512):

Tried and rejected schemes (20-pair PIL ``bf16`` baseline = 0.7381; default decoder ``bf16`` = 0.7324):

- Unsmoothed per-token dynamic INT8 (all 293L): match_agree 0.5418 (-0.1963)
- SmoothQuant-only dynamic INT8 (all 293L, alpha=0.25/0.35/0.50/0.60/0.75): 0.5614..0.6339 (-0.1042 at a=0.60),
  16.35 ms (old ``int8``)
- Pure static / dynamic / rotated MXFP8 (all 293L): 0.5004..0.5610 — E4M3's 3-bit mantissa fails decoder
- Pure RowWise FP8 E4M3 (``torch._scaled_mm``, FP32 per-row/per-col scales, rotated + H_64, all 293L): 0.5532
  (and 1.6-2.7x slower than Triton INT8 GEMM on RTX 5070 Ti across every MASt3R shape)
- Hybrid MXFP8 on encoder+head + SmoothQuant INT8 decoder (old ``fp8``): 0.6141 (default 0.6126), 15.85 ms
- Rotated hybrid MXFP8 on encoder+head + rotated INT8 decoder (full/block-16/unrot, static or dynamic MXFP8):
  0.6414..0.6496; encoder-only MXFP8 (``enc_0..23``): 0.6734..0.6960; ``enc_12..23`` MXFP8 + head BF16: 0.7020
  (amax), 0.6846 (p999), 0.6735 (p995 — percentile clipping strictly worse than amax); ``enc_16..23`` MXFP8 + head
  BF16: 0.7113 (-0.0268, 17.12 ms)
- Rotated RowWise FP8 on ``enc_2..23`` / ``enc_0..23`` / ``enc_12..23`` + rotated INT8: 0.6778..0.7040 (and slower)
- Rotated dynamic INT8 (all 293L, no ``fc2`` fix): 0.6375 (-0.1006)
- Rotated dynamic INT8 + SmoothQuant on ``fc2`` only (all 293L, alpha=0.50/0.60/0.75): 0.6809 / 0.6703 / 0.6317
- Rotated dynamic INT8 + online unsigned ``H_128`` / ``H_256`` on ``fc2`` (all 293L): 0.6990 / 0.6989 (DC spike on
  positive GELU); signed ``H_16..1024`` (all 293L): 0.6989..0.7137; ``SQ(0.5)+sH_64`` (all 293L): 0.7162 (-0.0219)
- Rotated dynamic INT8 + Triton ``H_64`` on all 293L (0 BF16): 0.7033 (PIL; 15.45 ms); with ``head_fc2`` (2L) BF16:
  0.7133 (PIL; 16.41 ms)

Selected presets (verified with ``benchmarks/accuracy/fidelity.py --models mast3r --presets bf16 int8 fp8
--pairs-per-scene 4`` and RTX 5070 Ti ``match_prepared`` median ms on Toronto 384x512 / Sacre 352x512+512x384):

- ``bf16``: match_agree 0.7324, match_ratio 0.9967 | 26.35 ms / 27.37 ms (vs ``fp32`` 108.43 ms / 105.37 ms).
- ``int8`` (calibration-free: ``rotate_mast3r`` + dynamic per-token INT8 on all 289 backbone linears with Triton
  ``H_64`` on all 48 ``mlp.fc2`` + fused encoder/decoder blocks; 4 ``head_local_features`` linears BF16):
  match_agree 0.7237 (-0.0087 vs ``bf16``; +0.0911 over prior SmoothQuant ``int8``), match_ratio 1.0003 |
  17.00 ms / 18.68 ms (1.55x / 1.47x faster than optimized ``bf16``).
- ``fp8`` (hybrid: ``rotate_mast3r`` + static MXFP8 cuBLAS ``F.scaled_mm`` on all 24 ``enc_blocks.*.mlp.fc2``
  ``[1536, 4096->1024]`` linears — 33% of encoder GEMM FLOPs and the shape where cuBLAS MXFP8 beats Triton INT8 ~2x
  — fused into ``DynamicInt8TransformerBlock`` via ``gelu_quantize``, + rotated dynamic INT8 on the other 265
  backbone linears with Triton ``H_64`` on all 24 decoder ``mlp.fc2``; 4 ``head_local_features`` linears BF16):
  match_agree 0.7139 (-0.0185 vs ``bf16``; +0.1013 over prior ``fp8``), match_ratio 0.9993 |
  16.71 ms / 18.08 ms (1.58x / 1.51x faster than optimized ``bf16``).
"""

from __future__ import annotations

from pathlib import Path

import torch
from torch import Tensor, nn

from flashim import ops
from flashim.models.mast3r.croco import DecoderBlock
from flashim.quant.methods.int8_linear import DynamicInt8Linear
from flashim.quant.rotation import hadamard, layernorm_orthogonal
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

ENC_NON_FC2_LINEARS = (
    "enc_blocks.*.attn.qkv",
    "enc_blocks.*.attn.proj",
    "enc_blocks.*.mlp.fc1",
)
ENC_FC2_LINEARS = ("enc_blocks.*.mlp.fc2",)
ENC_LINEARS = (*ENC_NON_FC2_LINEARS, *ENC_FC2_LINEARS)
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
BACKBONE_LINEARS = (*ENC_LINEARS, *DEC_LINEARS)
ALL_TRANSFORMER_LINEARS = (*BACKBONE_LINEARS, *HEAD_MLP_LINEARS)

# Never quantized: input patch embedding and the 3D-point/confidence DPT conv heads.
PROTECTED = (
    "patch_embed.*",
    "downstream_head1.dpt.*",
    "downstream_head2.dpt.*",
)

# Calibration-free rotated dynamic INT8 on all 289 encoder+decoder linears (4 head_local_features linears stay BF16).
INT8_CONFIG = QuantConfig(
    {
        "transformer": QuantScheme(
            targets=BACKBONE_LINEARS,
            weights=INT8_WEIGHTS,
            input_activations=INT8_TOKEN_ACTIVATIONS,
        ),
    },
    name="int8",
)

# Hybrid FP8: static MXFP8 (cuBLAS scaled_mm) on the 24 large encoder MLP fc2 projections [4096->1024], rotated
# per-token dynamic INT8 on the remaining 265 encoder/decoder linears (4 head_local_features linears stay BF16).
FP8_CONFIG = QuantConfig(
    {
        "mxfp8": QuantScheme(
            targets=ENC_FC2_LINEARS,
            weights=MXFP8,
            input_activations=MXFP8,
        ),
        "backbone_int8": QuantScheme(
            targets=(*ENC_NON_FC2_LINEARS, *DEC_LINEARS),
            weights=INT8_WEIGHTS,
            input_activations=INT8_TOKEN_ACTIVATIONS,
        ),
    },
    name="fp8",
)


@torch.no_grad()
def rotate_mast3r(
    model: nn.Module,
    *,
    enc_block: int | None = None,
    dec_block: int | None = None,
    rotate_enc: bool = True,
    rotate_dec: bool = True,
    rotate_v: bool = True,
) -> int:
    """Fold exact LayerNorm-commuting orthogonal rotations (:func:`~flashim.quant.rotation.layernorm_orthogonal`) into
    MASt3R's encoder / decoder residual streams and per-head Sylvester Hadamard rotations into attention ``V -> proj``.

    Every LayerNorm's affine weight and bias are folded into its consuming linear(s) / 1x1 DPT conv(s) first so the
    remaining un-affined ``LayerNorm`` satisfies ``LN_0(x Q) = LN_0(x) Q`` identically (``Q^T Q = I``, ``Q 1 = 1``).
    Exact in real arithmetic (preserves ``pts3d``, ``conf``, ``desc``, ``desc_conf``) with zero runtime overhead.
    Returns the number of rotated weight tensors.
    """
    dev = next(model.parameters()).device
    q_enc = layernorm_orthogonal(model.enc_embed_dim, block_size=enc_block, device=dev, dtype=torch.float64)
    q_dec = layernorm_orthogonal(model.dec_embed_dim, block_size=dec_block, device=dev, dtype=torch.float64)
    enc_hd = model.enc_embed_dim // model.enc_num_heads
    dec_hd = model.dec_embed_dim // model.dec_num_heads
    enc_vh = torch.block_diag(*([hadamard(enc_hd, device=dev, dtype=torch.float64)] * model.enc_num_heads))
    dec_vh = torch.block_diag(*([hadamard(dec_hd, device=dev, dtype=torch.float64)] * model.dec_num_heads))
    rotated: set[int] = set()

    def fold_ln(norm: nn.LayerNorm, consumers: list[tuple[nn.Module, slice | None]]) -> None:
        w, b = norm.weight.double(), norm.bias.double()
        for mod, sl in consumers:
            if isinstance(mod, nn.Linear):
                wt = mod.weight.double()
                if sl is None:
                    if mod.bias is not None:
                        mod.bias.copy_((mod.bias.double() + wt @ b).to(mod.bias.dtype))
                    mod.weight.copy_((wt * w[None, :]).to(mod.weight.dtype))
                else:
                    ws = wt[:, sl]
                    if mod.bias is not None:
                        mod.bias.copy_((mod.bias.double() + ws @ b).to(mod.bias.dtype))
                    wt[:, sl] = ws * w[None, :]
                    mod.weight.copy_(wt.to(mod.weight.dtype))
            elif isinstance(mod, nn.Conv2d) and mod.kernel_size == (1, 1):
                wt = mod.weight.double()[:, :, 0, 0]
                if mod.bias is not None:
                    mod.bias.copy_((mod.bias.double() + wt @ b).to(mod.bias.dtype))
                mod.weight[:, :, 0, 0].copy_((wt * w[None, :]).to(mod.weight.dtype))
            else:
                raise TypeError(f"Unsupported LayerNorm consumer: {type(mod).__name__}")
        norm.weight.fill_(1.0)
        norm.bias.zero_()

    def read_lin(lin: nn.Linear, q: Tensor, sl: slice | None = None) -> None:
        wt = lin.weight.double()
        if sl is None:
            lin.weight.copy_((wt @ q).to(lin.weight.dtype))
        else:
            wt[:, sl] = wt[:, sl] @ q
            lin.weight.copy_(wt.to(lin.weight.dtype))
        rotated.add(id(lin))

    def read_c1(conv: nn.Conv2d, q: Tensor) -> None:
        conv.weight[:, :, 0, 0].copy_((conv.weight.double()[:, :, 0, 0] @ q).to(conv.weight.dtype))
        rotated.add(id(conv))

    def write_lin(lin: nn.Linear, q: Tensor, sl: slice | None = None) -> None:
        if sl is None:
            lin.weight.copy_((q.T @ lin.weight.double()).to(lin.weight.dtype))
            if lin.bias is not None:
                lin.bias.copy_((q.T @ lin.bias.double()).to(lin.bias.dtype))
        else:
            lin.weight[sl].copy_((q.T @ lin.weight[sl].double()).to(lin.weight.dtype))
            if lin.bias is not None:
                lin.bias[sl].copy_((q.T @ lin.bias[sl].double()).to(lin.bias.dtype))
        rotated.add(id(lin))

    def write_conv(conv: nn.Conv2d, q: Tensor) -> None:
        wt = conv.weight.double()
        conv.weight.copy_((q.T @ wt.reshape(wt.shape[0], -1)).reshape_as(wt).to(conv.weight.dtype))
        if conv.bias is not None:
            conv.bias.copy_((q.T @ conv.bias.double()).to(conv.bias.dtype))
        rotated.add(id(conv))

    if rotate_enc:
        write_conv(model.patch_embed.proj, q_enc)
        for blk in model.enc_blocks:
            fold_ln(blk.norm1, [(blk.attn.qkv, None)])
            read_lin(blk.attn.qkv, q_enc)
            write_lin(blk.attn.proj, q_enc)
            fold_ln(blk.norm2, [(blk.mlp.fc1, None)])
            read_lin(blk.mlp.fc1, q_enc)
            write_lin(blk.mlp.fc2, q_enc)
            if rotate_v:
                dim = model.enc_embed_dim
                write_lin(blk.attn.qkv, enc_vh, slice(2 * dim, 3 * dim))
                read_lin(blk.attn.proj, enc_vh)
        ed = model.enc_embed_dim
        fold_ln(
            model.enc_norm,
            [
                (model.decoder_embed, None),
                (model.downstream_head1.head_local_features.fc1, slice(0, ed)),
                (model.downstream_head2.head_local_features.fc1, slice(0, ed)),
                (model.downstream_head1.dpt.act_postprocess[0][0], None),
                (model.downstream_head2.dpt.act_postprocess[0][0], None),
            ],
        )
        read_lin(model.decoder_embed, q_enc)
        read_lin(model.downstream_head1.head_local_features.fc1, q_enc, slice(0, ed))
        read_lin(model.downstream_head2.head_local_features.fc1, q_enc, slice(0, ed))
        read_c1(model.downstream_head1.dpt.act_postprocess[0][0], q_enc)
        read_c1(model.downstream_head2.dpt.act_postprocess[0][0], q_enc)

    if rotate_dec:
        write_lin(model.decoder_embed, q_dec)
        for branch in ("dec_blocks", "dec_blocks2"):
            for blk in getattr(model, branch):
                fold_ln(blk.norm1, [(blk.attn.qkv, None)])
                read_lin(blk.attn.qkv, q_dec)
                write_lin(blk.attn.proj, q_dec)
                fold_ln(blk.norm2, [(blk.cross_attn.projq, None)])
                read_lin(blk.cross_attn.projq, q_dec)
                if isinstance(blk.norm_y, nn.LayerNorm):
                    fold_ln(blk.norm_y, [(blk.cross_attn.projk, None), (blk.cross_attn.projv, None)])
                read_lin(blk.cross_attn.projk, q_dec)
                read_lin(blk.cross_attn.projv, q_dec)
                write_lin(blk.cross_attn.proj, q_dec)
                fold_ln(blk.norm3, [(blk.mlp.fc1, None)])
                read_lin(blk.mlp.fc1, q_dec)
                write_lin(blk.mlp.fc2, q_dec)
                if rotate_v:
                    dim = model.dec_embed_dim
                    write_lin(blk.attn.qkv, dec_vh, slice(2 * dim, 3 * dim))
                    read_lin(blk.attn.proj, dec_vh)
                    write_lin(blk.cross_attn.projv, dec_vh)
                    read_lin(blk.cross_attn.proj, dec_vh)
        ed, dd = model.enc_embed_dim, model.dec_embed_dim
        fold_ln(
            model.dec_norm,
            [
                (model.downstream_head1.head_local_features.fc1, slice(ed, ed + dd)),
                (model.downstream_head2.head_local_features.fc1, slice(ed, ed + dd)),
                (model.downstream_head1.dpt.act_postprocess[3][0], None),
                (model.downstream_head2.dpt.act_postprocess[3][0], None),
            ],
        )
        read_lin(model.downstream_head1.head_local_features.fc1, q_dec, slice(ed, ed + dd))
        read_lin(model.downstream_head2.head_local_features.fc1, q_dec, slice(ed, ed + dd))
        read_c1(model.downstream_head1.dpt.act_postprocess[3][0], q_dec)
        read_c1(model.downstream_head2.dpt.act_postprocess[3][0], q_dec)
        for head in (model.downstream_head1, model.downstream_head2):
            read_c1(head.dpt.act_postprocess[1][0], q_dec)
            read_c1(head.dpt.act_postprocess[2][0], q_dec)

    return len(rotated)


def attach_fc2_hadamard(model: nn.Module, *, include_encoder: bool = True, block_size: int = 64) -> int:
    """Attach a Sylvester Hadamard matrix (``_flashim_block_rotate``) to each quantized backbone ``mlp.fc2``
    linear so :func:`~flashim.quant.apply.quantize_model` folds ``W @ block_diag(H)`` into its weights and uses
    :data:`flashim.ops.rotate_quantize_int8_rows` at runtime."""
    dev = next(model.parameters()).device
    blocks: list[nn.Module] = [*model.dec_blocks, *model.dec_blocks2]
    if include_encoder:
        blocks = [*model.enc_blocks, *blocks]
    cache: dict[int, Tensor] = {}
    attached = 0
    for blk in blocks:
        fc2 = blk.mlp.fc2
        if not isinstance(fc2, nn.Linear):
            continue
        b = block_size if fc2.in_features % block_size == 0 else min(block_size, fc2.in_features)
        if b < 16 or (b & (b - 1)) != 0 or fc2.in_features % b != 0:
            continue
        if b not in cache:
            cache[b] = hadamard(b, device=dev, dtype=torch.bfloat16).contiguous()
        fc2._flashim_block_rotate = cache[b]
        attached += 1
    return attached


# -- dynamic INT8 decoder-block fusion -------------------------------------------------------------------


def _is_fusable_decoder_block(block: nn.Module) -> bool:
    if not isinstance(block, DecoderBlock):
        return False
    ca = block.cross_attn
    linears = (block.attn.qkv, block.attn.proj, ca.projq, ca.projk, ca.projv, ca.proj, block.mlp.fc1, block.mlp.fc2)
    return (
        all(isinstance(m, DynamicInt8Linear) for m in linears)
        and all(getattr(m, "channel_scale", None) is None and getattr(m, "block_rotate", None) is None for m in linears[:-1])
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
        fc2_rot = self.mlp.fc2.block_rotate
        if fc2_rot is not None:
            q, s = ops.rotate_quantize_int8_rows(h, fc2_rot, channel_scale=fc2_cs)
        elif fc2_cs is not None:
            q, s = ops.quantize_int8_rows(h, channel_scale=fc2_cs)
        else:
            q, s = ops.quantize_int8_rows(h)
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
