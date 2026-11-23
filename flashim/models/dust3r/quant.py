"""DUSt3R quantization targets, outlier diagnosis, and measured accuracy evidence.

All 289 linear layers in DUSt3R live in the CroCo ViT-L encoder and two-branch ViT-B cross-attention decoder;
the patch embedding and two DPT 3D-pointmap heads are convolutions (kept in ``PROTECTED``).

Outlier & precision diagnosis (on MegaDepth-1500 / Toronto activations and FP32 weights):

- **DPT head precision (critical)**: BF16's 7-bit mantissa quantizes 3D coordinates coarser than 1-pixel spacing
  (~0.004 vs ~0.0012 units at depth 0.6), collapsing ~35% of reciprocal 3D matches (10-pair MegaDepth 2px match
  agreement 0.0912 vs 0.2788 FP32 DPT, 4px 0.6559 vs 0.9436, match count ratio 0.6448 vs 0.9679). Switching the two
  DPT heads to **FP16 NHWC** (10-bit mantissa, tensor cores) eliminates the staircase (2px 0.2623, 4px 0.9294,
  count ratio 0.8948, zero non-finite values) at +0.2 ms cost (5.1 ms faster than FP32 DPT).
- **Backbone outliers**: milder than MASt3R (``enc_blocks.0.mlp.fc2`` activation amax/med 268.0x vs 3860x; max
  weight row kurtosis 398.2 on ``enc_blocks.0.mlp.fc1``, 291.2 on ``dec_blocks2.6.cross_attn.projk``). Folding
  LayerNorm affine parameters via :func:`rotate_dust3r` inflated weight ranges and *hurt* 2px agreement on DUSt3R
  (0.2664 -> 0.2083 on INT8, 0.2549 -> 0.2097 on hybrid FP8), whereas unrotated per-token dynamic INT8 with online
  block-64 Sylvester Hadamard (``H_64`` via :data:`flashim.ops.rotate_quantize_int8_rows`) on post-GELU ``mlp.fc2``
  layers matches or exceeds ``bf16`` across all 289 linears (100% quantized).

Tried and rejected schemes (10-pair MegaDepth-1500 screen, ``max_matches=5000``, FP16 DPT unless noted;
``bf16+fp16_dpt`` ref: 2px 0.2623, 4px 0.9294, match count ratio 0.8948):

- ``bf16`` + BF16 DPT: 2px 0.0912, 4px 0.6559, count ratio 0.6448 (BF16 coordinate staircase)
- ``bf16`` + FP32 DPT: 2px 0.2788, 4px 0.9436, count ratio 0.9679 (+5.1 ms slower than FP16 DPT)
- Rotated dynamic INT8 (no ``fc2`` ``H_64``): 2px 0.1728, 4px 0.9186
- Rotated dynamic INT8 + ``H_64`` on ``fc2``: 2px 0.2083, 4px 0.9203
- Unrotated dynamic INT8 (no ``fc2`` ``H_64``): 2px 0.2567, 4px 0.9251
- Pure unrotated static MXFP8 (all 289L): 2px 0.2266, 4px 0.9039 (E4M3 3-bit mantissa degrades decoder)
- Rotated hybrid MXFP8 (``enc_fc2``) + dynamic INT8: 2px 0.2097, 4px 0.9284
- SageAttention (``sage`` / ``sage-fp8`` / ``sage-fp8-warp``): 1.4-2.0x slower at N=768 and collapses agreement to 0.0000

Selected presets (verified with ``benchmarks/accuracy/fidelity.py --models dust3r --presets bf16 int8 fp8
--pairs-per-scene 4`` on 20 MegaDepth-1500 pairs, ``max_matches=5000``, 2px original-image radius vs FP32;
interleaved best-of-3 eager+CUDA-graph pair latencies on RTX 5070 Ti ``sm_120`` for Toronto 384x512 / Sacre 352x512
vs FP32 109.76 / 106.61 ms):

- ``bf16`` (BF16 backbone + FP16 NHWC DPT + Triton ``upsample_bilinear2x`` + fused ``dust3r_pts3d_conf`` + CUDA BVH
  ``reciprocal_nn`` + CUDA graphs): ``match_agree`` 0.2437, ``match_ratio`` 1.0000; 31.71 / 33.05 ms.
- ``int8`` (calibration-free: unrotated per-token dynamic INT8 on all 289/289 backbone linears with online ``H_64``
  on all 48 ``mlp.fc2`` + fused encoder/decoder blocks + FP16 NHWC DPT + CUDA graphs): ``match_agree`` 0.2455
  (+0.0018 vs ``bf16``), ``match_ratio`` 1.0000; 20.01 / 21.63 ms.
- ``fp8`` (hybrid: unrotated static MXFP8 cuBLAS ``F.scaled_mm`` on all 24 ``enc_blocks.*.mlp.fc2`` linears fused
  into ``DynamicInt8TransformerBlock`` via ``gelu_quantize`` + unrotated dynamic INT8 on the other 265 linears with
  ``H_64`` on decoder ``mlp.fc2`` + FP16 NHWC DPT + CUDA graphs): ``match_agree`` 0.2453 (+0.0016 vs ``bf16``),
  ``match_ratio`` 1.0000; 19.72 / 21.29 ms.
"""

from __future__ import annotations

from pathlib import Path

import torch
from torch import Tensor, nn

from flashim.models.mast3r.quant import (
    BACKBONE_LINEARS,
    DEC_LINEARS,
    ENC_FC2_LINEARS,
    ENC_NON_FC2_LINEARS,
    attach_fc2_hadamard,
    fuse_dynamic_decoder_blocks,
)
from flashim.quant.rotation import hadamard, layernorm_orthogonal
from flashim.quant.scheme import (
    INT8_TOKEN_ACTIVATIONS,
    INT8_WEIGHTS,
    MXFP8,
    QuantConfig,
    QuantScheme,
)

CALIBRATION_DIR = Path(__file__).parent / "calibration"
FP8_CALIBRATION = CALIBRATION_DIR / "dust3r_fp8.safetensors"

# Never quantized: input patch embedding and the two DPT 3D-pointmap / confidence conv heads.
PROTECTED = (
    "patch_embed.*",
    "downstream_head1.dpt.*",
    "downstream_head2.dpt.*",
)

# Calibration-free rotated dynamic INT8 on all 289 encoder+decoder linears (100% of linears in DUSt3R).
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
# per-token dynamic INT8 on the remaining 265 encoder/decoder linears.
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
def rotate_dust3r(
    model: nn.Module,
    *,
    enc_block: int | None = None,
    dec_block: int | None = None,
    rotate_enc: bool = True,
    rotate_dec: bool = True,
    rotate_v: bool = True,
) -> int:
    """Fold exact LayerNorm-commuting orthogonal rotations (:func:`~flashim.quant.rotation.layernorm_orthogonal`)
    into DUSt3R's encoder / decoder residual streams and per-head Sylvester Hadamard rotations into attention
    ``V -> proj``.

    Every LayerNorm's affine weight and bias are folded into its consuming linear(s) / 1x1 DPT conv(s) first so
    the remaining un-affined ``LayerNorm`` satisfies ``LN_0(x Q) = LN_0(x) Q`` identically (``Q^T Q = I``,
    ``Q 1 = 1``). Exact in real arithmetic (preserves ``pts3d`` and ``conf``) with zero runtime overhead.
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

    def fold_ln(norm: nn.LayerNorm, consumers: list[nn.Module]) -> None:
        w, b = norm.weight.double(), norm.bias.double()
        for mod in consumers:
            if isinstance(mod, nn.Linear):
                wt = mod.weight.double()
                if mod.bias is not None:
                    mod.bias.copy_((mod.bias.double() + wt @ b).to(mod.bias.dtype))
                mod.weight.copy_((wt * w[None, :]).to(mod.weight.dtype))
            elif isinstance(mod, nn.Conv2d) and mod.kernel_size == (1, 1):
                wt = mod.weight.double()[:, :, 0, 0]
                if mod.bias is not None:
                    mod.bias.copy_((mod.bias.double() + wt @ b).to(mod.bias.dtype))
                mod.weight[:, :, 0, 0].copy_((wt * w[None, :]).to(mod.weight.dtype))
            else:
                raise TypeError(f"Unsupported LayerNorm consumer: {type(mod).__name__}")
        norm.weight.fill_(1.0)
        norm.bias.zero_()

    def read_lin(lin: nn.Linear, q: Tensor) -> None:
        lin.weight.copy_((lin.weight.double() @ q).to(lin.weight.dtype))
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
            fold_ln(blk.norm1, [blk.attn.qkv])
            read_lin(blk.attn.qkv, q_enc)
            write_lin(blk.attn.proj, q_enc)
            fold_ln(blk.norm2, [blk.mlp.fc1])
            read_lin(blk.mlp.fc1, q_enc)
            write_lin(blk.mlp.fc2, q_enc)
            if rotate_v:
                dim = model.enc_embed_dim
                write_lin(blk.attn.qkv, enc_vh, slice(2 * dim, 3 * dim))
                read_lin(blk.attn.proj, enc_vh)
        fold_ln(
            model.enc_norm,
            [
                model.decoder_embed,
                model.downstream_head1.dpt.act_postprocess[0][0],
                model.downstream_head2.dpt.act_postprocess[0][0],
            ],
        )
        read_lin(model.decoder_embed, q_enc)
        read_c1(model.downstream_head1.dpt.act_postprocess[0][0], q_enc)
        read_c1(model.downstream_head2.dpt.act_postprocess[0][0], q_enc)

    if rotate_dec:
        write_lin(model.decoder_embed, q_dec)
        for branch in ("dec_blocks", "dec_blocks2"):
            for blk in getattr(model, branch):
                fold_ln(blk.norm1, [blk.attn.qkv])
                read_lin(blk.attn.qkv, q_dec)
                write_lin(blk.attn.proj, q_dec)
                fold_ln(blk.norm2, [blk.cross_attn.projq])
                read_lin(blk.cross_attn.projq, q_dec)
                if isinstance(blk.norm_y, nn.LayerNorm):
                    fold_ln(blk.norm_y, [blk.cross_attn.projk, blk.cross_attn.projv])
                read_lin(blk.cross_attn.projk, q_dec)
                read_lin(blk.cross_attn.projv, q_dec)
                write_lin(blk.cross_attn.proj, q_dec)
                fold_ln(blk.norm3, [blk.mlp.fc1])
                read_lin(blk.mlp.fc1, q_dec)
                write_lin(blk.mlp.fc2, q_dec)
                if rotate_v:
                    dim = model.dec_embed_dim
                    write_lin(blk.attn.qkv, dec_vh, slice(2 * dim, 3 * dim))
                    read_lin(blk.attn.proj, dec_vh)
                    write_lin(blk.cross_attn.projv, dec_vh)
                    read_lin(blk.cross_attn.proj, dec_vh)
        fold_ln(
            model.dec_norm,
            [
                model.downstream_head1.dpt.act_postprocess[3][0],
                model.downstream_head2.dpt.act_postprocess[3][0],
            ],
        )
        read_c1(model.downstream_head1.dpt.act_postprocess[3][0], q_dec)
        read_c1(model.downstream_head2.dpt.act_postprocess[3][0], q_dec)
        for head in (model.downstream_head1, model.downstream_head2):
            read_c1(head.dpt.act_postprocess[1][0], q_dec)
            read_c1(head.dpt.act_postprocess[2][0], q_dec)

    return len(rotated)


__all__ = [
    "FP8_CALIBRATION",
    "FP8_CONFIG",
    "INT8_CONFIG",
    "PROTECTED",
    "attach_fc2_hadamard",
    "fuse_dynamic_decoder_blocks",
    "rotate_dust3r",
]
