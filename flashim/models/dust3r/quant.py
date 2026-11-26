"""DUSt3R quantization targets, outlier & precision diagnosis, and measured accuracy evidence.

All 289 linear layers in DUSt3R live in the CroCo ViT-L encoder and two-branch ViT-B cross-attention decoder;
the patch embedding and two DPT 3D-pointmap heads are convolutions (kept in ``PROTECTED``).

Why 2 px original-image ``match_agree`` is NOT meaningful for DUSt3R
--------------------------------------------------------------------
On the 20 MegaDepth-1500 pairs (``fidelity.py --pairs-per-scene 4``), original images are 2.50x (1.99x-3.13x) larger
than DUSt3R's 512 px working grid, so a 2 px original-image radius is **0.80 working pixels** (down to 0.64 working px
on 1600 px images) — strictly smaller than 1 working pixel. ``agree@2px`` therefore demands exact 0-pixel shift on
both views for dense reciprocal 3D nearest neighbors whose adjacent-pixel depth spacing is only ~0.1-0.2%. Even
toggling TF32 in FP32 drops ``agree@2px`` to **0.7926** despite 0.0105% median relative 3D point error and 1.000000
confidence correlation (while ``agree@4px = 0.9863`` and ``agree@8px = 0.9992``; two identical ``fp32`` TF32-off runs
agree at 1.0000).

Outlier & precision diagnosis (MegaDepth-1500 activations and FP32 weights)
---------------------------------------------------------------------------
- **BF16 7-bit mantissa degrades both DPT and CroCo residual streams**:
  1. In BF16 DPT heads, the 7-bit mantissa quantizes 3D coordinates coarser than 1-pixel spacing (~0.004 vs ~0.0012
     at depth 0.6), collapsing ~35% of reciprocal matches (10-pair 2px agree 0.0912 vs 0.2788 FP32 DPT).
  2. Even with FP16 or FP32 DPT heads, DUSt3R's CroCo residual stream reaches peak magnitude **648.65**
     (``enc_blocks.23``, 100x below FP16 max 65504; zero non-finite outputs across 100 MegaDepth pairs). At
     ``[512, 1024)`` BF16 has ULP = **4.0** vs FP16 ULP = **0.5** (8x finer). Both encoder and decoder residual streams
     contribute equally (isolated on pairs 12/14/16: ``fp32`` 2.87/1.32/2.28 px, ``enc=fp16+dec=bf16`` 3.63/1.92/1.99,
     ``enc=bf16+dec=fp16`` 3.66/1.80/2.10, ``bf16`` 5.78/3.25/4.04, ``fp32+autocast(bf16)`` 5.90/3.07/4.85,
     ``fp16`` 2.81/1.47/1.88). On 100 MegaDepth pairs (20/scene x 5 scenes), BF16 compute dropped 5px epipolar
     precision by -3.0% to -4.3% across all reduced-precision presets and dropped ``int8`` ``poselib@1px`` AUC@5 by
     -0.027 (0.3199 vs 0.3468 FP32). Switching ``compute_dtype`` to **FP16** (with fused FP16 INT8/LayerNorm
     kernels) eliminates the error on all three presets and runs faster (zero backbone->DPT dtype casts).
- **Backbone outliers**: milder than MASt3R (``enc_blocks.0.mlp.fc2`` activation amax/med 268.0x vs 3860x; max
  weight row kurtosis 398.2 on ``enc_blocks.0.mlp.fc1``, 291.2 on ``dec_blocks2.6.cross_attn.projk``). Folding
  LayerNorm affine parameters via :func:`rotate_dust3r` inflated weight ranges and hurt 2px agreement on DUSt3R
  (0.2664 -> 0.2083 on INT8, 0.2549 -> 0.2097 on hybrid FP8), whereas unrotated per-token dynamic INT8 with online
  block-64 Sylvester Hadamard (``H_64`` via :data:`flashim.ops.rotate_quantize_int8_rows`) on post-GELU ``mlp.fc2``
  layers matches FP32 across all 289 linears (100% quantized).

Tried and rejected schemes (10-pair / 100-pair MegaDepth-1500 screens):
- ``bf16`` + BF16 DPT: 10p 2px 0.0912, 4px 0.6559, count ratio 0.6448 (BF16 coordinate staircase)
- ``bf16`` + FP16 DPT (BF16 backbone): 100p all/top5k epi@5px 78.69%/89.85% (-3.3%/-4.3% vs FP32), ``int8`` AUC@5 -0.027
- Rotated dynamic INT8 (with / without ``fc2`` ``H_64``): 10p 2px 0.2083 / 0.1728 (worse than unrotated 0.2664)
- Pure unrotated static MXFP8 (all 289L): 10p 2px 0.2266, 4px 0.9039 (E4M3 3-bit mantissa degrades decoder)
- SageAttention (``sage`` / ``sage-fp8`` / ``sage-fp8-warp``): 1.4-2.0x slower at N=768 and collapses agreement to 0.0000

Measured accuracy & speed of selected presets (RTX 5070 Ti ``sm_120``, FP16 compute + FP16 NHWC DPT + fused FP16
encoder/decoder INT8 blocks + Triton ``upsample_bilinear`` + fused ``dust3r_pts3d_conf`` + CUDA BVH
``reciprocal_nn`` + CUDA graphs):

- **20-pair MegaDepth-1500** (``--pairs-per-scene 4``, vs ``fp32`` TF32-off; ``fidelity.py`` uses TF32-on FP32;
  ``poselib`` LO-RANSAC 3 seeds 17/29/42, 5000 uniform matches):
  - ``fp32`` (TF32 off, r1 vs r2): agree@2/4/8px 1.0000/1.0000/1.0000, pts3d rel p50/p90 0/0, conf r/s 1.000000/1.000000,
    all/top5k epi med 2.323/1.711 px (@5px 81.31%/93.84%); poselib@1px 0.2827/0.4839/0.6378, @2px 0.3052/0.5226/0.6851.
  - ``fp32`` (TF32 on): agree@2/4/8px 0.7926/0.9863/0.9992, pts3d rel p50/p90 0.011%/0.020%, conf r/s 1.000000/1.000000,
    all/top5k epi med 2.294/1.705 px (@5px 81.47%/93.72%); poselib@1px 0.2911/0.4775/0.6166, @2px 0.3214/0.5257/0.6856.
  - ``bf16`` (FP16 compute): agree@2/4/8px 0.5486/0.9493/0.9975 (``fidelity.py`` 0.5495), pts3d rel p50/p90
    0.034%/0.074%, conf r/s 0.999998/0.999997, all/top5k epi med 2.301/1.623 px (@5px 80.77%/93.59%);
    poselib@1px 0.3063/0.5039/0.6644, @2px 0.3358/0.5337/0.6974.
  - ``int8`` (calibration-free, 289/289 linears, fused FP16): agree@2/4/8px 0.4011/0.9349/0.9962 (``fidelity.py``
    0.4034), pts3d rel p50/p90 0.153%/0.527%, conf r/s 0.999823/0.999515, all/top5k epi med 2.330/1.623 px
    (@5px 80.39%/93.48%); poselib@1px 0.3272/0.5321/0.7001, @2px 0.3231/0.5428/0.7039.
  - ``fp8`` (hybrid 24 MXFP8 ``enc_fc2`` + 265 INT8, fused FP16): agree@2/4/8px 0.4234/0.9386/0.9966
    (``fidelity.py`` 0.4096), pts3d rel p50/p90 0.165%/0.638%, conf r/s 0.999814/0.999479, all/top5k epi med
    2.323/1.585 px (@5px 80.23%/93.15%); poselib@1px 0.3613/0.5348/0.7022, @2px 0.3677/0.5614/0.7183.
- **100-pair MegaDepth-1500** (20/scene x 5 scenes, ``poselib`` LO-RANSAC 3 seeds 17/29/42, 5000 uniform matches):
  - ``fp32``: all/top5k epi@5px 81.96%/94.18%; poselib@1px AUC@5/10/20 0.3468/0.5273/0.6829; @2px 0.3582/0.5343/0.7008.
  - ``bf16`` (FP16): all/top5k epi@5px 81.30%/93.99%; poselib@1px 0.3487/0.5292/0.6917; @2px 0.3765/0.5646/0.7174.
  - ``int8`` (FP16): all/top5k epi@5px 80.74%/93.89%; poselib@1px 0.3515/0.5348/0.6910; @2px 0.3713/0.5542/0.7074.
  - ``fp8`` (FP16):  all/top5k epi@5px 81.00%/94.24%; poselib@1px 0.3293/0.5189/0.6819; @2px 0.3887/0.5569/0.7047.
- **Interleaved best-of-3 eager+CUDA-graph pair latency** (Toronto 384x512 / Sacre 352x512):
  ``fp32`` 110.03 / 106.94 ms; ``bf16`` 29.55 / 31.93 ms; ``int8`` 19.61 / 21.92 ms; ``fp8`` 19.79 / 21.42 ms.
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
    DynamicInt8DecoderBlock,
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
    "DynamicInt8DecoderBlock",
    "attach_fc2_hadamard",
    "fuse_dynamic_decoder_blocks",
    "rotate_dust3r",
]
