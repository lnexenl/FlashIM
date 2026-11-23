"""UFM quantization schemes, LayerNorm-commuting orthogonal rotation, calibration paths and measured evidence.

Architecture: 24-layer DINOv2 ViT-L/14 encoder (``dim=1024``, 16 heads) -> ``info_sharing.proj_embed`` (1024 -> 768)
-> 12-layer global-attention ``info_sharing`` (``dim=768``, 12 heads) -> two DPT heads (``head1`` for dense flow,
``uncertainty_head`` for covisibility). All 36 transformer blocks are :class:`flashim.layers.TransformerBlock`
instances (144 block linears + 1 ``proj_embed`` = 145 backbone linears).

Outlier & weight-kurtosis diagnosis (on MegaDepth calibration activations after LayerScale folding, FP32 weights):

- **Severe channel activation outliers**: ``encoder.model.blocks.1.attn.qkv`` channel amax ratio 8719x (weight
  kurtosis 466.9), ``blocks.0.attn.qkv`` 666x (w_kurt 137.9); ``info_sharing.self_attention_blocks.{3,1,4,2}.mlp.fc2``
  421x / 340x / 257x / 233x (w_kurt 60.8-317.7); ``encoder.model.blocks.{23,17}.mlp.fc2`` 290x / 236x with token
  ratios 23.2x / 34.8x (w_kurt 84.9 / 312.2).
- **Extreme weight kurtosis**: ``encoder.model.blocks.12.mlp.fc2`` w_kurt 2169.1, ``blocks.9.mlp.fc2`` 606.9,
  ``blocks.1.attn.qkv`` 466.9, ``blocks.{11,14}.mlp.fc2`` 415.4 / 403.5, ``blocks.13.mlp.fc1`` 338.8.
- **Per-region sensitivity** (unrotated dynamic INT8 vs ``bf16`` 0.1841 px / 99.85% <2px / 0.9923 IoU, SDPA):
  - ``enc 0..5``: 0.2006 px / 0.9908 IoU; ``enc 6..11``: 0.1823 px / 0.9914 IoU; ``enc 12..17``: 0.2139 px / 0.9888;
    ``enc 18..23``: 0.2308 px / 0.9802 IoU (accounts for nearly the entire unrotated encoder IoU drop).
  - ``info_sharing.proj_embed`` (single 1024->768 linear): 0.2197 px / 0.9907 IoU (+0.036 px alone; remains +0.028 px
    even post-rotation because ``encoder.model.norm``'s affine scale is folded into its input columns).
  - ``info 0..5``: 0.2095 px / 0.9907 IoU; ``info 6..11``: 0.2280 px / 0.9893 IoU.
  - Attention: ``sage-fp8-warp`` on ``enc 2..23`` alone costs only +0.0018 px (0.1859 px), whereas on ``info_sharing``
    it costs +0.042 px and -0.0038 IoU (0.2262 px / 0.9885); FP16-PV ``sage`` on both costs +0.008 px (0.1922 px),
    and pure torch SDPA stays both most accurate and CUDA-graph safe.
  - DPT heads: stay BF16 (channels-last + Triton ``ops.upsample_bilinear``, cutting heads 21.85 ms FP32 -> 8.34 ms):
    static INT8 on ``head1`` clips wide-baseline flows exceeding calibration range (``assets/toronto``, 311.6 px max
    flow: ``p90=17.70 px, <2px=73.6%`` vs ``p90=0.79 px, <2px=99.93%`` BF16), static INT8 on ``uncertainty_head``
    drops ``sacre_coeur`` IoU 0.960 -> 0.936, and MXFP8 DPT convs are both slower and worse (0.756 px EPE).

Solution (adapted from ``model/mast3r`` commit ``7741ccc``, extended with signed Hadamard for post-GELU ``fc2``):

1. Fold exact LayerNorm-commuting orthogonal rotations (:func:`~flashim.quant.rotation.layernorm_orthogonal`:
   ``Q_1024`` on the encoder residual stream, ``Q_768`` on ``info_sharing``, ``Q^T Q = I`` and ``Q 1 = 1`` so
   ``LN_0(x Q) = LN_0(x) Q`` identically after folding LayerNorm affine weights/biases into consuming linears and
   1x1 DPT ``input_process`` convs) plus per-head Sylvester Hadamard ``H_64`` on attention ``V -> proj`` across all
   36 blocks (FP32 rotation exactness: median flow diff 0.0021 px, max covis diff 0.0008, zero runtime cost).
2. Apply an online signed block-64 Hadamard transform (:func:`~flashim.quant.rotation.signed_hadamard`,
   ``diag(s) @ H_64`` with exact ``+-0.125`` BF16 entries) fused into per-row dynamic INT8 quantization
   (:data:`flashim.ops.rotate_quantize_int8_rows`, 22.3 us on ``[2402, 4096]``) on every quantized ``mlp.fc2`` layer.
   Deterministic Rademacher signs cancel the all-ones DC row of unsigned ``H_64`` that otherwise creates a large DC
   spike on mostly-positive post-GELU activations (cutting all-145 INT8 med EPE from 0.2670 -> 0.2437 px).
3. Keep only the single ``info_sharing.proj_embed`` linear (1 / 145 backbone linears, < 0.05 ms) in BF16; all
   36 / 36 transformer blocks (144 / 145 linears, 99.3%) run quantized and fused in
   :class:`~flashim.quant.passes.dynamic_transformer_block.DynamicInt8TransformerBlock`.
4. For ``fp8``, place static block-32 MXFP8 (cuBLAS ``F.scaled_mm``) on the 14 mid-encoder ``mlp.fc1`` layers
   (``encoder.model.blocks.2..15.mlp.fc1``, ``1024 -> 4096`` — whose inputs already carry ``Q_1024`` rotation and
   avoid both ``enc 0..1``'s input sensitivity and ``enc 16..23`` / ``info_sharing``'s sub-pixel E4M3 mantissa loss)
   fused with ``layernorm_quantize`` inside ``DynamicInt8TransformerBlock``, and rotated dynamic INT8 on the other
   130 block linears.

Measured fidelity & eager latency on 10 MegaDepth-1500 pairs (2 per scene) plus wide-baseline ``assets/toronto``
(311.6 px max flow, RTX 5070 Ti ``sm_120``, 420x560, vs FP32 reference; eager timings updated in final table):

======================================================================  =======  ========  =======  =======  =======
Scheme (ablation / rejected)                                            ms       med EPE   <1px     <2px     cov IoU
======================================================================  =======  ========  =======  =======  =======
fp32 (upstream bit-exact reference)                                     152.97   0.000 px  100.0%   100.0%   1.0000
bf16 baseline (FP32 DPT heads, upstream default)                         51.11   0.129 px   98.95%   99.82%   0.9927
bf16 preset (BF16 cl heads + Triton upsample_bilinear)                   39.59   0.184 px   99.38%   99.85%   0.9923
int8 static (all 36 blocks) — REJECTED                                   25.86   1.825 px   25.96%   60.39%   0.9024
int8 dyn36 + static INT8 DPT40 — REJECTED (clips toronto 73.6% <2px)     18.18   0.503 px   89.67%   98.14%   0.9533
int8 unrotated dyn36 + sage-fp8-warp (prior int8 preset) — REJECTED      22.83   0.384 px   93.51%   98.59%   0.9707
int8 unrotated dyn36 (SDPA) — REJECTED                                    —      0.309 px   95.61%   98.89%   0.9789
SmoothQuant dyn-INT8 alpha=0.25 / 0.50 / 0.75 (unrot, SDPA) — REJECTED    —      0.538 / 0.405 / 7.995 px (w_kurt!)
fp8 static MXFP8 all 36 blocks — REJECTED                                26.46   0.766 px   68.64%   94.83%   0.9625
fp8 dynamic block-32 MXFP8 all 36 blocks — REJECTED                       —      0.699 px   74.19%   95.56%   0.9670
fp8 rotated static MXFP8 all 145 linears — REJECTED                       —      0.834 px   61.93%   92.11%   0.9564
fp8 unrot MXFP8 enc + dyn-INT8 info + sage (prior fp8) — REJECTED        23.18   0.459 px   89.26%   98.48%   0.9767
fp8 rotated MXFP8 enc ALL(96) / 0..21 / 0..19 + rot-INT8 — REJECTED       —      0.457 / 0.386 / 0.382 px
fp8 MXFP8 enc 0..23.fc2(24L) + rot-INT8 (MASt3R-style) — REJECTED         —      0.346 px   93.81%   99.09%   0.9823
fp8 MXFP8 enc 2..11.fc2(10L) + rot-sH64 (BF16 proj, SDPA) — REJECTED     27.61   0.250 px   98.30%   99.57%   0.9875
fp8 MXFP8 enc 2..11 ALL(40L) + rot-sH64 (BF16 proj, SDPA) — REJECTED     27.79   0.249 px   97.38%   99.35%   0.9868
fp8 MXFP8 enc 2..11.fc1(10L) + rot-sH64 (BF16 proj, SDPA) — REJECTED     25.90   0.247 px   98.36%   99.68%   0.9882
bf16 backbone + MXFP8 DPT convs (40 convs) — REJECTED                    37.27   0.756 px   65.12%   89.76%   0.9880
rot_stream only (no fc2 H, all 145 INT8, SDPA) — REJECTED                 —      0.324 px   95.24%   99.36%   0.9785
fc2 unsigned H_64 only (no stream rot, all 145, SDPA) — REJECTED          —      0.299 px   95.88%   99.29%   0.9844
rot + unsigned H_64 (all 145 INT8, SDPA) — REJECTED                       —      0.267 px   97.81%   99.52%   0.9874
rot + signed sH_64 (all 145 INT8, 0 BF16, SDPA) — REJECTED                —      0.244 px   98.16%   99.72%   0.9866
rot + sH_64 + BF16 proj + sage-fp8 enc + SDPA info — REJECTED            26.57   0.243 px   98.29%   99.64%   0.9888
rot + sH_64 + BF16 proj + sage(fp16PV) both — REJECTED                   26.75   0.253 px   98.10%   99.44%   0.9875
int8 preset (rot + sH_64, 144L dyn-INT8, BF16 proj_embed, SDPA)          27.66   0.241 px   98.73%   99.70%   0.9897
fp8 preset (rot + sH_64, 14L MXFP8 enc 2..15.fc1 + 130L INT8, SDPA)      27.36   0.252 px   97.58%   99.54%   0.9891
======================================================================  =======  ========  =======  =======  =======

Interleaved best-of-3 eager timings on RTX 5070 Ti (420x560, ``assets/toronto``): ``bf16`` 41.67 ms (41.87 ms
CUDA-graphed), ``int8`` 27.66 ms (1.51x; 27.44 ms CUDA-graphed), ``fp8`` 27.36 ms (1.52x; 25.99 ms CUDA-graphed,
1.61x vs ``bf16``). Removing SageAttention also restores CUDA-graph safety for both ``int8`` and ``fp8``.
"""

from __future__ import annotations

from pathlib import Path

import torch
from torch import Tensor, nn

from flashim.quant.rotation import hadamard, layernorm_orthogonal, signed_hadamard
from flashim.quant.scheme import (
    INT8_TOKEN_ACTIVATIONS,
    INT8_WEIGHTS,
    MXFP8,
    QuantConfig,
    QuantScheme,
)

CALIBRATION_DIR = Path(__file__).parent / "calibration"
FP8_CALIBRATION = CALIBRATION_DIR / "ufm_fp8_mega.safetensors"

# Never quantized: patch embedding, the single 1024->768 encoder->info_sharing projection (carries folded
# encoder.model.norm affine scale; protecting this 1 linear recovers -0.028 px EPE at < 0.05 ms cost), and both
# DPT prediction heads (flow magnitude is unbounded -> static conv quantization clips on wide-baseline pairs).
PROTECTED = (
    "encoder.model.patch_embed.*",
    "info_sharing.proj_embed",
    "head1.*",
    "uncertainty_head.*",
)

ENCODER_LINEARS = (
    "encoder.model.blocks.*.attn.qkv",
    "encoder.model.blocks.*.attn.proj",
    "encoder.model.blocks.*.mlp.fc1",
    "encoder.model.blocks.*.mlp.fc2",
)
INFO_BLOCK_LINEARS = (
    "info_sharing.self_attention_blocks.*.attn.qkv",
    "info_sharing.self_attention_blocks.*.attn.proj",
    "info_sharing.self_attention_blocks.*.mlp.fc1",
    "info_sharing.self_attention_blocks.*.mlp.fc2",
)
BLOCK_LINEARS = (*ENCODER_LINEARS, *INFO_BLOCK_LINEARS)

# 14 mid-encoder MLP fc1 projections [1024 -> 4096] for the hybrid fp8 preset (cuBLAS block-32 MXFP8, fused with
# layernorm_quantize inside DynamicInt8TransformerBlock). Skips blocks 0..1 (raw-patch sensitivity) and 16..23
# (final encoder representation feeding DPT & info_sharing).
FP8_MXFP8_LINEARS = tuple(f"encoder.model.blocks.{i}.mlp.fc1" for i in range(2, 16))

# Calibration-free rotated dynamic per-token INT8 across all 36 transformer blocks (144/145 backbone linears;
# info_sharing.proj_embed stays BF16).
INT8_CONFIG = QuantConfig(
    {
        "transformer": QuantScheme(
            targets=BLOCK_LINEARS, weights=INT8_WEIGHTS, input_activations=INT8_TOKEN_ACTIVATIONS
        ),
    },
    name="int8",
)

# Hybrid FP8: static MXFP8 (cuBLAS scaled_mm) on the 14 mid-encoder mlp.fc1 [1024->4096] projections + rotated
# dynamic per-token INT8 on the other 130 transformer-block linears (info_sharing.proj_embed stays BF16).
# First-match group ordering puts FP8_MXFP8_LINEARS ahead of BLOCK_LINEARS.
FP8_CONFIG = QuantConfig(
    {
        "mxfp8": QuantScheme(targets=FP8_MXFP8_LINEARS, weights=MXFP8, input_activations=MXFP8),
        "transformer_int8": QuantScheme(
            targets=BLOCK_LINEARS, weights=INT8_WEIGHTS, input_activations=INT8_TOKEN_ACTIVATIONS
        ),
    },
    name="fp8",
)

QUANT_CONFIGS = {"int8": INT8_CONFIG, "fp8": FP8_CONFIG}
CALIBRATIONS = {"fp8": FP8_CALIBRATION}


@torch.no_grad()
def rotate_ufm(
    model: nn.Module,
    *,
    enc_block: int | None = None,
    info_block: int | None = None,
    rotate_enc: bool = True,
    rotate_info: bool = True,
    rotate_v: bool = True,
) -> int:
    """Fold exact LayerNorm-commuting orthogonal rotations (:func:`~flashim.quant.rotation.layernorm_orthogonal`) into
    UFM's DINOv2 encoder and ``info_sharing`` residual streams and per-head Sylvester Hadamard rotations into
    attention ``V -> proj`` (adapted from ``model/mast3r``, commit ``7741ccc``).

    Every LayerNorm's affine weight and bias are folded into its consuming linear(s) / 1x1 DPT ``input_process``
    conv(s) first so the remaining un-affined ``LayerNorm`` satisfies ``LN_0(x Q) = LN_0(x) Q`` identically
    (``Q^T Q = I``, ``Q 1 = 1``). Exact in real arithmetic (preserves flow and covisibility) with zero runtime cost;
    must run after :func:`~flashim.quant.passes.transformer_block.fold_layer_scale`. Returns the number of rotated
    weight/embedding tensors.
    """
    dev = next(model.parameters()).device
    enc = model.encoder.model
    info = model.info_sharing
    ed = enc.cls_token.shape[-1]
    idim = info.dim
    enc_heads = enc.blocks[0].attn.num_heads
    info_heads = info.self_attention_blocks[0].attn.num_heads

    q_enc = layernorm_orthogonal(ed, block_size=enc_block, device=dev, dtype=torch.float64)
    q_info = layernorm_orthogonal(idim, block_size=info_block, device=dev, dtype=torch.float64)
    enc_vh = torch.block_diag(*([hadamard(ed // enc_heads, device=dev, dtype=torch.float64)] * enc_heads))
    info_vh = torch.block_diag(*([hadamard(idim // info_heads, device=dev, dtype=torch.float64)] * info_heads))
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
        write_conv(enc.patch_embed.proj, q_enc)
        enc.cls_token.copy_((enc.cls_token.double() @ q_enc).to(enc.cls_token.dtype))
        enc.pos_embed.copy_((enc.pos_embed.double() @ q_enc).to(enc.pos_embed.dtype))
        enc._pos_cache.clear()
        rotated.update((id(enc.cls_token), id(enc.pos_embed)))
        for blk in enc.blocks:
            fold_ln(blk.norm1, [blk.attn.qkv])
            read_lin(blk.attn.qkv, q_enc)
            write_lin(blk.attn.proj, q_enc)
            fold_ln(blk.norm2, [blk.mlp.fc1])
            read_lin(blk.mlp.fc1, q_enc)
            write_lin(blk.mlp.fc2, q_enc)
            if rotate_v:
                write_lin(blk.attn.qkv, enc_vh, slice(2 * ed, 3 * ed))
                read_lin(blk.attn.proj, enc_vh)
        enc_dpt = [
            model.head1[0][0].input_process[0][0][0],
            model.uncertainty_head[0][0].input_process[0][0][0],
        ]
        fold_ln(enc.norm, [info.proj_embed, *enc_dpt])
        read_lin(info.proj_embed, q_enc)
        for conv in enc_dpt:
            read_c1(conv, q_enc)

    if rotate_info:
        write_lin(info.proj_embed, q_info)
        info.view_pos_table.copy_((info.view_pos_table.double() @ q_info).to(info.view_pos_table.dtype))
        rotated.add(id(info.view_pos_table))
        for blk in info.self_attention_blocks:
            fold_ln(blk.norm1, [blk.attn.qkv])
            read_lin(blk.attn.qkv, q_info)
            write_lin(blk.attn.proj, q_info)
            fold_ln(blk.norm2, [blk.mlp.fc1])
            read_lin(blk.mlp.fc1, q_info)
            write_lin(blk.mlp.fc2, q_info)
            if rotate_v:
                write_lin(blk.attn.qkv, info_vh, slice(2 * idim, 3 * idim))
                read_lin(blk.attn.proj, info_vh)
        info_dpt = [
            head.input_process[idx][0][0]
            for head in (model.head1[0][0], model.uncertainty_head[0][0])
            for idx in (1, 2, 3)
        ]
        fold_ln(info.norm, info_dpt)
        for conv in info_dpt:
            read_c1(conv, q_info)

    return len(rotated)


def attach_fc2_hadamard(
    model: nn.Module, *, block_size: int = 64, signed: bool = True, seed: int = 0
) -> int:
    """Attach a block Hadamard matrix (``_flashim_block_rotate``) to each transformer block's ``mlp.fc2`` linear
    so :func:`~flashim.quant.apply.quantize_model` folds ``W @ block_diag(R)`` into its weights and uses
    :data:`flashim.ops.rotate_quantize_int8_rows` at runtime (adapted from ``model/mast3r`` commit ``7741ccc``,
    defaulting to :func:`~flashim.quant.rotation.signed_hadamard` to cancel the positive-GELU DC spike)."""
    dev = next(model.parameters()).device
    blocks = [*model.encoder.model.blocks, *model.info_sharing.self_attention_blocks]
    mat = (
        signed_hadamard(block_size, seed=seed, device=dev, dtype=torch.float32)
        if signed
        else hadamard(block_size, device=dev, dtype=torch.float32)
    )
    attached = 0
    for blk in blocks:
        fc2 = blk.mlp.fc2
        if isinstance(fc2, nn.Linear) and fc2.in_features % block_size == 0:
            fc2._flashim_block_rotate = mat
            attached += 1
    return attached
