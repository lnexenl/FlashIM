"""UFM quantization schemes, protected modules, calibration paths and measured evidence.

Architecture: 24-layer DINOv2 ViT-L/14 encoder (``dim=1024``) -> 12-layer global-attention ``info_sharing``
(``dim=768``) -> two DPT heads (``head1`` for dense flow, ``uncertainty_head`` for covisibility). All 36
transformer blocks are :class:`flashim.layers.TransformerBlock` instances.

Measured fidelity & latency on 10 MegaDepth-1500 pairs (2 per scene) plus wide-baseline ``assets/toronto``
(311.6 px max flow, RTX 5070 Ti ``sm_120``, working resolution 420x560, eager forward, vs FP32 reference):

=============================================================  =======  ========  =======  =======  =======
Scheme (ablation)                                              ms       med EPE   <1px     <2px     cov IoU
=============================================================  =======  ========  =======  =======  =======
fp32 (upstream bit-exact reference)                            152.97   0.000 px  100.0%   100.0%   1.0000
bf16 baseline (FP32 DPT heads, upstream default)                51.11   0.129 px   98.95%   99.82%   0.9927
bf16 preset (BF16 cl heads + Triton upsample_bilinear)          39.59   0.184 px   99.38%   99.85%   0.9923
int8 static (all 36 blocks) — REJECTED                          25.86   1.825 px   25.96%   60.39%   0.9024
int8 dyn36 + static INT8 DPT40 — REJECTED (clips large flow)    18.18   0.503 px   89.67%   98.14%   0.9533
fp8 static MXFP8 (all 36 blocks) — REJECTED                     26.46   0.766 px   68.64%   94.83%   0.9625
fp8 dynamic block-32 MXFP8 (all 36 blocks) — REJECTED            —      0.699 px   74.19%   95.56%   0.9670
bf16 backbone + MXFP8 DPT convs (40 convs) — REJECTED           37.27   0.756 px   65.12%   89.76%   0.9880
int8 preset (dyn-INT8 36 blk + sage + BF16 heads+up)            22.83   0.384 px   93.51%   98.59%   0.9707
fp8 preset (MXFP8 enc + dyn-INT8 info + sage + BF16 heads+up)   23.18   0.459 px   89.26%   98.48%   0.9767
=============================================================  =======  ========  =======  =======  =======

Root causes diagnosed by activation outlier analysis, stage ablation and wide-baseline stress testing:
- DINOv2 encoder has massive token+channel outliers (``blocks.0.attn.qkv`` channel ratio 712x; ``blocks.23.mlp.fc2``
  token ratio 23x, channel ratio 292x): per-tensor static INT8 breaks it (``<1px`` 26%), whereas per-token dynamic
  INT8 (``INT8_TOKEN_ACTIVATIONS``, calibration-free) and block-32 static MXFP8 both preserve accuracy (``<1px`` > 91%).
- ``info_sharing`` has clean activations (token ratio 1.6-1.8x), yet BOTH static and dynamic block-32 MXFP8 degrade
  flow EPE to 0.70-0.77 px because E4M3's 3-bit mantissa loses sub-pixel precision in cross-view coordinate
  regression; dynamic INT8 (7-bit mantissa) preserves it. The ``fp8`` preset therefore uses static MXFP8 on the 24
  encoder blocks and dynamic INT8 on the 12 ``info_sharing`` blocks.
- DPT heads stay BF16 (channels-last + Triton ``ops.upsample_bilinear``, cutting heads from 21.85 ms FP32 -> 8.34 ms):
  static INT8 on ``head1`` DPT convs clips on wide-baseline pairs whose flow exceeds MegaDepth's calibration range
  (``assets/toronto``, max flow 311.6 px: ``p90=17.70 px, <2px=73.6%`` with static INT8 DPT vs ``p90=0.79 px,
  <2px=99.93%`` with BF16 DPT), and static INT8 on ``uncertainty_head`` drops ``sacre_coeur`` covisibility IoU from
  0.960 -> 0.936; MXFP8 DPT convs are both slower (7.83 ms) and worse (0.76 px EPE).
"""

from __future__ import annotations

from pathlib import Path

from flashim.quant.scheme import (
    INT8_TOKEN_ACTIVATIONS,
    INT8_WEIGHTS,
    MXFP8,
    QuantConfig,
    QuantScheme,
)

CALIBRATION_DIR = Path(__file__).parent / "calibration"
FP8_CALIBRATION = CALIBRATION_DIR / "ufm_fp8_mega.safetensors"

# Never quantized: patch embedding and both DPT prediction heads (flow magnitude is unbounded -> static conv
# quantization clips on wide-baseline pairs; see module docstring).
PROTECTED = (
    "encoder.model.patch_embed.*",
    "head1.*",
    "uncertainty_head.*",
)

ENCODER_LINEARS = (
    "encoder.model.blocks.*.attn.qkv",
    "encoder.model.blocks.*.attn.proj",
    "encoder.model.blocks.*.mlp.fc1",
    "encoder.model.blocks.*.mlp.fc2",
)
INFO_LINEARS = (
    "info_sharing.proj_embed",
    "info_sharing.self_attention_blocks.*.attn.qkv",
    "info_sharing.self_attention_blocks.*.attn.proj",
    "info_sharing.self_attention_blocks.*.mlp.fc1",
    "info_sharing.self_attention_blocks.*.mlp.fc2",
)
TRANSFORMER_LINEARS = (*ENCODER_LINEARS, *INFO_LINEARS)

# Calibration-free dynamic per-token INT8 across all 36 transformer blocks + proj_embed (145 linears).
INT8_CONFIG = QuantConfig(
    {
        "transformer": QuantScheme(
            targets=TRANSFORMER_LINEARS, weights=INT8_WEIGHTS, input_activations=INT8_TOKEN_ACTIVATIONS
        ),
    },
    name="int8",
)

# Hybrid FP8: static MXFP8 on the 24 DINOv2 encoder blocks (96 linears) + dynamic INT8 on info_sharing (49 linears).
FP8_CONFIG = QuantConfig(
    {
        "encoder": QuantScheme(targets=ENCODER_LINEARS, weights=MXFP8, input_activations=MXFP8),
        "info_sharing": QuantScheme(
            targets=INFO_LINEARS, weights=INT8_WEIGHTS, input_activations=INT8_TOKEN_ACTIVATIONS
        ),
    },
    name="fp8",
)

QUANT_CONFIGS = {"int8": INT8_CONFIG, "fp8": FP8_CONFIG}
CALIBRATIONS = {"fp8": FP8_CALIBRATION}
