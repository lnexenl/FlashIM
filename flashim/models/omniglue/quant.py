"""Quantization schemes and measured fidelity/speed evidence for OmniGlue (extractor + matcher).

Measured on 12 MegaDepth pairs (disjoint calibration split, ``thresh=0.02``, RTX 5070 Ti ``sm_120``,
eager mode, compared against FP32 matches):

**Matcher (on shared FP32 features, 18 GNN layers + 100 Sinkhorn iterations)**:
- ``bf16``: ``agree=0.9677``, ``ratio=0.9908`` (baseline eager ``9.27 ms`` before kernel opts)
- ``dynamic-int8`` (per-token activations on GNN linears, zero calibration table): ``agree=0.9354``,
  ``ratio=1.0116``. Per-token activation outlier ratios across GNN leaves are ``token_max/med = 3.0–4.3``
  (vs ``channel_max/med = 2.7–4.5``), so per-token dynamic scaling recovers most of the gap over static INT8.
- ``static-int8`` (rejected): ``agree=0.8708``, ``ratio=1.0207``
- ``rotated+smooth static-int8`` (rejected): ``agree=0.8694``, ``ratio=0.9996``
- ``static-mxfp8`` (rejected): ``agree=0.7905``, ``ratio=1.0025`` (E4M3 3-bit mantissa noise compounds
  across 18 GNN layers + 100 Sinkhorn iterations; ``omniglue-matcher`` therefore omits ``fp8`` so
  ``omniglue`` ``fp8`` degrades the matcher to ``int8`` via ``build.NEAREST``).

**Extractor (SuperPoint ``sp_v6`` + DINOv2 ViT-B/14, matched with FP32 matcher)**:
- ``bf16`` (``sp_v6`` channels-last): ``agree=0.7641``, ``ratio=1.0562`` (-2.74 ms vs NCHW on 1600 px)
- ``int8`` (dynamic per-token INT8 on DINOv2 ViT-B/14 + BF16 ``sp_v6``, zero calibration table):
  ``agree=0.7657``, ``ratio=1.0625`` (matches BF16; quantizing ``sp_v6`` convs dropped agreement to ``0.6855``
  and im2col INT8 convs are slower than BF16 on small VGG nets)
- ``fp8`` (calibrated static MXFP8 on DINOv2 ViT-B/14 + BF16 ``sp_v6``):
  ``agree=0.7653``, ``ratio=0.8891`` (quantizing ``sp_v6`` convs dropped agreement to ``0.6792``).

**End-to-end MegaDepth-1500 fidelity (``benchmarks/accuracy/fidelity.py --models omniglue --pairs-per-scene 2``,
10 pairs across all 5 scenes, 2 px tolerance vs FP32)**:
- ``bf16``: ``kp_overlap=0.9750``, ``desc_cos=0.9998``, ``match_agree=0.9003``, ``match_ratio=0.9962``
- ``int8``: ``kp_overlap=0.9750``, ``desc_cos=0.9998``, ``match_agree=0.8838``, ``match_ratio=1.0112``
- ``fp8`` (matcher degrades to ``int8``): ``kp_overlap=0.9750``, ``desc_cos=0.9998``, ``match_agree=0.8860``,
  ``match_ratio=1.0123``.
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

# -- Extractor (SuperPoint sp_v6 + DINOv2 ViT-B/14) --------------------------------------------------------

VIT_LINEARS = (
    "dinov2.blocks.*.attn.qkv",
    "dinov2.blocks.*.attn.proj",
    "dinov2.blocks.*.mlp.fc1",
    "dinov2.blocks.*.mlp.fc2",
)
# SuperPoint sp_v6 convs stay BF16: quantizing them drops match agreement by ~8 pp (0.764 -> 0.686 / 0.679)
# and im2col INT8/MXFP8 convs are slower than cuDNN BF16 channels-last on small 64/128-ch VGG nets.
EXTRACTOR_PROTECTED = (
    "superpoint.*",
    "dinov2.patch_embed.*",
)
EXTRACTOR_QUANT_CONFIGS = {
    "int8": QuantConfig(
        {"vit": QuantScheme(VIT_LINEARS, INT8_WEIGHTS, INT8_TOKEN_ACTIVATIONS)},
        name="int8",
    ),
    "fp8": QuantConfig(
        {"vit": QuantScheme(VIT_LINEARS, MXFP8, MXFP8)},
        name="fp8",
    ),
}

# -- Matcher (KeypointEncoder + 18 AttentionalPropagation layers + final_proj) -----------------------------

MATCHER_LINEARS = (
    "gnn.layers.*.attn.qk_proj",
    "gnn.layers.*.attn.q_proj",
    "gnn.layers.*.attn.k_proj",
    "gnn.layers.*.attn.v_proj",
    "gnn.layers.*.attn.out_proj",
    "gnn.layers.*.mlp.0",
    "gnn.layers.*.mlp.3",
)
MATCHER_PROTECTED = (
    "kenc.*",
    "final_proj",
)

# Dynamic per-token INT8 on GNN linears (no calibration table needed; agree=0.9354 vs 0.8708 static INT8).
MATCHER_QUANT_CONFIGS = {
    "int8": QuantConfig(
        {"gnn": QuantScheme(MATCHER_LINEARS, INT8_WEIGHTS, INT8_TOKEN_ACTIVATIONS)},
        name="int8",
    ),
}
