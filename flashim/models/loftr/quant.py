"""LoFTR & MINIMA-LoFTR quantization targets, protected modules, and measured scheme selection.

``loftr`` (outdoor / indoor) keeps its original uniform static INT8 / static MXFP8 ``extractor_spec`` presets (bit-
identical to upstream FlashIM). ``minima-loftr`` uses the diagnosed, accuracy-recovered presets below.

Outlier & weight-error diagnosis on ``minima-loftr`` (MegaDepth-1500, 840 px, BF16 vs FP32):

- **Backbone**: ``backbone.layer1.0.conv1`` has an extreme activation channel outlier ratio (``A_chR = 4.6e7``,
  ``Akurt = 84.5``, ``Wkurt = 81.6``, INT8 weight rel-L2 ``0.0246`` vs ``0.0073-0.0129`` elsewhere), causing over half
  of the entire 21-conv backbone INT8 loss (``-0.0081`` of ``-0.0159``). In MXFP8, E4M3's 3-bit mantissa increases
  weight rel-L2 2.2-3.6x (``0.0264-0.0267`` vs ``0.0073-0.0129`` in INT8), hitting ``layer3`` (``-0.0135`` vs
  ``-0.0040`` INT8) and ``layer[123]_outconv`` (``-0.0149`` vs ``-0.0016`` INT8), while ``layer1.1`` (``-0.0028``),
  ``layer2`` (``-0.0038``), and FPN ``outconv2`` (``+0.0000``) stay nearly lossless in MXFP8.
- **Coarse transformer** (``loftr_coarse``, 48 linears): ``q/k/v_proj`` (``+0.0009``), ``merge`` (``+0.0003``), and
  ``mlp.2`` (``-0.0022``) are essentially lossless in dynamic per-token INT8, whereas ``mlp.0`` alone causes 73% of
  the coarse INT8 loss (``-0.0172`` of ``-0.0237``) due to ``11.3-15.3x`` channel magnitude disparity across the
  concatenated input ``cat([x, norm1(msg)])`` (growing residual ``x`` next to unit-variance ``norm1(msg)``). In MXFP8,
  ``mlp.0`` (``-0.0167``) and post-ReLU ``mlp.2`` (``Akurt = 36.1``, ``-0.0197``) degrade sharply, whereas ``layers.0-5``
  ``q/k/v_proj`` and ``merge`` stay accurate.
- **Fine transformer** (``loftr_fine``, 12 linears): lossless (``0.0000``) in both dynamic INT8 and static MXFP8.

Measured scheme selection on MegaDepth-1500 (20 pairs = 5 scenes x 4 pairs/scene, 840 px, match agreement within 2 px
vs ``fp32``; SmoothQuant scales calibrated strictly on ``data/quantization/calibration_local.jsonl``, 24 pairs; eager
latency on RTX 5070 Ti / sm_120, ``sacre_coeur1/2`` 840 px):

- Baselines (pre-recovery, fused passes):
  - ``bf16``: ``fidelity.py`` ``0.9218`` (20-pair clean ``0.9707``, RGB-Depth demo ``0.9371``), 25.69 ms best / 26.13 ms
    med (vs ``fp32`` 63.03 ms).
  - Old ``int8`` (all 21 static INT8 convs + 60 dynamic INT8 linears): ``fidelity.py`` ``0.8933`` (``-0.0285`` vs
    ``bf16``; clean ``0.9414``, demo ``0.8639``), 14.89 ms.
  - Old ``fp8`` (all 81 static MXFP8): ``fidelity.py`` ``0.8586`` (``-0.0632`` vs ``bf16``; clean ``0.9062``, demo
    ``0.8520``), 16.36 ms.
- Algorithmic probes on ``loftr_coarse`` only (rest BF16; ``bf16 = 0.9707``):
  - Weight-only INT8 ``0.9687`` (``-0.0021``); weight-only MXFP8 ``0.9621`` (``-0.0087``).
  - Unsmoothed INT8-static ``0.9437`` (``-0.0272``); INT8-dynamic ``0.9472`` (``-0.0237``); MXFP8-static ``0.9330``
    (``-0.0378``).
  - Online block Hadamard on INT8-dynamic: ``H_32`` ``0.9575`` (``-0.0133``), ``H_64`` ``0.9584`` (``-0.0124``),
    ``H_256`` ``0.9541`` (``-0.0167``).
  - SmoothQuant on ``mlp.0`` only (zero runtime cost, fused into ``ops.layernorm_cat_quantize_int8_rows``):
    ``alpha=0.35`` ``0.9635``, ``0.45`` ``0.9649``, ``0.50`` ``0.9654``, ``0.55`` ``0.9657`` (``-0.0050``),
    ``0.65`` ``0.9656``.
  - SmoothQuant ``alpha=0.50`` on ``mlp.0`` + zero-cost per-head ``H_32`` on ``v_proj -> merge``: ``0.9669``
    (``-0.0038``); + online ``H_64`` on ``mlp.2``: ``0.9647`` (no gain over ``mlp.0`` SQ + ``H_32``).
  - SmoothQuant ``alpha=0.50`` on all 48 coarse linears: ``0.9677`` (``-0.0030``, adds runtime scaling to ``qkv``/``merge``).
  - Pure MXFP8 on coarse + SmoothQuant ``alpha=0.50`` on ``mlp.0``: ``0.9333`` (``+0.0003`` — E4M3 mantissa bound).
- End-to-end ``int8`` candidates (sweep unfused timing, selected preset fused timing):
  - ``int8_all81_SQ_mlp0`` (81/81 quantized): clean ``0.9508`` (``-0.0199``), 17.14 ms unfused.
  - ``int8_all81_SQ_coarse`` (81/81 quantized): clean ``0.9526`` (``-0.0181``), 17.32 ms unfused.
  - **Selected ``int8``** (80/81 quantized = 20 static INT8 convs + 60 dynamic INT8 linears with fused SmoothQuant
    ``alpha=0.55`` on coarse ``mlp.0`` + zero-cost per-head ``H_32`` on coarse ``v_proj -> merge``; only 1 conv
    ``backbone.layer1.0.conv1`` stays NHWC BF16): ``fidelity.py`` ``0.9071`` (``-0.0147`` vs ``bf16``, ``+0.0138`` vs
    old ``int8``; clean ``0.9561``, RGB-Depth demo ``0.9116`` = ``+0.0477``), 16.00 ms best / 16.38 ms med fused
    (1.61x vs ``bf16``).
  - ``int8_drop_l10_SQ_mlp0`` (79/81 quantized, both ``layer1.0`` convs BF16): clean ``0.9586`` (``+0.0025`` for
    +1.2 ms slower — rejected in favor of dropping only ``layer1.0.conv1``).
- End-to-end ``fp8`` candidates (sweep unfused timing, selected preset fused timing):
  - Pure MXFP8 (all 81) + SmoothQuant ``mlp.0``: clean ``0.9084`` (``-0.0623``), 17.95 ms unfused — rejected.
  - ``fp8_mixA_all81`` (55 MXFP8 + 26 INT8): clean ``0.9442`` (``-0.0265``), 18.29 ms unfused — rejected.
  - ``fp8_mixB_79`` (55 MXFP8 + 24 INT8 + ``layer1.0`` BF16): clean ``0.9504`` (``-0.0203``), 19.50 ms unfused.
  - ``fp8_mixC_80`` (55 MXFP8 + 25 INT8 + ``layer1.0.conv1`` BF16): clean ``0.9511`` (``-0.0196``), 18.90 ms unfused.
  - **Selected ``fp8``** (80/81 quantized: 47 MXFP8 on ``layer1.1``/``layer2``/``outconv2``/``coarse.0-5.(qkv,merge)``/
    ``fine.*`` + 9 static INT8 backbone convs on ``layer1.0.conv2``/``layer3``/``outconv`` + 24 dynamic INT8 coarse
    linears on ``coarse.6-7.(qkv,merge)``/``mlp.0``(SmoothQuant)/``mlp.2`` + zero-cost ``H_32`` on ``coarse.6-7``
    ``v_proj -> merge``; only 1 conv ``backbone.layer1.0.conv1`` stays NHWC BF16): ``fidelity.py`` ``0.9033``
    (``-0.0185`` vs ``bf16``, ``+0.0447`` over old ``fp8``; clean ``0.9521``, RGB-Depth demo ``0.8912`` = ``+0.0392``),
    17.24 ms best / 17.82 ms med fused (1.49x vs ``bf16``).
"""

from __future__ import annotations

import torch
from torch import nn

from flashim.quant.rotation import hadamard
from flashim.quant.scheme import (
    INT8_ACTIVATIONS,
    INT8_TOKEN_ACTIVATIONS,
    INT8_WEIGHTS,
    MXFP8,
    QuantConfig,
    QuantScheme,
)

_LINEARS = ("q_proj", "k_proj", "v_proj", "merge", "mlp.0", "mlp.2")
TRANSFORMER_LINEARS = tuple(
    f"{stage}.layers.*.{leaf}" for stage in ("loftr_coarse", "loftr_fine") for leaf in _LINEARS
)
QUANTIZED = (
    "backbone.layer[123].[01].conv[12]",
    "backbone.layer[123].[01].downsample.0",
    "backbone.layer[123]_outconv",
    "backbone.layer[12]_outconv2.[03]",
    *TRANSFORMER_LINEARS,
)
PROTECTED = ("backbone.conv1", "fine_preprocess.*")

# Only 1 extreme-outlier conv (A_chR = 4.6e7, Wkurt = 81.6) stays NHWC BF16 in minima-loftr int8 / fp8 (80/81 quantized).
MINIMA_IGNORE = ("backbone.layer1.0.conv1",)

# Mixed-precision ``fp8`` targets for minima-loftr:
# - 47 MXFP8 layers: 11 high-res backbone convs (layer1.1, layer2, FPN outconv2) + 24 coarse layers 0-5 q/k/v/merge
#   + all 12 fine transformer linears (cuBLAS block-scaled MXFP8 / Triton mxfp8_conv2d).
# - 33 INT8 layers (applied via ``fp8_mixed_int8``): 9 static INT8 backbone convs (layer1.0.conv2, layer3, 1x1 outconvs)
#   + 24 dynamic per-token INT8 coarse linears (layers 6-7 q/k/v/merge, all mlp.0 with fused SmoothQuant, all mlp.2).
MINIMA_FP8_MXFP8 = (
    "backbone.layer1.1.conv[12]",
    "backbone.layer2.[01].conv[12]",
    "backbone.layer2.0.downsample.0",
    "backbone.layer[12]_outconv2.[03]",
    "loftr_coarse.layers.[012345].[qkv]_proj",
    "loftr_coarse.layers.[012345].merge",
    *(f"loftr_fine.layers.*.{leaf}" for leaf in _LINEARS),
)
MINIMA_FP8_INT8_CONVS = (
    "backbone.layer1.0.conv2",
    "backbone.layer3.[01].conv[12]",
    "backbone.layer3.0.downsample.0",
    "backbone.layer[123]_outconv",
)
MINIMA_FP8_INT8_LINEARS = (
    "loftr_coarse.layers.[67].[qkv]_proj",
    "loftr_coarse.layers.[67].merge",
    "loftr_coarse.layers.*.mlp.0",
    "loftr_coarse.layers.*.mlp.2",
)

MINIMA_QUANT_CONFIGS: dict[str, QuantConfig] = {
    "int8": QuantConfig(
        {
            "dynamic": QuantScheme(TRANSFORMER_LINEARS, INT8_WEIGHTS, INT8_TOKEN_ACTIVATIONS),
            "conv": QuantScheme(QUANTIZED[:4], INT8_WEIGHTS, INT8_ACTIVATIONS),
        },
        ignore=MINIMA_IGNORE,
        name="int8",
    ),
    "fp8": QuantConfig(
        {"mxfp8": QuantScheme(MINIMA_FP8_MXFP8, MXFP8, MXFP8)},
        ignore=MINIMA_IGNORE,
        name="fp8",
    ),
}

MINIMA_FP8_MIXED_INT8_CONFIG = QuantConfig(
    {
        "dynamic": QuantScheme(MINIMA_FP8_INT8_LINEARS, INT8_WEIGHTS, INT8_TOKEN_ACTIVATIONS),
        "conv": QuantScheme(MINIMA_FP8_INT8_CONVS, INT8_WEIGHTS, INT8_ACTIVATIONS),
    },
    ignore=MINIMA_IGNORE,
    name="fp8_mixed_int8",
)


def rotate_v_merge_hadamard(model: nn.Module, *, layer_indices: tuple[int, ...] | None = None) -> int:
    """Fold exact per-head Sylvester Hadamard ``H_32`` into ``loftr_coarse`` ``v_proj`` (output) and ``merge`` (input).

    Exact in real arithmetic because ELU+1 linear attention is linear in ``V`` (zero runtime overhead). Only applied to
    layers whose ``merge`` is dynamically quantized (leaves calibrated static MXFP8 ``merge`` inputs untouched).
    """
    coarse = getattr(model, "loftr_coarse", None)
    if coarse is None:
        return 0
    layers = list(coarse.layers)
    indices = range(len(layers)) if layer_indices is None else layer_indices
    if not indices:
        return 0
    sample = layers[0].v_proj.weight
    nhead, dim = layers[0].nhead, layers[0].dim
    h = hadamard(dim, device=sample.device, dtype=torch.float64)
    bh = torch.block_diag(*([h] * nhead))
    count = 0
    for idx in indices:
        layer = layers[idx]
        if isinstance(layer.v_proj, nn.Linear) and isinstance(layer.merge, nn.Linear):
            layer.v_proj.weight.data.copy_((bh @ layer.v_proj.weight.data.double()).to(layer.v_proj.weight.dtype))
            layer.merge.weight.data.copy_((layer.merge.weight.data.double() @ bh).to(layer.merge.weight.dtype))
            count += 1
    return count
