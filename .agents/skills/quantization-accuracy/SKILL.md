---
name: quantization-accuracy
description: Use when an int8 or fp8 preset of a FlashIM model loses accuracy (lower match agreement, fidelity, or pose AUC than bf16), or when choosing how to quantize a new model. Diagnose the outlier type, pick the scheme, and decide when to drop a preset.
---

# quantization-accuracy

A quantized preset that builds is not an accepted preset. When it loses accuracy, change the scheme before falling back
to BF16 or protecting a layer; protecting layers is the last resort and needs the tried alternatives on record.

## Diagnose first

1. Quantize one region at a time (encoder blocks by depth, decoder, heads, individual convs) and measure the loss per
   region with `benchmarks/accuracy/fidelity.py` (match agreement vs FP32 on a few MegaDepth pairs, GPU lock held).
2. Classify the outliers on calibration activations:
   - per-token (a few tokens with massive activations; DINOv2 / CroCo ViTs) -> per-token dynamic INT8:
     `DynamicInt8Linear` (`quant/methods/int8_linear.py`), `quant/passes/dynamic_transformer_block.py`;
   - per-channel (a few channels; matchers without LayerNorm in the residual stream, concatenated inputs) ->
     Hadamard or LayerNorm-commuting rotation folded into weights (`quant/rotation.py`, `ops.rotate_quantize_int8_rows`)
     and SmoothQuant channel scales (`Int8CalibrationTable.smooth`, `ops.quantize_int8(channel_scale=)`);
   - weight outliers, or E4M3's 3-bit mantissa (small conv nets, every ViT block alike) -> INT8 for those layers: the fp8
     preset may mix MXFP8 and INT8 per layer (`xoftr`, `ufm`, `minima-loftr`);
   - a residual stream that peaks far above BF16's resolution (CroCo: ~350-650 where the BF16 ULP is 4) -> FP16 compute
     (`mast3r`, `dust3r`).
3. Attention: SageAttention-style INT8 QK / FP8 PV vs BF16 attention, per block range (`loma` `vit_attention` layers).

## Decide

- Judge variants by fidelity vs FP32 first (sharper than pose AUC, which only resolves about +-0.005 on 1500 pairs),
  measured with the real kernels: emulated quantization overstates gains. `benchmarks/accuracy/fidelity.py` reports the
  2 px match agreement for pair matchers and extractors, and flow EPE / `<2px` / covisibility IoU for dense flow models
  (`ufm`); say which metric a number is. For `mast3r` / `dust3r` 2 px agreement is meaningless: use epipolar error.
- Record every scheme tried with its accuracy and ms in the model's `quant.py`, rejected ones included, and the reason a preset
  was dropped there and in the preset notes of `docs/models.md`.
- Target: within about 0.02 fidelity of BF16 and faster than BF16. If a preset ends up strictly worse than the next one
  down (`fp8` vs `int8`), drop it from the spec so `build.NEAREST` builds the better one; for a pipeline omit it from
  `_PIPELINE_PRESETS` (`superpoint-superglue`: SuperPoint `conv1b` has 41x channel outliers; `omniglue`: DINOv2 ViT-B under
  MXFP8).
- Small conv nets (channels < 128) lose to BF16 on speed too: keep them BF16 (`tiny-roma`, `superpoint`, `xfeat`).
- The acceptance test is task-level: `verify-change` (AUC@5 drop < 0.005 vs BF16 with a paired bootstrap CI).
