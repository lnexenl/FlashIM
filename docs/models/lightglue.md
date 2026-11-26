# LightGlue

Sparse matcher: [LightGlue](https://github.com/cvg/LightGlue) as a native port, matching `SparseFeatures` of two images.
cvg/LightGlue trains one LightGlue per (detector, descriptor) pair, so each pairing below has its own weights, selected
with the `variant` config key (`flashim.load("lightglue", "bf16", overrides=["variant=aliked"])`): `rdd`, `aliked`,
`raco-aliked`, `superpoint`, `disk`, `xfeat`, `ripepp` (XFeat uses
verlab's LighterGlue; RIPE++ uses Fraunhofer's weakly-supervised RL weights).

## Pipelines

An extractor and its LightGlue in one model:

| pipeline | extractor | weights |
|---|---|---|
| `rdd-lightglue` | [RDD](rdd.md) | RDD's `rdd_lg.pth` |
| `aliked-lightglue` | [ALIKED](aliked.md) | cvg's `aliked_lightglue.pth` |
| `raco-aliked-lightglue` | [RaCo + ALIKED](raco-aliked-lightglue.md) | cvg's `raco_aliked_lightglue.pth` |
| `superpoint-lightglue` | SuperPoint | cvg |
| `disk-lightglue` | DISK | cvg |
| `xfeat-lightglue` | XFeat | verlab (LighterGlue) |
| `ripepp-lightglue` | RIPE++ | Fraunhofer (`ripe++lightglue.tar`, non-commercial) |

```python
pipeline = flashim.load("aliked-lightglue", "bf16")
m = pipeline.match("a.jpg", "b.jpg", resize=1024, max_matches=1024)   # Matches; 1024 keypoints per image
```

A pipeline's preset (`fp32`, `bf16`, `int8`, `fp8`; `fp8` for `lightglue` and `rdd-lightglue` by default) is every
component's, and each builds the nearest it has: `aliked-lightglue` `int8` runs ALIKED in `bf16` and LightGlue in
`int8`, `superpoint-lightglue` `int8` explicitly keeps SuperPoint in `bf16` with rotated `int8` LightGlue, and
`ripepp-lightglue` `int8` runs RIPE++ in `int8` (31/35 convs, `dec.{1,2}` BF16) with LightGlue in `bf16`. Presets
omitted because quantization lost pose AUC resolve via `build.NEAREST`: `aliked-lightglue`, `superpoint-lightglue`,
`disk-lightglue` and `ripepp-lightglue` omit `fp8` (`fp8 -> int8`); `xfeat-lightglue` omits `int8` and `fp8` (`-> bf16`). `rdd-lightglue` and
`raco-aliked-lightglue` ship all four presets. `components.extractor.preset` / `components.matcher.preset` set one
component's ([configuration](../configuration.md)).

## Quantization and speed

The `int8` / `fp8` presets use calibration tables computed on each pairing's features, shipped in
`flashim/models/lightglue/calibration/`. Regenerate them with `scripts/calibrate_lightglue.py --extractor <model>`.

- **`int8`**: folds orthogonal Hadamard rotations into weights (`rotate_residual_stream`, using Walsh-Hadamard for
  `dim=256` and Paley-12 Kronecker Hadamard for `dim=96` XFeat), applies SmoothQuant scaling to `ffn.3`, and fuses the
  transformer glue (`fused_int8_glue`).
- **`fp8` / shared passes**: `low_precision_similarity` (`ops.matmul_nt_fp32`), batched `CrossBlock.to_out`, `fused_bias`
  in `fp8`, and repeat-shape `cuda_graph` capture (triggers at `hits >= 2` for `max(N0, N1) <= 8192`, releasing pools on
  fallback).

Matcher and pair latencies of every pairing (`eager / compile`, 4096 keypoints per image, RTX 5070 Ti `sm_120`) are in the
[inference benchmark](../benchmarks.md#sparse-matchers): for example `rdd-lightglue` `match_features` takes 11.7 / 13.5 ms in
`bf16`, 9.89 / 10.1 ms in `int8` and 10.3 / 17.9 ms in `fp8`.

## Why int8 loses accuracy for some pairings

Most pairings keep their pose AUC under `int8`; two did not, for different reasons (MegaDepth-1500, 3 seeds, RTX 5070 Ti
`sm_120`, `eager`, cached features, paired bootstrap against the `bf16` extractor + `bf16` matcher):

- **`ripepp-lightglue`** (`-0.0075` with the extractor and the matcher in `int8`): two causes that add up.
  - *The extractor, not LightGlue, dominates.* `int8` extractor + `bf16` matcher is `-0.0060 [-0.0106, -0.0015]`; swapping only
    the descriptors (`bf16` keypoints + `int8` descriptors) is `-0.0011`, swapping only the keypoints is `-0.0072`. The loss
    is the sub-pixel keypoint localization: RIPE++'s 3x3 softmax refinement is sensitive to the four small
    `net.decoder.layers.{1,2}.*` convolutions in `int8` (keypoints within 0.5 px of the `bf16` ones: 0.903 -> 0.952 once
    they stay in `bf16`). A plain mutual-nearest-neighbour match hides it (`ripepp` `int8` with MNN, before the exclusion,
    was `-0.0001`), a trained matcher does not.
  - *The matcher is more sensitive than the others.* Its weakly supervised (reinforcement-learning) layers 4-8 and assignment
    head work at extreme logit temperatures (mean absolute error of the similarity matrix under `int8` 36.85, against 0.58
    for `rdd-lightglue`) and its confidences saturate, so `int8` LightGlue alone costs `-0.0034` (it is not channel or token
    outliers, not a calibration mismatch and not a shift of `filter_threshold`). `int8` LightGlue therefore stays `bf16`
    in this pairing.
  - Shipped: `int8` runs RIPE++ in `int8` with `net.decoder.layers.{1,2}.*` in `bf16` and LightGlue in `bf16`:
    `-0.0011 [-0.0052, +0.0030]`, 52.6 ms per pair against 79.4 ms in `bf16`.
- **`xfeat-lightglue`** (`-0.0074`): the `int8` extractor loses nothing (`+0.0041`); the 6-layer, 96-dimensional matcher does.
  The Paley-12 Hadamard rotation removes its 8x channel outliers but creates per-token outliers (8.85x and 7.35x) in the
  attention output projections of layers 4 and 5, which shifts the similarity and matchability scores (mean confidence
  0.707 -> 0.654). Per-token dynamic `int8` restores the accuracy (`-0.0024` / `-0.0029`) but is slower than `bf16`
  (3.67 against 3.49 ms), so the pairing stays `bf16`.

When an `int8` pairing loses accuracy, split the loss into extractor and matcher precision first (the grid above), then
look at the keypoint localization and at the similarity/matchability error of the matcher.

## Accuracy

MegaDepth-1500 and ScanNet-1500 paired-bootstrap gate results on RTX 5070 Ti (`sm_120`, `eager`, 3 seeds)
for every LightGlue pairing (`0 FAIL`, including `+0.0126` / `+0.0073` on `aliked-lightglue bf16` and `+0.0186` /
`+0.0122` on `raco-aliked-lightglue bf16` from restoring FP32 SDDH coordinates) are in
[benchmarks.md](../benchmarks.md#accuracy) and
[platforms.md](../platforms.md).
