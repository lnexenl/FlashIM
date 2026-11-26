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
[inference benchmark](../benchmarks.md#sparse-matchers): for example `rdd-lightglue` `match_features` takes 11.7 / 11.1 ms in
`bf16`, 9.93 / 9.98 ms in `int8` and 10.3 / 18.3 ms in `fp8`.

## Accuracy

MegaDepth-1500 and ScanNet-1500 paired-bootstrap gate results on RTX 5070 Ti (`sm_120`, `eager`, 3 seeds)
for every LightGlue pairing (`0 FAIL`, including `+0.0126` / `+0.0073` on `aliked-lightglue bf16` and `+0.0186` /
`+0.0122` on `raco-aliked-lightglue bf16` from restoring FP32 SDDH coordinates) are in
[benchmarks.md](../benchmarks.md#accuracy) and
[platforms.md](../platforms.md).
