# LightGlue

Sparse matcher: [LightGlue](https://github.com/cvg/LightGlue) as a native port, matching `SparseFeatures` of two images.
cvg/LightGlue trains one LightGlue per (detector, descriptor) pair, so each pairing below has its own weights, selected
with the `variant` config key (`flashim.load("lightglue", "bf16", overrides=["variant=aliked"])`): `rdd`, `aliked`,
`raco-aliked`, `superpoint`, `disk`, `xfeat` (XFeat uses
verlab's LighterGlue).

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

```python
pipeline = flashim.load("aliked-lightglue", "bf16")
m = pipeline.match("a.jpg", "b.jpg", resize=1024, max_matches=1024)   # Matches; 1024 keypoints per image
```

A pipeline's preset (`fp32`, `bf16`, `int8`, `fp8`; `fp8` for `lightglue` and `rdd-lightglue` by default) is every
component's, and each builds the nearest it has: `aliked-lightglue` `int8` runs ALIKED in `bf16` and LightGlue in
`int8`, and `superpoint-lightglue` `int8` explicitly keeps SuperPoint in `bf16` with rotated `int8` LightGlue. Presets
omitted because quantization lost pose AUC resolve via `build.NEAREST`: `aliked-lightglue`, `superpoint-lightglue` and
`disk-lightglue` omit `fp8` (`fp8 -> int8`); `xfeat-lightglue` omits `int8` and `fp8` (`-> bf16`). `rdd-lightglue` and
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

Measured `lightglue` matcher latencies on **RTX 5070 Ti (`sm_120`, `eager`)**:

| variant | keypoints | `fp8` (base -> new) | `int8` (base -> new) |
|---|---|---:|---:|
| `rdd` | N = 1024 | `7.09 -> 1.56 ms` (4.55x) | `6.29 -> 1.48 ms` (4.25x) |
| `aliked` | N = 2048 | `7.07 -> 3.44 ms` (2.06x) | `6.03 -> 3.36 ms` (1.79x) |
| `rdd` | N = 4096 | `13.01 -> 10.24 ms` (1.27x) | `12.60 -> 9.90 ms` (1.27x) |

## Accuracy

Historical MegaDepth-1500 AUC@5 (2026-10 unified protocol, 4 seeds, RTX 5070 Ti `sm_120`, `eager`; ALIKED pipelines with
`components.extractor.preset=fp32`):

| pipeline | `fp32` | `bf16` | `int8` | `fp8` |
|---|---|---|---|---|
| `aliked-lightglue` | 0.627 | 0.627 | 0.620 | 0.613 |
| `raco-aliked-lightglue` | | 0.640 | 0.638 | 0.634 |

Final 2026-11 MegaDepth-1500 and ScanNet-1500 paired-bootstrap gate results on RTX 5070 Ti (`sm_120`, `eager`, 3 seeds)
for every LightGlue pairing (`0 FAIL`, including `+0.0126` / `+0.0073` on `aliked-lightglue bf16` and `+0.0186` /
`+0.0122` on `raco-aliked-lightglue bf16` from restoring FP32 SDDH coordinates) are in
[benchmarks.md](../benchmarks.md#accuracy-of-the-pre-2026-11-models-2026-11-gate-sm_120) and
[platforms.md](../platforms.md).
