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
`int8`. `components.extractor.preset` / `components.matcher.preset` set one component's
([configuration](../configuration.md)).

## Quantization

The `int8` / `fp8` presets use calibration tables computed on each pairing's features, shipped in
`flashim/models/lightglue/calibration/`. Regenerate them with `scripts/calibrate_lightglue.py --extractor <model>`.

## Accuracy

MegaDepth-1500 AUC@5 (unified protocol, 4 seeds; ALIKED pipelines with the `fp32` extractor,
`components.extractor.preset=fp32`):

| pipeline | `fp32` | `bf16` | `int8` | `fp8` |
|---|---|---|---|---|
| `aliked-lightglue` | 0.627 | 0.627 | 0.620 | 0.613 |
| `raco-aliked-lightglue` | | 0.640 | 0.638 | 0.634 |

The AUC change of the SuperPoint, DISK and XFeat pairings is in the
[inference benchmark](../benchmarks.md#accuracy); the `int8` / `fp8` presets of `superpoint-lightglue` and
`xfeat-lightglue` fail the accuracy gate there.
