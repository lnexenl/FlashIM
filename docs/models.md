# Models

What each model is and which presets it has. Models with their own page link to it there (usage, presets, speed,
accuracy). The latency of every model under one protocol is in the [inference benchmark](benchmarks.md).

`flashim.available_models()` lists the registered names. The default preset is in bold. Every model accepts all four
presets (`fp32`, `bf16`, `int8`, `fp8`) and builds the nearest one it has (`fp8` -> `int8` -> `bf16` -> `fp32`), and
every model that matches images returns `Matches` from `match(a, b)` ([configuration](configuration.md#matching)).

| model | what it is | presets |
|---|---|---|
| [`romav2`](models/romav2.md) | RoMa v2 dense matcher (DINOv3 ViT-L/16, MV-ViT, DPT head, conv refiners) | `bf16`, `int8`, **`fp8`** |
| [`rdd`](models/rdd.md) | RDD sparse detector + descriptor (ConvNeXt-tiny, multi-scale deformable transformer) | `fp32`, `bf16`, **`fp8`** |
| [`aliked`](models/aliked.md) | ALIKED sparse detector + descriptor (deformable-conv encoder, SDDH) | **`fp32`**, `bf16` |
| [`raco`](models/raco-aliked-lightglue.md) | RaCo keypoint detector + ranker | **`fp32`**, `bf16` |
| [`raco-aliked`](models/raco-aliked-lightglue.md) | RaCo keypoints + ALIKED descriptors | **`fp32`**, `bf16` |
| [`raco-<model>`](models/raco-aliked-lightglue.md#raco-with-other-models) | RaCo keypoints + the descriptors of `superpoint`, `alike`, `disk`, `xfeat`, `dedode`, `dedode-g`, `dad`, `ripe` or `rdd` (e.g. `raco-superpoint`) | the model's |
| `superpoint` | SuperPoint (VGG encoder, 256-d descriptors) | `fp32`, **`bf16`**, `int8`, `fp8` |
| `disk` | DISK (U-Net of 5x5 convs, 128-d descriptors) | `fp32`, **`bf16`**, `int8`, `fp8` |
| `alike` | ALIKE (variants `alike-t`, `alike-s`, `alike-n`, `alike-l`) | `fp32`, **`bf16`**, `int8`, `fp8` |
| `xfeat` | XFeat (0.7M-parameter conv net, 64-d descriptors) | `fp32`, **`bf16`**, `int8`, `fp8` |
| `dedode` | DeDoDe v2 detector L + descriptor B (VGG19) | `fp32`, **`bf16`**, `int8`, `fp8` |
| `dedode-g` | DeDoDe v2 detector L + descriptor G (VGG19 + DINOv2 ViT-L/14) | `fp32`, **`bf16`**, `int8`, `fp8` |
| `dad` | DaD detector (VGG11) + DeDoDe-G descriptors | `fp32`, **`bf16`**, `int8`, `fp8` |
| `ripe` | RIPE (VGG19 + ConvRefiner heatmap, hypercolumn 256-d descriptors) | `fp32`, **`bf16`**, `int8`, `fp8` |
| `loftr` | LoFTR detector-free, semi-dense matcher (variants `outdoor`, `indoor`); `match(a, b)` only | `fp32`, **`bf16`**, `int8`, `fp8` |
| `sift`, `orb` | OpenCV baselines (CPU detection, GPU matching) | `fp32` |
| [`lightglue`](models/lightglue.md) | LightGlue sparse matcher (variants `rdd`, `aliked`, `raco-aliked`, `superpoint`, `disk`, `xfeat`) | `fp32`, `bf16`, `int8`, **`fp8`** |
| [`rdd-lightglue`](models/lightglue.md) | RDD + LightGlue | `fp32`, `bf16`, `int8`, **`fp8`** |
| [`aliked-lightglue`](models/lightglue.md) | ALIKED + LightGlue (cvg's ALIKED-trained weights) | **`fp32`**, `bf16`, `int8`, `fp8` |
| [`raco-aliked-lightglue`](models/raco-aliked-lightglue.md) | RaCo keypoints + ALIKED descriptors + LightGlue (cvg's RaCo + ALIKED weights) | **`fp32`**, `bf16`, `int8`, `fp8` |
| [`superpoint-lightglue`](models/lightglue.md) | SuperPoint + LightGlue (cvg weights) | **`fp32`**, `bf16`, `int8`, `fp8` |
| [`disk-lightglue`](models/lightglue.md) | DISK + LightGlue (cvg weights) | **`fp32`**, `bf16`, `int8`, `fp8` |
| [`xfeat-lightglue`](models/lightglue.md) | XFeat + LighterGlue (verlab weights) | **`fp32`**, `bf16`, `int8`, `fp8` |
| [`raco-<model>-lightglue`](models/raco-aliked-lightglue.md#raco-with-other-models) | RaCo keypoints + `superpoint`, `disk`, `xfeat` or `rdd` descriptors + LightGlue with that model's weights | as `<model>-lightglue` |
| [`loma-<variant>`](models/loma.md) | LoMa: DaD keypoints + LoMa-trained DeDoDe descriptors + LoMa's LightGlue-style matcher (`loma-b128`, `loma-b`, `loma-l`, `loma-g`, `loma-r`; `loma` = `loma-b`) | `fp32`, **`bf16`**, `int8`, `fp8` |
| [`loma-extractor`](models/loma.md), [`loma-matcher`](models/loma.md) | LoMa's components (variant `b128` \| `b` \| `l` \| `g` \| `r`) | `fp32`, **`bf16`**, `int8`, `fp8` |

Every learned model is a native PyTorch port that is checked against its upstream implementation (`tests/parity`).

**Presets** name a precision:

| preset | meaning |
|---|---|
| `fp32` | the checkpoint's own precision |
| `bf16` | BF16 compute; keypoint selection, softmaxes and sampling / position math stay FP32 |
| `int8` | INT8 with symmetric per-output-channel weights and one calibrated scale per activation tensor |
| `fp8` | MXFP8: FP8 (E4M3) values with one power-of-two (E8M0) scale per 32 elements |

**Engines** (`eager`, `compile`; RoMaV2 also `compile-full`) are separate from presets: any preset runs with any of
them ([configuration](configuration.md)).

**Pipelines.** A pipeline's preset is every component's, and each builds the nearest one it has: `aliked-lightglue`
`int8` runs ALIKED in `bf16` and LightGlue in `int8`. `components.extractor.preset` sets the extractor's on its own
([configuration.md](configuration.md)).

**GPUs without FP8** build the nearest other preset in place of `fp8` (`int8`, else `bf16`) ([platforms.md](platforms.md)).
