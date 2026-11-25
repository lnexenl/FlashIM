# Models

What each model is and which presets it has. Models with their own page link to it there (usage, presets, speed,
accuracy). The latency of every model under one protocol is in the [inference benchmark](benchmarks.md).

`flashim.available_models()` lists the registered names. The default preset is in bold. Every model accepts all four
presets (`fp32`, `bf16`, `int8`, `fp8`) and builds the nearest one it has (`fp8` -> `int8` -> `bf16` -> `fp32`), and
every model that matches images returns `Matches` from `match(a, b)` ([configuration](configuration.md#matching)).

| model | what it is | presets |
|---|---|---|
| [`romav2`](models/romav2.md) | RoMa v2 dense matcher (DINOv3 ViT-L/16, MV-ViT, DPT head, conv refiners) | `bf16`, `int8`, **`fp8`** |
| `roma`, `roma-indoor`, `minima-roma` | RoMa (v1) dense matcher (VGG19-BN, DINOv2 ViT-L/14, GP + Transformer decoder, conv refiners; `roma` = outdoor, `minima-roma` = MINIMA modality-invariant weights) | `fp32`, **`bf16`**, `int8`, `fp8` |
| `tiny-roma` | Tiny-RoMa dense matcher (`tiny_roma_v1_outdoor`: XFeat backbone + coarse/fine conv refiners) | `fp32`, **`bf16`**, `int8`, `fp8` |
| `mast3r` | MASt3R 3D-grounded dense matcher (CroCo ViT-L encoder, ViT-B cross-attention decoder, CatMLP+DPT heads; **FP16 compute** under `bf16`/`int8`/`fp8`; CC BY-NC-SA 4.0) | `fp32`, **`bf16`**, `int8`, `fp8` |
| `dust3r` | DUSt3R stereo 3D pointmap matcher (CroCo ViT-L encoder, ViT-B cross-attention decoder, DPT heads; **FP16 compute** under `bf16`/`int8`/`fp8`; CC BY-NC-SA 4.0) | `fp32`, **`bf16`**, `int8`, `fp8` |
| `ufm` | UFM-Base dense flow & covisibility matcher (DINOv2 ViT-L/14 + 12-layer global ViT + dual DPT heads; BSD-3-Clause code, CC BY-NC-SA 4.0 weights) | `fp32`, **`bf16`**, `int8`, `fp8` |
| [`rdd`](models/rdd.md) | RDD sparse detector + descriptor (ConvNeXt-tiny, multi-scale deformable transformer) | `fp32`, `bf16`, **`fp8`** |
| [`aliked`](models/aliked.md) | ALIKED sparse detector + descriptor (deformable-conv encoder, SDDH) | **`fp32`**, `bf16` |
| [`raco`](models/raco-aliked-lightglue.md) | RaCo keypoint detector + ranker | **`fp32`**, `bf16` |
| [`raco-aliked`](models/raco-aliked-lightglue.md) | RaCo keypoints + ALIKED descriptors | **`fp32`**, `bf16` |
| [`raco-<model>`](models/raco-aliked-lightglue.md#raco-with-other-models) | RaCo keypoints + the descriptors of `superpoint`, `alike`, `disk`, `xfeat`, `dedode`, `dedode-g`, `dad`, `ripe` or `rdd` (e.g. `raco-superpoint`) | the model's |
| `superpoint` | SuperPoint (VGG encoder, 256-d descriptors) | `fp32`, **`bf16`**, `int8`, `fp8` |
| `disk` | DISK (U-Net of 5x5 convs, 128-d descriptors) | `fp32`, **`bf16`**, `int8`, `fp8` |
| `alike` | ALIKE (variants `alike-t`, `alike-s`, `alike-n`, `alike-l`) | `fp32`, **`bf16`**, `int8`, `fp8` |
| `xfeat` | XFeat (0.7M-parameter conv net, 64-d descriptors) | `fp32`, **`bf16`**, `int8`, `fp8` |
| `dedode` | DeDoDe v2 detector L + descriptor B (VGG19) | `fp32`, **`bf16`**, `int8` |
| `dedode-g` | DeDoDe v2 detector L + descriptor G (VGG19 + DINOv2 ViT-L/14) | `fp32`, **`bf16`**, `int8`, `fp8` |
| `dad` | DaD detector (VGG11) + DeDoDe-G descriptors | `fp32`, **`bf16`**, `int8`, `fp8` |
| `ripe` | RIPE (VGG19 + ConvRefiner heatmap, hypercolumn 256-d descriptors) | `fp32`, **`bf16`**, `int8`, `fp8` |
| `loftr`, `minima-loftr` | LoFTR detector-free, semi-dense matcher (variants `outdoor`, `indoor`, `minima`); `match(a, b)` only | `fp32`, **`bf16`**, `int8`, `fp8` |
| `eloftr` | EfficientLoFTR detector-free, semi-dense matcher (RepVGG + aggregated RoPE attention; variants `full`, `opt`); `match(a, b)` only | `fp32`, **`bf16`**, `int8`, `fp8` |
| `xoftr` | XoFTR cross-modal (visible-thermal) detector-free matcher (default 640 px); `match(a, b)` only | `fp32`, **`bf16`**, `int8`, `fp8` |
| `sift`, `orb` | OpenCV baselines (CPU detection, GPU matching) | `fp32` |
| [`lightglue`](models/lightglue.md) | LightGlue sparse matcher (variants `rdd`, `aliked`, `raco-aliked`, `superpoint`, `disk`, `xfeat`) | `fp32`, `bf16`, `int8`, **`fp8`** |
| [`rdd-lightglue`](models/lightglue.md) | RDD + LightGlue | `fp32`, `bf16`, `int8`, **`fp8`** |
| [`flexglue`](models/flexglue.md) | FlexGlue sparse-to-dense matcher (DINOv3-ConvNeXt search map, entmax readout) | `fp32`, `bf16`, `int8`, **`fp8`** |
| [`aliked-lightglue`](models/lightglue.md) | ALIKED + LightGlue (cvg's ALIKED-trained weights) | **`fp32`**, `bf16`, `int8` |
| [`raco-aliked-lightglue`](models/raco-aliked-lightglue.md) | RaCo keypoints + ALIKED descriptors + LightGlue (cvg's RaCo + ALIKED weights) | **`fp32`**, `bf16`, `int8`, `fp8` |
| [`superpoint-lightglue`](models/lightglue.md) | SuperPoint + LightGlue (cvg weights) | **`fp32`**, `bf16`, `int8` |
| [`disk-lightglue`](models/lightglue.md) | DISK + LightGlue (cvg weights) | **`fp32`**, `bf16`, `int8` |
| [`xfeat-lightglue`](models/lightglue.md) | XFeat + LighterGlue (verlab weights) | **`fp32`**, `bf16` |
| [`raco-<model>-lightglue`](models/raco-aliked-lightglue.md#raco-with-other-models) | RaCo keypoints + `superpoint`, `disk`, `xfeat` or `rdd` descriptors + LightGlue with that model's weights | **`fp32`**, `bf16`, `int8`, `fp8` *(`raco-rdd-lightglue` defaults to `fp8`)* |
| [`loma-<variant>`](models/loma.md) | LoMa: DaD keypoints + LoMa-trained DeDoDe descriptors + LoMa's LightGlue-style matcher (`loma-b128`, `loma-b`, `loma-l`, `loma-g`, `loma-r`; `loma` = `loma-b`) | `fp32`, **`bf16`**, `int8`, `fp8` |
| [`loma-extractor`](models/loma.md), [`loma-matcher`](models/loma.md) | LoMa's components (variant `b128` \| `b` \| `l` \| `g` \| `r`) | `fp32`, **`bf16`**, `int8`, `fp8` |
| `superglue` | SuperGlue attentional GNN + Sinkhorn OT sparse matcher (variants `outdoor`, `indoor`; Magic Leap non-commercial) | `fp32`, **`bf16`**, `int8` |
| `superpoint-superglue` | SuperPoint (`nms=4`, `threshold=0.005`) + SuperGlue (Magic Leap non-commercial; variant `outdoor` \| `indoor`) | `fp32`, **`bf16`**, `int8` |
| `omniglue` | OmniGlue: SuperPoint (`sp_v6`, MIT) + DINOv2 ViT-B/14 + 18-layer DINO-masked GNN + Sinkhorn matcher (Apache-2.0) | `fp32`, **`bf16`**, `int8` |
| `omniglue-extractor`, `omniglue-matcher` | OmniGlue's components (`omniglue-matcher` omits `fp8` and degrades `fp8` -> `int8`) | `fp32`, **`bf16`**, `int8` *(extractor adds `fp8`)* |

Every learned model is a native PyTorch port that is checked against its upstream implementation (`tests/parity`).

**Preset notes.**
- Presets omitted because MXFP8 (or INT8) loses pose AUC and is slower or dominated by the fallback (resolved via `build.NEAREST`):
  - `fp8` -> `int8`: `dedode` (pure VGG19; `dedode-g` keeps `fp8`), `aliked-lightglue`, `superpoint-lightglue` (runs SuperPoint in `bf16` and LightGlue in rotated `int8`), `disk-lightglue`, `superpoint-superglue`, and `omniglue`.
  - `fp8` / `int8` -> `bf16`: `xfeat-lightglue` (96-dim LighterGlue drops AUC under both MXFP8 and rotated INT8 and is slower than BF16 on ScanNet-1500).
- Targeted BF16 layer exclusions for quantization fidelity: `xfeat` protects `keypoint_head.*` and `block2.*` in `int8` / `fp8`; `superpoint` keeps `conv1b` in BF16 for `fp8` (`extractor_spec(..., fp8_ignore=...)`); `dedode` excludes `descriptor.encoder.layers.{3,7,10}` in `int8`; `ripe` excludes `net.encoder.layers.{3,7,10}` in `fp8`; `loma-b128` keeps `*.decoder.*` and matcher `transformers.{4,5}.*` in BF16 for `fp8` via `ModelSpec.quant_configs` per-variant `<variant>/<quant>` lookup (other LoMa variants keep full MXFP8).
- Licenses that restrict use: `romav2` and `flexglue` (`descriptor.backbone.*` in `flexglue.pth`) contain DINOv3 weights (DINOv3 License), `superglue` / `superpoint-superglue` (Magic Leap, non-commercial), `mast3r` / `dust3r` / `ufm` weights (CC BY-NC-SA 4.0), and `eloftr`, whose upstream changed its license to ZJU3DV's PRL v1.0 on 2026-09-15 (the vendored oracle and the weights are Apache-2.0 from before that); see `THIRD_PARTY_NOTICES.md`.

**Presets** name a precision:

| preset | meaning |
|---|---|
| `fp32` | the checkpoint's own precision |
| `bf16` | BF16 compute (`mast3r` and `dust3r` use FP16 compute because CroCo residual peaks reach ~344–649 where BF16 ULP degrades 3D precision); keypoint selection, softmaxes and sampling / position math stay FP32 |
| `int8` | INT8 with symmetric per-output-channel weights and static per-tensor calibrated activation scales, per-token dynamic INT8 (`DynamicInt8Linear`) for DINOv2 / CroCo ViTs (`dedode-g`, `dad`, `loma-*`, `roma`, `mast3r`, `dust3r`, `ufm`, `omniglue`), and folded Hadamard rotation + SmoothQuant (`rotate_residual_stream`) for residual-stream matchers (`lightglue`, `loma-matcher`, `superglue`, `mast3r`, `dust3r`, `ufm`) |
| `fp8` | MXFP8: FP8 (E4M3) values with one power-of-two (E8M0) scale per 32 elements |

**Engines** (`eager`, `compile`; RoMaV2 also `compile-full`) are separate from presets: any preset runs with any of
them ([configuration](configuration.md)).

**Pipelines.** A pipeline's preset is every component's, and each builds the nearest one it has: `aliked-lightglue`
`int8` runs ALIKED in `bf16` and LightGlue in `int8`. `components.extractor.preset` sets the extractor's on its own
([configuration.md](configuration.md)).

**GPUs without FP8** build the nearest other preset in place of `fp8` (`int8`, else `bf16`) ([platforms.md](platforms.md)).
