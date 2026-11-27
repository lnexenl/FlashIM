# Models

What each model is and which presets it has. Models with their own page link to it there (usage, presets, speed,
accuracy). The latency of every model under one protocol is in the [inference benchmark](benchmarks.md).

`flashim.available_models()` lists the registered names. The default preset is in bold. Every model accepts all four
presets (`fp32`, `bf16`, `int8`, `fp8`) and builds the nearest one it has (`fp8` -> `int8` -> `bf16` -> `fp32`), and
every model that matches images returns `Matches` from `match(a, b)` ([configuration](configuration.md#matching)).

### Dense and semi-dense models

| model | what it is | presets |
|---|---|---|
| [`romav2`](models/romav2.md) | RoMa v2 dense matcher (DINOv3 ViT-L/16, MV-ViT, DPT head, conv refiners) | `bf16`, `int8`, **`fp8`** |
| `roma`, `roma-indoor`, `minima-roma` | RoMa (v1) dense matcher (VGG19-BN, DINOv2 ViT-L/14, GP + Transformer decoder, conv refiners; `roma` = outdoor, `minima-roma` = MINIMA modality-invariant weights) | `fp32`, **`bf16`**, `int8`, `fp8` |
| `tiny-roma` | Tiny-RoMa dense matcher (`tiny_roma_v1_outdoor`: XFeat backbone + coarse/fine conv refiners) | `fp32`, **`bf16`**, `int8`, `fp8` |
| `mast3r` | MASt3R 3D-grounded dense matcher (CroCo ViT-L encoder, ViT-B cross-attention decoder, CatMLP+DPT heads; **FP16 compute** under `bf16`/`int8`/`fp8`; CC BY-NC-SA 4.0) | `fp32`, **`bf16`**, `int8`, `fp8` |
| `dust3r` | DUSt3R stereo 3D pointmap matcher (CroCo ViT-L encoder, ViT-B cross-attention decoder, DPT heads; **FP16 compute** under `bf16`/`int8`/`fp8`; CC BY-NC-SA 4.0) | `fp32`, **`bf16`**, `int8`, `fp8` |
| `ufm` | UFM-Base dense flow & covisibility matcher (DINOv2 ViT-L/14 + 12-layer global ViT + dual DPT heads; BSD-3-Clause code, CC BY-NC-SA 4.0 weights) | `fp32`, **`bf16`**, `int8`, `fp8` |
| `loftr`, `minima-loftr` | LoFTR detector-free, semi-dense matcher (variants `outdoor`, `indoor`, `minima`); `match(a, b)` only | `fp32`, **`bf16`**, `int8`, `fp8` |
| `eloftr` | EfficientLoFTR detector-free, semi-dense matcher (RepVGG + aggregated RoPE attention; variants `full`, `opt`); `match(a, b)` only | `fp32`, **`bf16`**, `int8`, `fp8` |
| `xoftr` | XoFTR cross-modal (visible-thermal) detector-free matcher (default 640 px); `match(a, b)` only | `fp32`, **`bf16`**, `int8`, `fp8` |

### Sparse extractors

| model | what it is | presets |
|---|---|---|
| [`rdd`](models/rdd.md) | RDD sparse detector + descriptor (ConvNeXt-tiny, multi-scale deformable transformer) | `fp32`, `bf16`, **`fp8`** |
| `traqpoint` | TraqPoint sparse detector + descriptor (DINOv3 ConvNeXt-B, 4-level deformable transformer, RL detector; DINOv3 License for backbone weights) | `fp32`, **`bf16`**, `int8`, `fp8` |
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
| `ripepp` | RIPE++ (VGG19 + ConvRefiner heatmap, subpixel NMS, hypercolumn 256-d descriptors, MNN; non-commercial) | `fp32`, **`bf16`**, `int8`, `fp8` |
| `ripepp-aachen`, `ripepp-scared` | RIPE++ specialist weights (MegaDepth + Tokyo 24/7; SCARED surgical video; non-commercial) | `fp32`, **`bf16`** |
| `sift`, `orb` | OpenCV baselines (CPU detection, GPU matching) | `fp32` |

### Sparse matchers and pipelines

| model | what it is | presets |
|---|---|---|
| [`lightglue`](models/lightglue.md) | LightGlue sparse matcher (variants `rdd`, `aliked`, `raco-aliked`, `superpoint`, `disk`, `xfeat`, `ripepp`) | `fp32`, `bf16`, `int8`, **`fp8`** |
| [`rdd-lightglue`](models/lightglue.md) | RDD + LightGlue | `fp32`, `bf16`, `int8`, **`fp8`** |
| [`aliked-lightglue`](models/lightglue.md) | ALIKED + LightGlue (cvg's ALIKED-trained weights) | **`fp32`**, `bf16`, `int8` |
| [`raco-aliked-lightglue`](models/raco-aliked-lightglue.md) | RaCo keypoints + ALIKED descriptors + LightGlue (cvg's RaCo + ALIKED weights) | **`fp32`**, `bf16`, `int8`, `fp8` |
| [`superpoint-lightglue`](models/lightglue.md) | SuperPoint + LightGlue (cvg weights) | **`fp32`**, `bf16`, `int8` |
| [`disk-lightglue`](models/lightglue.md) | DISK + LightGlue (cvg weights) | **`fp32`**, `bf16`, `int8` |
| [`xfeat-lightglue`](models/lightglue.md) | XFeat + LighterGlue (verlab weights) | **`fp32`**, `bf16` |
| [`ripepp-lightglue`](models/lightglue.md) | RIPE++ + weakly-supervised LightGlue (RIPE++ RL weights; non-commercial) | `fp32`, **`bf16`**, `int8` |
| [`raco-<model>-lightglue`](models/raco-aliked-lightglue.md#raco-with-other-models) | RaCo keypoints + `superpoint`, `disk`, `xfeat` or `rdd` descriptors + LightGlue with that model's weights | **`fp32`**, `bf16`, `int8`, `fp8` *(`raco-rdd-lightglue` defaults to `fp8`)* |
| [`loma-<variant>`](models/loma.md) | LoMa: DaD keypoints + LoMa-trained DeDoDe descriptors + LoMa's LightGlue-style matcher (`loma-b128`, `loma-b`, `loma-l`, `loma-g`, `loma-r`; `loma` = `loma-b`) | `fp32`, **`bf16`**, `int8`, `fp8` |
| [`loma-extractor`](models/loma.md), [`loma-matcher`](models/loma.md) | LoMa's components (variant `b128` \| `b` \| `l` \| `g` \| `r`) | `fp32`, **`bf16`**, `int8`, `fp8` |
| `superglue` | SuperGlue attentional GNN + Sinkhorn OT sparse matcher (variants `outdoor`, `indoor`; Magic Leap non-commercial) | `fp32`, **`bf16`**, `int8` |
| `superpoint-superglue` | SuperPoint (`nms=4`, `threshold=0.005`) + SuperGlue (Magic Leap non-commercial; variant `outdoor` \| `indoor`) | `fp32`, **`bf16`**, `int8` |
| `omniglue` | OmniGlue: SuperPoint (`sp_v6`, MIT) + DINOv2 ViT-B/14 + 18-layer DINO-masked GNN + Sinkhorn matcher (Apache-2.0) | `fp32`, **`bf16`**, `int8` |
| `omniglue-extractor`, `omniglue-matcher` | OmniGlue's components (`omniglue-matcher` omits `fp8` and degrades `fp8` -> `int8`) | `fp32`, **`bf16`**, `int8` *(extractor adds `fp8`)* |

### Sparse-to-dense

| model | what it is | presets |
|---|---|---|
| [`flexglue`](models/flexglue.md) | FlexGlue sparse-to-dense matcher (DINOv3-ConvNeXt search map, entmax readout) | `fp32`, `bf16`, `int8`, **`fp8`** |

Every learned model is a native PyTorch port that is checked against its upstream implementation (`tests/parity`).

**Preset notes.**
- Presets omitted because MXFP8 (or INT8) loses pose AUC and is slower or dominated by the fallback (resolved via `build.NEAREST`):
  - `fp8` -> `int8`: `dedode` (pure VGG19; `dedode-g` keeps `fp8`), `aliked-lightglue`, `superpoint-lightglue` (runs SuperPoint in `bf16` and LightGlue in rotated `int8`), `disk-lightglue`, `ripepp-lightglue` (runs RIPE++ in `int8` and LightGlue in `bf16`), `superpoint-superglue`, and `omniglue`.
  - `fp8` / `int8` -> `bf16`: `xfeat-lightglue` (96-dim LighterGlue drops AUC under MXFP8 and rotated static INT8 due to 8.85x post-rotation token outliers, while dynamic INT8 is slower than BF16, 3.67 vs 3.49 ms), and `ripepp-aachen` / `ripepp-scared` (no Tokyo 24/7 / SCARED calibration & gate datasets available).
- Targeted BF16 layer exclusions for quantization fidelity: `xfeat` protects `keypoint_head.*` and `block2.*` in `int8` / `fp8`; `superpoint` keeps `conv1b` in BF16 for `fp8` (`extractor_spec(..., fp8_ignore=...)`); `dedode` excludes `descriptor.encoder.layers.{3,7,10}` in `int8`; `ripe` excludes `net.encoder.layers.{3,7,10}` in `fp8`; `ripepp` excludes `net.decoder.layers.{1,2}.*` in `int8` and `net.encoder.layers.{3,7,10}` plus `net.decoder.layers.{1,2}.*` in `fp8`; `loma-b128` keeps `*.decoder.*` and matcher `transformers.{4,5}.*` in BF16 for `fp8` via `ModelSpec.quant_configs` per-variant `<variant>/<quant>` lookup (other LoMa variants keep full MXFP8).
- Licenses for the default weights and upstream code of every model and pipeline are in the [per-model license table](licenses.md). The `eloftr` entry covers the pinned Apache-2.0 release. Upstream EfficientLoFTR changed to ZJU3DV's PRL v1.0 on 2026-09-15. See the [third-party notices](../THIRD_PARTY_NOTICES.md).

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

## Usage examples

`match(a, b, *, resize, max_matches, threshold)` is the same for every model; the values below are each model's usual
call (`resize` is the long side of the working resolution).

```python
import flashim

roma = flashim.load("roma")                                  # RoMa (v1) dense matcher (outdoor; "roma-indoor", "minima-roma")
m = roma.match("a.jpg", "b.jpg", max_matches=5000)           # 5000 balanced samples of the dense warp (560 -> 864)

tiny = flashim.load("tiny-roma", "int8")                     # Tiny-RoMa dense matcher (fp32, bf16, int8, fp8)
m = tiny.match("a.jpg", "b.jpg", max_matches=5000)           # 5000 balanced samples of the dense warp

eloftr = flashim.load("eloftr")                              # EfficientLoFTR (bf16, reparameterized; variant full | opt)
m = eloftr.match("a.jpg", "b.jpg", resize=832)               # semi-dense matches (sides floored to multiples of 32)

xoftr = flashim.load("xoftr")                                # XoFTR: cross-modal visible-thermal matcher (default 640 px)
m = xoftr.match("a.jpg", "b.jpg")                            # Matches at original image resolution

minima = flashim.load("minima-loftr")                        # MINIMA modality-invariant LoFTR ("minima-roma" for RoMa)
m = minima.match("a.jpg", "b.jpg", resize=840)               # cross-modal & RGB semi-dense matches

sg = flashim.load("superpoint-superglue")                    # SuperPoint + SuperGlue (outdoor default; variant indoor)
m = sg.match("a.jpg", "b.jpg", resize=1024)                  # 1024 keypoints per image, match_threshold=0.2

mast3r = flashim.load("mast3r")                              # MASt3R: 3D-grounded dense matcher (512 px long side)
m = mast3r.match("a.jpg", "b.jpg")                           # Matches from fast reciprocal NNs on 24-D descriptors

dust3r = flashim.load("dust3r")                              # DUSt3R: stereo 3D pointmap matcher (512 px long side)
m = dust3r.match("a.jpg", "b.jpg")                           # Matches from exact reciprocal NNs in 3D pointmaps

ufm = flashim.load("ufm", "int8")                            # UFM: DINOv2 ViT-L + global ViT + dual DPT dense flow
m = ufm.match("a.jpg", "b.jpg", max_matches=5000)            # 5000 balanced samples of the dense flow + covisibility

og = flashim.load("omniglue")                                # OmniGlue: SuperPoint (sp_v6) + DINOv2 ViT-B/14 + matcher
m = og.match("a.jpg", "b.jpg", threshold=0.02)               # 1024 keypoints per image (default)

fg = flashim.load("flexglue")                                # FlexGlue: DINOv3-ConvNeXt search map + entmax readout (fp8)
m = fg.match("a.jpg", "b.jpg", max_matches=5000)             # dense warp samples (or fg.match_keypoints(...))

loma = flashim.load("loma-g")                                # LoMa: DaD + LoMa descriptors + matcher (b128, b, l, g, r)
m = loma.match("a.jpg", "b.jpg", max_matches=4096)           # 4096 keypoints per image (default 2048)

ripepp = flashim.load("ripepp", "int8")                      # RIPE++ (also "ripepp-aachen", "ripepp-scared")
m = ripepp.match("a.jpg", "b.jpg", resize=1200, max_matches=4096)  # subpixel keypoints + 256-d hypercolumns + MNN

ripepp_lg = flashim.load("ripepp-lightglue", "int8")         # RIPE++ (int8) + weakly-supervised LightGlue (bf16)
m = ripepp_lg.match("a.jpg", "b.jpg", resize=1200, max_matches=4096)

tp = flashim.load("traqpoint")                               # TraqPoint (DINOv3 ConvNeXt-B + deformable transformer + RL detector)
m = tp.match("a.jpg", "b.jpg", resize=1600, max_matches=4096)
```
