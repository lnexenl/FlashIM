# RaCo + ALIKED + LightGlue

RaCo keypoints, ALIKED descriptors and LightGlue matching wired into one pipeline (`flashim/models/raco`).

```python
pipeline = flashim.load("raco-aliked-lightglue", "bf16", engine="compile")
m = pipeline.match(a, b, max_matches=1024, threshold=0.1)   # images must be 32-aligned, or pass resize=
```

## How it works

1. [RaCo](https://github.com/cvg/RaCo) is a detector network (ALIKED-style CNN, NMS, 2K candidates with soft-argmax) plus
   a ranker (9 residual blocks at full resolution) that reorders the candidates and keeps the top K.
2. [ALIKED](aliked.md) only describes those keypoints (encoder + SDDH, no score head or DKD).
3. [LightGlue](lightglue.md) matches them with cvg's weights for this pairing (`raco_aliked_lightglue.pth`, trained on
   RaCo's keypoints).

In FP32, keypoints and ranker scores are bit-exact against their PyTorch modules (identical keypoint set,
`tests/parity/raco`), and the full pipeline returns the same 172 matches on the test pair.

## Presets

| preset | what it does |
|---|---|
| `fp32` (default) | the checkpoint's own precision (TF32 convs) |
| `bf16` | BF16 detector trunk, ranker and descriptors; the fastest, and the one to use with `engine="compile"` |
| `int8`, `fp8` | LightGlue quantized ([LightGlue](lightglue.md)); RaCo + ALIKED, which have neither, run in `bf16` |

The pipeline preset is every component's (each builds the nearest it has); `components.extractor.preset=fp32` keeps
the extractor in FP32 under a quantized LightGlue.

## Implementation

- **Fused multi-scale kernel** (`ops.multiscale_fuse`): upsample x3, concat, the score head's first layer (FP32-accurate
  `tf32x3`) and the L2 normalize in one pass, never materializing the 512 MB FP32 concat.
- **Fused ranker blocks**: each residual block is one kernel (input read once, output written once, intermediate kept on
  chip) plus a fused 5x5 reflect head. The default is a CUDA C++ implicit GEMM on `mma.sync.m16n8k16`
  (`backends/cuda/csrc/raco_resblock.cu`), JIT-built with `nvcc` on first use and cached in `~/.cache/flashim/cuda`. It
  falls back to a Triton kernel (`backends/triton/raco.py`) without a toolchain; pick one with
  `RaCo.use_fused_ranker(backend=...)` (`"auto"`, `"cuda"` or `"triton"`).
- **BatchNorm folding** (exact).

## RaCo with other models

RaCo's keypoints can be described by any model that reads descriptors at given positions: `raco-superpoint`, `raco-alike`,
`raco-disk`, `raco-xfeat`, `raco-dedode`, `raco-dedode-g`, `raco-dad`, `raco-ripe` and `raco-rdd`. Matching is that
model's own (mutual NN, dual softmax, ...), or LightGlue with the model's weights in `raco-superpoint-lightglue`,
`raco-disk-lightglue`, `raco-xfeat-lightglue` and `raco-rdd-lightglue`.

```python
m = flashim.load("raco-dedode", "bf16").match(a, b, resize=1024, max_matches=2048)
pipeline = flashim.load("raco-superpoint-lightglue", overrides=["components.extractor.preset=bf16"])
```

The presets are the descriptor model's (`fp32`, `bf16`, `int8`, `fp8`); RaCo builds the nearest it has (`fp32` for
`fp32`, `bf16` otherwise), and `components.raco` / `components.descriptor` take their own presets and settings. Images are resized to a
multiple of 32 (224 for DeDoDe-G, whose ViT needs 56). Keypoints come back in the pixel-center convention of every
other model; LightGlue still sees RaCo's own pixel-edge ones (pixel index + 0.5), as in `raco-aliked-lightglue`. These models never saw RaCo's keypoints in training, so a LightGlue checkpoint trained on their own
detector is applied out of distribution, and DISK (whole-pixel descriptors) rounds RaCo's sub-pixel positions. MegaDepth-1500 AUC@5 (1200 px, 4096 keypoints, poselib 1.0 px, mean of 3 RANSAC seeds, BF16; RaCo runs at multiples of
32, so 1184 px instead of 1200) against the model's own detector:

| descriptor | own keypoints | RaCo keypoints | | descriptor + LightGlue | own keypoints | RaCo keypoints |
|---|---|---|---|---|---|---|
| SuperPoint | 0.452 | 0.503 | | SuperPoint | 0.599 | 0.634 |
| DISK | 0.552 | 0.566 | | DISK | 0.617 | 0.609 |
| ALIKE | 0.567 | 0.584 | | XFeat | 0.560 | 0.608 |
| XFeat | 0.445 | 0.516 | | RDD | 0.655 | 0.655 |
| DeDoDe | 0.617 | 0.607 | | ALIKED (RaCo's own weights) | 0.611 | 0.621 |
| DeDoDe-G | 0.650 | 0.621 | | | | |
| DaD (DeDoDe-G descriptors) | 0.648 | 0.621 | | | | |
| RIPE | 0.576 | 0.595 | | | | |
| RDD | 0.643 | 0.652 | | | | |

RaCo's keypoints help the weaker detectors and cost 0.01-0.03 with the DeDoDe family, whose detectors are tuned to its
descriptors. `test_describe_keypoints_reproduces_detect` checks that describing a model's own keypoints returns its own
descriptors.

## Speed

One 1024x1024 pair, 1024 keypoints, threshold 0.1, decode and resize excluded, RTX 5070 Ti:

| pipeline | ranker | time per pair |
|---|---|---|
| LightGlue-ONNX, TensorRT 11.3 fp16 | dense (exact) | 18.1 ms |
| LightGlue-ONNX, TensorRT 11.3 fp16 | `boundary` (their default: reranks only a 512-candidate window) | 13.8 ms |
| FlashIM, `bf16` + `compile` | dense (exact) | **14.6 ms** |
| FlashIM, `bf16` eager | dense (exact) | 19.9 ms |
| FlashIM, `fp32` eager | dense (exact) | 57.3 ms |

On the same exact algorithm FlashIM is about 20% faster than their TensorRT fp16 engine. Their number is engine-only GPU
time; FlashIM's includes Python launch overhead. FlashIM has no `boundary` ranker, so that row has no like-for-like
counterpart.

## Accuracy

The fast path against the exact FP32 reference on the test pair: 98.5-98.9% of the keypoints are the same pixels, RANSAC
inliers are within run-to-run noise (73-81 across the FP32 / TF32 / BF16 variants), and BF16 descriptors give about 9%
fewer matches (155 vs 170) at the same inlier count. MegaDepth-1500 AUC@5: [LightGlue](lightglue.md#accuracy).
