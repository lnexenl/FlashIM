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
- **Fused detector score head & NHWC trunk** (`fused_detector`, `channels_last` passes in `bf16`): CUDA C++
  `ops.raco_score_head` (`backends/cuda/csrc/raco_score_head.cu`) fuses `score_head[2:]` (`0.780 -> 0.251 ms` at
  1024x1024 on RTX 5070 Ti `sm_120`, 3.10x), `block1` reuses `raco_resblock`, and `raco/keypoints.py` uses direct padded
  index gather instead of `F.unfold` (bit-identical, saving 28-69 MB). Together with `channels_last`, `raco bf16`
  `extract_keypoints` is fast at 1024x1024 (see [Speed](#speed)).
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

The presets of `raco-<model>` are the descriptor model's (e.g. `raco-dedode` omits `fp8` -> `int8`), while
`raco-<model>-lightglue` exposes `fp32`, `bf16`, `int8`, `fp8` (`raco-rdd-lightglue` defaults to `fp8`, the others to
`fp32`); RaCo builds the nearest it has (`fp32` for `fp32`, `bf16` otherwise), and `components.raco` /
`components.descriptor` take their own presets and settings. Images are resized to a
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

`extract` of one 1200 px image (4096 keypoints, RTX 5070 Ti `sm_120`, `eager / compile`): `raco-aliked` `fp32` 26.8 / 18.1 ms,
`bf16` 7.03 / 5.76 ms. Pair time of `raco-aliked-lightglue` (2 x `extract` + `match_features`): `fp32` 111.7 / 93.5 ms, `bf16`
25.6 / 22.5 ms, `int8` 23.8 / 21.5 ms, `fp8` 24.4 / 30.0 ms. The comparison with the LightGlue-ONNX TensorRT engine is not
measured on the current code.

## Accuracy

The fast path against the exact FP32 reference on the test pair: 98.5-98.9% of the keypoints are the same pixels, RANSAC
inliers are within run-to-run noise (73-81 across the FP32 / TF32 / BF16 variants), and BF16 descriptors give about 9%
fewer matches (155 vs 170) at the same inlier count. Restoring FP32 SDDH pixel coordinates in ALIKED lifted `raco-aliked`
`bf16` by `+0.0397` / `+0.0056` and `raco-aliked-lightglue` `bf16` by `+0.0186` / `+0.0122` on MegaDepth-1500 /
ScanNet-1500 (RTX 5070 Ti `sm_120`, `eager`, 3 seeds); full gate results: [LightGlue](lightglue.md#accuracy) and
[benchmarks.md](../benchmarks.md#accuracy).
