# Models, presets and results

What each model is, what its presets do, and the speed and accuracy measured for them. All numbers are from an
RTX 5070 Ti. The latency of every model, preset and engine under one protocol is in the
[inference benchmark](benchmarks.md).

- [Overview](#overview)
- [RoMaV2](#romav2)
- [RDD](#rdd)
- [ALIKED](#aliked)
- [ALIKED + LightGlue](#aliked--lightglue)
- [RaCo + ALIKED + LightGlue](#raco--aliked--lightglue)

## Overview

`flashim.available_models()` lists the registered names. The default preset is in bold.

| model | what it is | presets |
|---|---|---|
| `romav2` | RoMa v2 dense matcher (DINOv3 ViT-L/16, MV-ViT, DPT head, conv refiners) | `bf16`, `int8`, **`fp8`** |
| `rdd` | RDD sparse detector + descriptor (ConvNeXt-tiny, multi-scale deformable transformer) | `fp32`, `bf16`, **`fp8`**, `fp8-full`, `fp8-backbone` |
| `aliked` | ALIKED sparse detector + descriptor (deformable-conv encoder, SDDH) | **`fp32`**, `bf16` |
| `raco` | RaCo keypoint detector + ranker | **`fp32`**, `bf16` |
| `raco-aliked` | RaCo keypoints + ALIKED descriptors | **`fp32`**, `bf16` |
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
| `lightglue` | LightGlue sparse matcher (variants `rdd`, `aliked`, `raco-aliked`, `superpoint`, `disk`, `xfeat`) | `fp32`, `bf16`, `int8`, **`fp8`** |
| `rdd-lightglue` | RDD + LightGlue | `fp32`, `bf16`, `int8`, **`fp8`** |
| `aliked-lightglue` | ALIKED + LightGlue (cvg's ALIKED-trained weights) | **`fp32`**, `bf16`, `int8`, `fp8` |
| `raco-aliked-lightglue` | RaCo keypoints + ALIKED descriptors + LightGlue (cvg's RaCo + ALIKED weights) | **`fp32`**, `bf16`, `int8`, `fp8` |
| `superpoint-lightglue` | SuperPoint + LightGlue (cvg weights) | **`fp32`**, `bf16`, `int8`, `fp8` |
| `disk-lightglue` | DISK + LightGlue (cvg weights) | **`fp32`**, `bf16`, `int8`, `fp8` |
| `xfeat-lightglue` | XFeat + LighterGlue (verlab weights) | **`fp32`**, `bf16`, `int8`, `fp8` |

Every learned model is a native PyTorch port that is checked against its upstream implementation (`tests/parity`).

**Presets** name a precision:

| preset | meaning |
|---|---|
| `fp32` | the checkpoint's own precision |
| `bf16` | BF16 compute; keypoint selection, softmaxes and sampling / position math stay FP32 |
| `int8` | INT8 with symmetric per-output-channel weights and one calibrated scale per activation tensor |
| `fp8` | MXFP8: FP8 (E4M3) values with one power-of-two (E8M0) scale per 32 elements |

**Engines** are separate from presets. Learned models run with `engine="eager"` or `"compile"`; RoMaV2 adds
`"compile-full"`.

**Pipelines.** A pipeline's preset sets LightGlue's precision only. The extractor keeps its own default (`fp32`; `fp8`
for RDD) unless you set `components.extractor.preset` ([configuration.md](configuration.md)).

**GPUs without FP8** build the INT8 counterpart of an FP8 preset ([platforms.md](platforms.md)).

## RoMaV2

Dense matcher: a native port of RoMa v2 (DINOv3 ViT-L/16, matcher ViT, DPT head, conv refiners).

```python
model = flashim.load("romav2", preset="fp8", setting="mega1500", engine="compile")
result = model.match("a.jpg", "b.jpg")                       # DenseMatchResult (warp, overlap, precision)
matches, conf, prec_a, prec_b = model.sample(result, 5000)
kpts_a, kpts_b = model.to_pixel_coordinates(matches, h_a, w_a, h_b, w_b)
```

### Presets

- **`bf16`**: the reference, bit-identical to the RoMaV2 implementation.
- **`fp8`**: MXFP8.
  - Transformers: every DINOv3 and matcher-ViT linear, with static calibrated activation scales and fused
    LayerNorm→quantize and GELU→quantize producers.
  - Refiners at strides 4 and 2: one Triton kernel for depthwise 5x5 + bias + ReLU + dynamic MXFP8 quantization, then
    a block-scaled MXFP8 pointwise GEMM.
- **`int8`**: RoMaV2's `int8_*` recipe. Symmetric per-output-channel weights, static per-tensor activation scales
  calibrated with the 99.99th-percentile observer over 64 pairs, INT32 accumulation with a fused dequantize + bias
  epilogue.
  - Transformers: the transformer linears and the DPT head's four 1x1 input projections (Triton GEMM;
    `torch._int_mm` is kept as an alternative implementation).
  - Refiners at strides 4 and 2: INT8 blocks. C=128 blocks run as one fused depthwise + INT8 GEMM kernel, C=512
    producers in TileLang.

`fp8` and `int8` share FlashIM's refiner kernels: fused BF16 stride-1 conv blocks, a fused refiner input (warp
sampling, displacement embedding and local correlation in one pass), fused FP32 output heads, and both match
directions refined as one batch. Neither is bit-identical to RoMaV2 (reduction order, rounding points); both are
validated by pose AUC.

Coordinate heads, the patch embedding, refiner projections and displacement embeddings are protected: a quantization
config that targets them is rejected.

The oracle presets reproduce RoMaV2's FastFP8 runtime bit for bit and exist for the parity tests: `fp8-p8`
(transformer linears only; BF16 attention and refiners), `fp8-s4` (+ fused stride-4 refiner blocks) and `fp8-s42`
(+ fused stride-2 blocks).

### Engines

| engine | compiled regions |
|---|---|
| `eager` | none |
| `compile` | DINOv3 and the matcher ViT (RoMaV2's validated setting) |
| `compile-full` | also VGG with folded BatchNorm, the DPT head and the refiner conv stacks |

### Speed

mega1500 setting (800² LR + 1024² HR), both directions, B=1, preloaded forward:

| configuration | forward | vs BF16 |
|---|---|---|
| `bf16` + `compile` | 344 ms | 1.0x |
| RoMaV2's own FastFP8 (stride-4 and stride-2 fused) + `compile` | 210 ms | 1.64x |
| `fp8` + `compile-full` | 110 ms | 3.13x |
| `int8` + `compile-full` | 103 ms | 3.36x |

Every preset and engine, with same-engine speed-ups: [inference benchmark](benchmarks.md#romav2-dense).

### Accuracy

Pose AUC@5 change vs `bf16` (3 seeds, `compile-full`). The gate is a 3-seed mean of at least -0.005; RANSAC seed noise
is about 0.003.

| preset | MegaDepth-1500 | ScanNet-1500 |
|---|---|---|
| `fp8` | -0.0023 | +0.0007 |
| `int8` | -0.0010 | +0.0004 |

Every `fp8` seed is within -0.005.

HardMatch (LoMa, ECCV 2026: 900 hand-labelled hard pairs, mAA@10px, 3 seeds, `wxbs` setting). Run with
`benchmarks/accuracy/run_hardmatch.py`, which follows the official `hardmatch` package protocol.

| preset | mAA@10px | vs `bf16` |
|---|---|---|
| `bf16` | 0.4786 | |
| `fp8-s42` (RoMaV2's own FP8 recipe) | 0.4707 | -0.0079 |
| `fp8` | 0.4712 | -0.0074 |

FP8 costs more on these deliberately hard pairs than on MegaDepth / ScanNet, and FlashIM's extra kernels add none of
that cost. HardMatch was not run for `int8`.

### Options

- **Match sampling.** `sample()` (KDE-balanced) takes 0.86 ms, against 16 ms and 1.5 GiB of scratch memory in RoMaV2.
  Pose AUC vs RoMaV2's sampling: +0.0012 / -0.0010 (3 seeds). `FLASHIM_FAST_SAMPLING=0` restores RoMaV2's.
  `match(..., num_samples=N)` returns pixel keypoints directly.
- **INT8 attention.** `flashim.load("romav2", preset="int8", attention="sage-fp8")` runs the attention of every
  quantized transformer block through [SageAttention](https://github.com/thu-ml/SageAttention): INT8 Q/K with K
  smoothing and an FP8 PV (`"sage"`: FP16 PV). It needs `sageattention` 2.x built for your GPU, which is not a project
  dependency (with GCC 16, build it with clang). With `int8` + `compile-full`:

  | attention | forward | AUC@5 vs `bf16`, MegaDepth / ScanNet (3 seeds) |
  |---|---|---|
  | `sdpa` (default) | 102.7 ms | |
  | `sage` | 99.7 ms | -0.0012 / -0.0002 |
  | `sage-fp8` | 95.4 ms | +0.0008 / +0.0010 |

- **Calibration.** `scripts/calibrate_romav2.py --format int8 --preset int8 [--merge]`. The shipped table is
  `flashim/models/romav2/calibration/romav2_int8_mega1500.safetensors`.

## RDD

Sparse detector + descriptor: a native port of [RDD](https://github.com/xtcpete/rdd) (CVPR'25). A ConvNeXt-tiny
backbone feeds a 5-level multi-scale deformable-attention encoder (4 layers; 140k tokens at 1600 px) that produces the
descriptor map. A small FP32 conv net produces the keypoint score map. `fp32` matches the reference to 1e-5.

```python
model = flashim.load("rdd", preset="fp8", engine="compile")
feats = model.extract("a.jpg", resize=1600)               # SparseFeatures: 4096 keypoints + 256-d descriptors
m = model.match("a.jpg", "b.jpg", resize=1600)            # SparseMatchResult (mutual dual-softmax NN)
```

`rdd.pth` and LightGlue's `rdd_lg.pth` are downloaded on first use from the Google Drive folder linked in the RDD
README ([checkpoints.md](checkpoints.md)).

### Presets

| preset | what it does |
|---|---|
| `fp32` | the checkpoint's own precision |
| `bf16` | descriptor branch in BF16; detector, softmax and all sampling / position math in FP32 |
| `fp8` | `bf16` + the encoder's value / output / FFN linears as static MXFP8, run by a fused encoder pass |
| `fp8-full` | `fp8` + MXFP8 sampling-offset / attention-weight projections. No faster: they share one BF16 GEMM in `fp8` |
| `fp8-backbone` | `fp8` + MXFP8 ConvNeXt MLPs. Slower: those GEMMs are small-K and memory-bound. Kept for experiments |

The fused encoder pass of `fp8`:

- deformable attention with the softmax fused in (one BF16 GEMM for sampling offsets + attention logits);
- biases in the GEMM epilogue and LayerNorm→quantize producers;
- one MXFP8 MLP kernel for `linear1` + bias + GELU + quantize + `linear2`. The 1024-wide hidden activation never leaves
  the SM, and the result is bit-identical to the two-GEMM path.

`engine="compile"` compiles the backbone, input projections, detector and encoder once per input resolution.

### Speed

`extract()` at 1056x1600, image preloaded, keypoints + descriptors included:

| configuration | extract | peak memory |
|---|---|---|
| RDD reference (PyTorch deformable path, no CUDA extension) | OOM (16 GB) | |
| `fp32` eager | 110.9 ms | 2.0 GiB |
| `bf16` eager | 57.6 ms | 1.1 GiB |
| `bf16` + `compile` | 40.3 ms | 1.1 GiB |
| `fp8` eager | 49.3 ms | 1.3 GiB |
| `fp8` + `compile` | 35.4 ms | 1.1 GiB |

### Accuracy

MegaDepth-1500 pose AUC, RDD protocol (1600 px, 4096 keypoints, mutual NN, OpenCV RANSAC), RGB input:

| configuration | AUC@5 | AUC@10 | AUC@20 |
|---|---|---|---|
| `fp32` | 0.5216 | 0.6827 | 0.8019 |
| `bf16` | 0.5270 | 0.6855 | 0.8031 |
| `fp8` | 0.5418 | 0.6974 | 0.8137 |
| `fp8` + `compile` | 0.5287 | 0.6887 | 0.8071 |
| `fp8-full` | 0.5242 | 0.6801 | 0.7980 |

The run-to-run spread of these sparse pipelines is about ±0.01 (`bf16` eager and `compile` differ by 0.006), so FP8
costs no measurable accuracy. RDD's own script feeds OpenCV BGR images; `fp32` scores 0.5155 AUC@5 that way.
Run with `benchmarks/accuracy/run_sparse.py`; the matrix is `benchmarks/accuracy/rdd_matrix.sh`.

### Detector precision

The keypoint detector stays FP32 by default. `detector_dtype="bfloat16"` runs its conv trunk in BF16 (score head FP32)
and saves 0.9 ms compiled (3.7 → 2.7 ms), but about 12% of the 4096 keypoints change. MegaDepth AUC@5 / @10 / @20 moved
-0.012 / -0.015 / -0.014 in eager and -0.004 / -0.007 / -0.006 compiled. That is inside run noise, but every paired
number went down.

## ALIKED

Sparse detector + descriptor: a native port of [ALIKED](https://github.com/Shiaoming/ALIKED) (TIM'23). A small conv
encoder (plain 3x3 convs at strides 1/2, deformable 3x3 convs at strides 8/32) feeds a multi-scale feature fusion. A
4-layer conv head produces the keypoint score map, and `SDDH` samples descriptors at deformable offsets around each
detected keypoint (DKD: soft-NMS + soft-argmax sub-pixel refinement).

```python
model = flashim.load("aliked", preset="fp32", variant="aliked-n16")
feats = model.extract("a.jpg")                       # SparseFeatures: threshold-mode keypoints + 128-d descriptors
m = model.match("a.jpg", "b.jpg")                    # SparseMatchResult (mutual nearest-neighbour, cosine >= 0.75)
```

Four released checkpoints, chosen with `variant=` and downloaded on first use from the `models/` folder of the ALIKED
repo ([checkpoints.md](checkpoints.md)):

| variant | descriptors |
|---|---|
| `aliked-t16` | 64-d |
| `aliked-n16` (default), `aliked-n16rot` | 128-d |
| `aliked-n32` | 128-d, 32 sample points |

### Presets

| preset | what it does |
|---|---|
| `fp32` | the checkpoint's own precision. Dense maps, keypoints and scores match the reference bit-exactly; descriptors to cosine > 0.99999 (deformable-sampling summation order only) |
| `bf16` | encoder and `SDDH` in BF16; the score head and NMS / soft-argmax / sampling math stay FP32. Deformable convs run their `deform_conv2d` call in FP32 (no BF16 / FP16 kernel exists) |

Every preset folds each block's eval BatchNorm into its convolution. The fold is exact (< 1e-6 max abs diff), and the
folded blocks are listed in `model.fused_batchnorm`.

### Speed

`extract()` at 1024x1024, image preloaded, `aliked-n16`, keypoints + descriptors included:

| configuration | extract |
|---|---|
| `fp32` eager, no BatchNorm fold | 15.3 ms |
| `bf16` eager, no BatchNorm fold | 15.1 ms |
| `bf16` eager, BatchNorm folded (default) | 10.5 ms |
| `bf16` + `compile`, BatchNorm folded | 10.3 ms |

The table shows what the BatchNorm fold buys. It predates the fused multi-scale kernel below; current latency is in the
[inference benchmark](benchmarks.md#sparse-extractors).

- **`compile` adds nothing** beyond BF16 + the BatchNorm fold: the workload is bound by convolutions and
  `deform_conv2d`, which cuDNN already schedules well. Eager is the practical default.
- **`channels_last=True`** is available but off by default: measured slower. ALIKED's channel counts are at most 128,
  and `deform_conv2d` always returns contiguous NCHW, so cuDNN has no channels-last algorithm for most of the encoder.
- **`ops.multiscale_fuse`** runs the multi-scale feature step (3 bilinear upsamples, the channel concat, the FP32 cast
  for the score head and the L2 normalize for the descriptor map) as one Triton kernel. Its bilinear coordinates
  are FP32 and inputs are upcast as they are loaded. A first version with FP64 coordinates and inputs pre-cast to
  FP32 was slower: `bf16` `extract()` at 1024x1024 took 6.97 ms against 5.94 ms now, and the ALIKED + LightGlue
  pipeline 16.6 ms against 14.6 ms. Numerics are unaffected (parity is still bit-exact in FP32, cosine > 0.9999 in
  BF16).

### Quantization

There is no quantized preset.

- **1x1 projections** (`conv1..4`, the `downsample`s, `desc_head.sf_conv`; all K, N <= 128) are the only convs the
  quantization infrastructure can target directly, since ALIKED has no `nn.Linear`. RDD's `fp8-backbone` preset found
  this class of GEMM (small-K, memory-bound) slower quantized than in BF16. ALIKED's are smaller still, so it was not
  re-measured.
- **Deformable convs** (`block3` / `block4`, where most of the runtime goes) can run in INT8, opt-in:
  `load_aliked(..., deform_int8_calibration=...)`, calibrated with `DeformConvCalibrator`. `ops.deform_patches_int8`
  gathers each output pixel's patch at its deformable offset and quantizes it in one Triton kernel that feeds the INT8
  GEMM (`flashim/models/aliked/deform_quant.py`); the sampling offsets stay FP32. Feature cosine vs FP32 is > 0.999.
  At batch 1 it is within about 2% of the FP32 `deform_conv2d` call; at batch >= 4 it is 25-39% faster, growing with
  the batch. It is worth it for batched workloads, not for a single-image `extract()`.

## ALIKED + LightGlue

```python
pipeline = flashim.load("aliked-lightglue", preset="fp32", aliked_preset="fp32")
m = pipeline.match("a.jpg", "b.jpg", resize=1024, top_k=1024)
```

cvg/LightGlue trains one LightGlue per (detector, descriptor) pair. This pipeline uses the weights trained on ALIKED's
own keypoints and descriptors (`aliked_lightglue.pth`: `input_dim=128 -> dim=256` through `input_proj`,
`filter_threshold=0.1`, see `lightglue.config.ALIKED_CONFIG`); the RaCo pairing below has its own. With
`load_lightglue()` they are `variant="aliked"` and `variant="raco-aliked"`.

The `int8` / `fp8` presets use calibration tables computed on each pairing's features
(`lightglue/calibration/lightglue_{aliked,raco-aliked}_{int8,fp8}_mega.safetensors`).

MegaDepth-1500 AUC@5 (unified protocol, 4 seeds; `aliked-lightglue` with the `fp32` extractor):

| pipeline | `fp32` | `bf16` | `int8` | `fp8` |
|---|---|---|---|---|
| `aliked-lightglue` | 0.627 | 0.627 | 0.620 | 0.613 |
| `raco-aliked-lightglue` | | 0.640 | 0.638 | 0.634 |

## RaCo + ALIKED + LightGlue

```python
pipeline = flashim.load("raco-aliked-lightglue", preset="bf16", extractor_preset="bf16", engine="compile")
m = pipeline.match(a, b, top_k=1024, threshold=0.1)   # images must be 32-aligned, or pass resize=
```

The `raco_aliked` pipeline of [fabio-sim/LightGlue-ONNX](https://github.com/fabio-sim/LightGlue-ONNX), ported natively
(`flashim/models/raco`). It is not the plain ALIKED pipeline above:

1. [RaCo](https://github.com/cvg/RaCo) is a separate detector network (ALIKED-style CNN, NMS, 2K candidates with
   soft-argmax) plus a ranker (9 residual blocks at full resolution) that reorders the candidates and keeps the top K.
2. ALIKED only describes those keypoints (encoder + SDDH, no score head or DKD).
3. LightGlue matches them with cvg/LightGlue's weights for this pairing (`raco_aliked_lightglue.pth`, trained on RaCo's
   keypoints).

In FP32, keypoints and ranker scores are bit-exact against their PyTorch modules (identical keypoint set,
`tests/parity/raco`), and the full pipeline returns the same 172 matches.

### Speed

One 1024x1024 pair, 1024 keypoints, threshold 0.1, decode and resize excluded:

| pipeline | ranker | time per pair |
|---|---|---|
| LightGlue-ONNX, TensorRT 11.3 fp16 | dense (exact) | 18.1 ms |
| LightGlue-ONNX, TensorRT 11.3 fp16 | `boundary` (their default: reranks only a 512-candidate window) | 13.8 ms |
| FlashIM, `bf16` + `compile` | dense (exact) | **14.6 ms** |
| FlashIM, `bf16` eager | dense (exact) | 19.9 ms |
| FlashIM, `fp32` eager (TF32 convs) | dense (exact) | 57.3 ms |

On the same exact algorithm FlashIM is about 20% faster than their TensorRT fp16 engine. Their number is engine-only
GPU time; FlashIM's includes Python launch overhead. FlashIM has no `boundary` ranker, so the 13.8 ms row has no
like-for-like counterpart.

Their engines were built and validated with `trtexec` directly, because ONNX Runtime's TensorRT execution provider
wants their pinned `libnvinfer.so.10`. Two fixes were needed: `onnxslim`, which constant-folds an axis the exporter
leaves as a `Reshape` that TensorRT 11's stricter parser rejects, and exporting with `--fp16` despite the tool's stale
warning (TensorRT >= 10 is strongly typed, so precision comes from the graph's dtypes).

### What makes it fast

The port started at 88.8 ms. In order of effect:

1. **Fused multi-scale kernel** (`ops.multiscale_fuse`). Upsample x3, concat, the score head's first layer (1x1 conv
   128→8 + SELU, FP32-accurate `tf32x3`) and the L2 normalize run in one pass that never materializes the 512 MB FP32
   concat. It takes 0.41 ms (0.57 ms for ALIKED's descriptor path) with channels-last coarse maps. With NCHW maps it
   took 2.3 ms: the bilinear gathers touched 32 cache lines per warp (Nsight Compute: L1/TEX 93%, DRAM 5%).
2. **BatchNorm folding** (exact), and a BF16 detector trunk and ranker under `compile`.
3. **Fused ranker blocks.** Each residual block is one kernel (input read once, output written once, intermediate kept
   on chip), plus a fused 5x5 reflect head. Per 12-channel block at 1024x1024:

   | implementation | time |
   |---|---|
   | torch / Inductor, unfused | ~0.33 ms |
   | Triton (`backends/triton/raco.py`) | 0.237 ms |
   | CUDA C++ (`backends/cuda/csrc/raco_resblock.cu`, default) | 0.175 ms |
   | TileLang (dropped: one 4-warp block per SM from a 51 KB im2col buffer) | 0.43 ms |

   The first three agree to about 1e-5 relative, and the whole ranker went from 3.0 ms to 2.3 ms to about 1.7 ms per
   image. The CUDA kernel is an implicit GEMM on `mma.sync.m16n8k16`:
   A-fragments are `ldmatrix`-ed straight from the NHWC tile in shared memory (no im2col), in 28 KB of shared memory so
   that several blocks fit per SM, with the shortcut as a tenth tap. It is JIT-built with `nvcc` on first use (cached in
   `~/.cache/flashim/cuda`, no torch headers) and falls back to the Triton kernel without a toolchain; pick one with
   `RaCo.use_fused_ranker(backend=...)` (`"auto"`, `"cuda"` or `"triton"`). The fused ranker is also closer to FP32
   than the compiled BF16 path: 99.8% of keypoints unchanged, against 99.6%. Padded channels and halo roughly double
   the MACs in every version; that, not the kernel language, is the remaining ceiling.

### Accuracy

The whole fast path against the exact FP32 reference, on the test pair:

- 98.5-98.9% of the keypoints are the same pixels.
- RANSAC inliers are within run-to-run noise (73-81 across the FP32 / TF32 / BF16 variants).
- BF16 descriptors give about 9% fewer matches (155 vs 170) at the same inlier count.

### Notes

- **Quantized LightGlue does not help at 1024 keypoints**: `bf16` 13.8 ms, `int8` 14.6 ms, `fp8` 15.3 ms in the
  plain-ALIKED pipeline (measured before each pairing had its own calibration table).
- **LightGlue-ONNX's ModelOpt FP8 engines are not comparable**: they also quantize RaCo (56-59 matches, against about
  165 in FP16) and were calibrated on 3 pairs. Their INT8 quantization was not completed.
- **CUDA streams.** Overlapping the CNNs on streams (`RaCoALIKED.concurrent`, off by default) gave no speedup: the GPU
  is already saturated at 1024x1024. It also exposed that `torchvision.ops.deform_conv2d` ignores the current CUDA
  stream and returns wrong values when its input was produced on a side stream, so ALIKED's encoder stays on the
  caller's stream.
