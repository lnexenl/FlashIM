# FlashIM

Fast, quantized, multi-platform inference for image-matching models.

```python
import flashim

model = flashim.load("romav2", preset="fp8", setting="mega1500", engine="compile")
result = model.match("a.jpg", "b.jpg")                 # DenseMatchResult (warp, overlap, precision)
matches, conf, prec_a, prec_b = model.sample(result, 5000)
kpts_a, kpts_b = model.to_pixel_coordinates(matches, h_a, w_a, h_b, w_b)
```

## Models

| model | presets | notes |
|---|---|---|
| `romav2` | `bf16`, `fp8-p8`, `fp8-s4`, `fp8` (default), `fp8-fast`, `int8`, `int8-dpt`, `int8-full`, `int8-fast` | native port of RoMa v2; `bf16`/`fp8-*` outputs are bit-identical to the RoMaV2 reference implementation |
| `rdd` | `fp32`, `bf16`, `fp8` (default), `fp8-full`, `fp8-backbone` | native port of [RDD](https://github.com/xtcpete/rdd) (CVPR'25, sparse detector + descriptor); `fp32` matches the reference to 1e-5 (see [RDD](#rdd)) |
| `aliked` | `fp32` (default), `bf16` | native port of [ALIKED](https://github.com/Shiaoming/ALIKED) (TIM'23, sparse detector + descriptor); `fp32` matches the reference bit-exactly (see [ALIKED](#aliked)) |

RoMaV2 presets (MXFP8 = E4M3 values, E8M0 scale per 32 elements):

- `fp8-p8` — every DINOv3 / matcher-ViT linear with *static* calibrated activation scales, fused LayerNorm→quant and GELU→quant producers, BF16 SDPA attention, BF16 refiners.
- `fp8-s4` / `fp8` — additionally fuse stride-4 (and stride-2) refiner blocks: one Triton kernel for depthwise 5x5 + bias + ReLU + *dynamic* MXFP8 quantization, then a block-scaled MXFP8 pointwise GEMM.
- `fp8-fast` — `fp8` plus FlashIM refiner kernels: fused BF16 stride-1 conv blocks, a fused refiner
  input (warp sampling, displacement embedding and local correlation in one pass) and fused FP32
  output heads. Not bit-identical to RoMaV2 (reduction order / rounding points); validated by pose AUC.

INT8 presets (RoMaV2's `int8_*` recipes: symmetric per-output-channel int8 weights, *static* per-tensor
activation scales calibrated with the 99.99th-percentile observer over 64 pairs, INT32 accumulation with a
fused dequantize + bias epilogue; Triton GEMM, `torch._int_mm` kept as an alternative implementation):

- `int8` — DINOv3 and matcher-ViT linears; fused LayerNorm→int8 / GELU→int8 producers; BF16 attention and refiners.
- `int8-dpt` — `int8` + the DPT head's four 1x1 input projections.
- `int8-full` — `int8-dpt` + the pointwise conv of every refiner block (depthwise 5x5 + ReLU writes the int8
  activation directly). RoMaV2's `int8_full.yaml`; its `layer*_rn` rules match 3x3 convs and are inert there.

- `int8-fast` — `int8-full` with the `fp8-fast` refiner kernels: INT8 blocks at strides 4/2 only (stride 1 stays
  fused BF16), fused refiner input/heads, batched directions. C=128 blocks run as one fused depthwise + INT8 GEMM
  kernel, C=512 producers as TileLang. Not RoMaV2-exact.

Optional INT8 attention: `flashim.load("romav2", preset="int8-fast", attention="sage-fp8")` (or `"sage"`: INT8 QK +
FP16 PV) runs every quantized transformer block's attention through [SageAttention](https://github.com/thu-ml/SageAttention)
(INT8 Q/K with K smoothing, FP8/FP16 PV). Needs `sageattention` 2.x built for your GPU (not a project dependency;
with GCC 16 build it with clang). `int8-fast` + `compile-full`: 102.7 ms -> 99.7 (`sage`) / 95.4 ms (`sage-fp8`);
AUC@5 vs `bf16` (3 seeds, Mega / ScanNet): `sage` -0.0012 / -0.0002, `sage-fp8` +0.0008 / +0.0010 (noise ~0.003).

Calibrate with `scripts/calibrate_romav2.py --format int8 --preset int8-full [--merge]`; the shipped table is
`flashim/models/romav2/calibration/romav2_int8_mega1500.safetensors`.

Engines: `eager`; `compile` (DINOv3 + matcher ViT, RoMaV2's validated setting); `compile-full`
(also VGG with folded BatchNorm, the DPT head and the refiner conv stacks).

Measured on an RTX 5070 Ti (mega1500 setting, 800² LR + 1024² HR, bidirectional, B=1, preloaded forward):

| configuration | forward | vs BF16 |
|---|---|---|
| `bf16` + `compile` | 344 ms | 1.0x |
| RoMaV2 FastFP8 (s4+s2 fused) + compile | 210 ms | 1.64x |
| `fp8` + `compile` | 204 ms | 1.69x |
| `fp8` + `compile-full` | 165 ms | 2.09x |
| `fp8-fast` + `compile-full` | 110 ms | 3.13x |
| `int8` + `compile` | 324 ms | 1.07x |
| `int8-full` + `compile` | 194 ms | 1.78x |
| `int8-full` + `compile-full` | 152 ms | 2.27x |
| `int8-fast` + `compile-full` | 103 ms | 3.36x |

(`fp8` + `compile` 203 ms and `fp8` + `compile-full` 164 ms in the same session; `int8-fast` is faster than `fp8-fast`.)

Pose AUC@5 change vs `bf16` (3 seeds): `fp8`+`compile` Mega1500 -0.0042 / ScanNet1500 -0.0017;
`fp8`+`compile-full` -0.0020 / -0.0026; `fp8-fast`+`compile-full` -0.0023 / +0.0007
(gate: 3-seed mean >= -0.005; every `fp8-fast` seed is within -0.005).

INT8 pose AUC@5 change vs `bf16` (3 seeds, `compile`, Mega1500 / ScanNet1500): `int8` +0.0031 / -0.0009;
`int8-full` -0.0010 / +0.0000 (per-seed Mega -0.0027..+0.0006), inside the RANSAC seed noise of ~0.003 and
better than `fp8` (-0.0038 / -0.0010); `int8-fast`+`compile-full` -0.0010 / +0.0004. HardMatch not run for INT8.

HardMatch (LoMa, ECCV 2026; 900 hand-labelled hard pairs, mAA@10px, 3 seeds, `wxbs` setting):
`bf16` 0.4786, `fp8` 0.4707 (-0.0079), `fp8-fast` 0.4712 (-0.0074). FP8's cost is larger on these
deliberately hard pairs than on MegaDepth/ScanNet; FlashIM's extra kernels add none of it.
Run with `benchmarks/accuracy/run_hardmatch.py` (official `hardmatch` package protocol).

`sample()` (KDE-balanced match sampling) takes 0.86 ms (RoMaV2: 16 ms, 1.5 GiB scratch); pose AUC
vs RoMaV2's sampling +0.0012 / -0.0010 (3 seeds; `FLASHIM_FAST_SAMPLING=0` restores it); `match(..., num_samples=N)` returns pixel keypoints.

Coordinate heads, the patch embedding, refiner projections and displacement
embeddings are protected: a quantization config that targets them is rejected.

## RDD

```python
model = flashim.load("rdd", preset="fp8", engine="compile")  # checkpoint: see flashim/models/rdd/weights.py
feats = model.extract("a.jpg", resize=1600)               # SparseFeatures: 4096 keypoints + 256-d descriptors
m = model.match("a.jpg", "b.jpg", resize=1600)            # SparseMatchResult (mutual dual-softmax NN)
```

ConvNeXt-tiny backbone -> 5-level multi-scale deformable-attention encoder (4 layers; 140k tokens at 1600 px) ->
descriptor map; a small FP32 conv net produces the keypoint score map. `rdd.pth` is distributed via the Google
Drive link in the RDD README (not auto-downloaded); pass `checkpoint=`, set `FLASHIM_RDD_CHECKPOINT`, or put it at
`<torch hub dir>/checkpoints/rdd.pth`.

- `bf16` — descriptor branch in BF16; detector, softmax and all sampling / position math in FP32.
- `fp8` — `bf16` + the encoder's value / output / FFN linears as static MXFP8, run by a fused encoder pass:
  deformable attention with the softmax fused in (one BF16 GEMM for sampling offsets + attention logits), GEMM-epilogue
  biases, LayerNorm->quant producers, and one fused MXFP8 MLP kernel (`linear1` + bias + GELU + quantize + `linear2`,
  the 1024-wide hidden activation never leaves the SM; bit-identical to the two-GEMM path).
- `fp8-full` — `fp8` + MXFP8 sampling-offset / attention-weight projections (no faster: they share one BF16 GEMM in `fp8`).
- `fp8-backbone` — `fp8` + MXFP8 ConvNeXt MLPs. Slower: those GEMMs are small-K and memory-bound (kept for experiments).

`engine="compile"` compiles backbone, input projections, detector and encoder once per input resolution.

The keypoint detector stays FP32 by default. `detector_dtype="bfloat16"` runs its conv trunk in BF16 (score head FP32):
−0.9 ms compiled (3.7 -> 2.7 ms), but ~12% of the 4096 keypoints change and MegaDepth AUC@5/@10/@20 moved
−0.012/−0.015/−0.014 (eager) and −0.004/−0.007/−0.006 (compile) vs the FP32 detector — inside run noise, but every
paired number went down.

`extract()` at 1056x1600 on an RTX 5070 Ti (image preloaded, keypoints + descriptors included):

| configuration | extract | peak memory |
|---|---|---|
| RDD reference (PyTorch deformable path, no CUDA extension) | OOM (16 GB) | — |
| `fp32` eager | 110.9 ms | 2.0 GiB |
| `bf16` eager | 57.6 ms | 1.1 GiB |
| `bf16` + `compile` | 40.3 ms | 1.1 GiB |
| `fp8` eager | 49.3 ms | 1.3 GiB |
| `fp8` + `compile` | 35.4 ms | 1.1 GiB |

MegaDepth-1500 pose AUC@5 / @10 / @20 (RDD protocol: 1600 px, 4096 keypoints, mutual NN, OpenCV RANSAC; RGB input —
RDD's own script feeds OpenCV BGR, `fp32` 0.5155 that way): `fp32` 0.5216 / 0.6827 / 0.8019, `bf16` 0.5270 / 0.6855 /
0.8031, `fp8` 0.5418 / 0.6974 / 0.8137, `fp8` + `compile` 0.5287 / 0.6887 / 0.8071, `fp8-full` 0.5242 / 0.6801 / 0.7980.
Run-to-run spread of these sparse pipelines is about ±0.01 (`bf16` eager vs `compile` differ by 0.006): FP8 costs no
measurable accuracy. `benchmarks/accuracy/run_sparse.py`, matrix `benchmarks/accuracy/rdd_matrix.sh`.

## ALIKED

```python
model = flashim.load("aliked", preset="fp32", variant="aliked-n16")  # checkpoint: see flashim/models/aliked/weights.py
feats = model.extract("a.jpg")                               # SparseFeatures: threshold-mode keypoints + 128-d descriptors
m = model.match("a.jpg", "b.jpg")                            # SparseMatchResult (mutual nearest-neighbour, cosine >= 0.75)
```

A small conv encoder (plain 3x3 convs at strides 1/2, deformable 3x3 convs at strides 8/32) feeds a multi-scale
feature fusion; a 4-layer conv head produces the keypoint score map and `SDDH` samples descriptors at
deformable offsets around each detected keypoint (DKD: soft-NMS + soft-argmax sub-pixel refinement). Four
released checkpoints (`variant=`): `aliked-t16` (64-d), `aliked-n16` / `aliked-n16rot` (128-d, default),
`aliked-n32` (128-d, 32 sample points). Checkpoints are distributed from the `models/` folder of the ALIKED
repo (not auto-downloaded); pass `checkpoint=`, set `FLASHIM_ALIKED_CHECKPOINT`, or put `<variant>.pth` at
`<torch hub dir>/checkpoints/`.

- `fp32` — the checkpoint's own precision; matches the reference implementation's dense maps, keypoints and
  scores bit-exactly, descriptors to cosine > 0.99999 (deformable-sampling summation order only).
- `bf16` — encoder and `SDDH` in BF16 (deformable convs run their `deform_conv2d` call in FP32: no BF16/FP16
  kernel exists); the score head and NMS / soft-argmax / sampling math stay FP32.

Every preset folds each block's eval BatchNorm into its convolution (`fuse.fold_batchnorm`; exact, not an
approximation — verified <1e-6 max abs diff) and reports the folded blocks as `model.fused_batchnorm`.
`load_aliked(..., channels_last=True)` is available but off by default: measured *slower* on an RTX 5070 Ti
(ALIKED's channel counts are tiny, <=128, and `deform_conv2d` always returns contiguous NCHW regardless of
its input's layout, so cuDNN has no channels-last algorithm to pick for most of the encoder).
`extract()` at 1024x1024 on an RTX 5070 Ti (image preloaded, `aliked-n16`, keypoints + descriptors included):

| configuration | extract |
|---|---|
| `fp32` eager, no BatchNorm fold | 15.3 ms |
| `bf16` eager, no BatchNorm fold | 15.1 ms |
| `bf16` eager, BatchNorm folded (default) | 10.5 ms |
| `bf16` + `compile`, BatchNorm folded | 10.3 ms |

`engine="compile"` gives no further win here beyond BF16 + the BatchNorm fold (this workload is
conv/`deform_conv2d`-bound, which cuDNN already schedules well; unlike RoMaV2's transformer-heavy graph,
there is little for Inductor's fusion to add) — kept available, but eager is the practical default.

No quantized preset by default. ALIKED has no `nn.Linear` layers; the small 1x1 projections
(`conv1..4`, the `downsample`s, `desc_head.sf_conv`, all K/N <= 128) are the only convs the existing
quantization infra can otherwise target, and RDD's `fp8-backbone` preset already found that class of GEMM
(small-K, memory-bound) *slower* to quantize than to leave in BF16 — ALIKED's are smaller still, so it
wasn't re-measured.

The deformable convs (`block3`/`block4`, where most of the runtime actually goes) *were* quantized: a new
op, `ops.deform_patches_int8`, gathers each output pixel's receptive-field patch at its (bilinearly
interpolated, data-dependent) offset and quantizes it to static INT8 in one fused Triton kernel — feeding
the existing INT8 GEMM (`flashim/models/aliked/deform_quant.py`). Correctness is good (feature cosine
> 0.999 vs FP32; the sampling offsets themselves stay FP32 — see the module's docstring for why) and it's
fully calibratable (`DeformConvCalibrator`, `load_aliked(..., deform_int8_calibration=...)`).

First cut was **10-35% slower** than the FP32 `torchvision.ops.deform_conv2d` call it replaces, at every
batch size tried. Profiling (not assumption) showed why: the INT8 GEMM itself was cheap (~7us); the
*gather* kernel was 4x slower than torchvision's, because `x` is NCHW, so gathering one pixel's channels
strides across `H*W`-sized planes with no memory coalescing at all. Converting `x` to channels-last inside
the Triton kernel fixed most of it: on an RTX 5070 Ti, `batch=1` is now within noise of FP32 (~2%), and
`batch>=4` is a clear **win** (25-39% faster, growing with batch). Kept opt-in — worth it for batched
workloads, not for ALIKED's typical single-image `extract()`.

`ops.multiscale_fuse` fuses `dense()`'s multi-scale feature step — 3 bilinear upsamples + a channel concat +
the FP32 cast (`score_head`'s input) + the L2 normalize (the descriptor feature map) — into one Triton
kernel and one pass (safe because `dim // 4 <= 32` for every released variant, so a kernel program already
holds all `4 * dim // 4` output channels of its pixel tile in registers). First cut used FP64 for the
bilinear coordinate math (to match `F.interpolate`'s own precision) and pre-cast every input to FP32 before
the kernel — both measured mistakes: FP64 has very limited throughput on consumer GPUs, and the pre-cast
redid a full memory pass over the biggest input. Fixed (FP32 coordinates, in-kernel upcast-on-load): `bf16`
`extract()` at 1024x1024 (`aliked-n16`) went from 6.97 ms to 5.94 ms; the full ALIKED+LightGlue pipeline
below from 16.6 ms to 14.6 ms. Numerics unaffected (parity tests still bit-exact FP32 / cosine > 0.9999 BF16).

## ALIKED + LightGlue

```python
pipeline = flashim.load("aliked-lightglue", preset="fp32", aliked_preset="fp32")  # LightGlue weights: cvg/LightGlue's raco_aliked_lightglue.pth
m = pipeline.match("a.jpg", "b.jpg", resize=1024, top_k=1024)
```

cvg/LightGlue's ALIKED-trained weights (`input_dim=128 -> dim=256` via `input_proj`, `filter_threshold=0.1`,
see `lightglue.config.ALIKED_CONFIG`); `variant="aliked"` in `load_lightglue()`, or the shortcut
`load_aliked_lightglue()` / `flashim.load("aliked-lightglue")`. `int8`/`fp8` LightGlue presets both load,
but their shipped calibration table was computed on RDD's descriptor distribution, not ALIKED's — usable
for a speed check, not a claim about their quantized accuracy for this pairing (no calibration table exists
for ALIKED yet).

## RaCo + ALIKED + LightGlue

```python
pipeline = flashim.load("raco-aliked-lightglue", preset="bf16", extractor_preset="bf16", engine="compile")
m = pipeline.match(a, b, top_k=1024, threshold=0.1)   # images must be 32-aligned, or pass resize=
```

The `raco_aliked` pipeline of [fabio-sim/LightGlue-ONNX](https://github.com/fabio-sim/LightGlue-ONNX), ported natively
(`flashim/models/raco`). It is **not** the plain ALIKED pipeline above: [RaCo](https://github.com/cvg/RaCo) is a separate
detector network (ALIKED-style CNN -> NMS -> 2K candidates with soft-argmax) plus a **ranker** (9 residual blocks at full
resolution) that reorders the candidates and keeps the top K; ALIKED then only *describes* those keypoints (encoder +
SDDH, no score head/DKD) and LightGlue matches them. Keypoints and ranker scores are bit-exact against their PyTorch
modules in FP32 (identical keypoint set, `tests/parity/raco`), and the full pipeline returns the same 172 matches.

Speed, same 1024x1024 pair, 1024 keypoints, threshold 0.1, RTX 5070 Ti, decode/resize excluded:

| pipeline | ranker | time/pair |
|---|---|---|
| LightGlue-ONNX, TensorRT 11.3 fp16 | dense (exact) | 18.1 ms |
| LightGlue-ONNX, TensorRT 11.3 fp16 | `boundary` (their default: reranks only a 512-candidate window) | 13.8 ms |
| FlashIM, bf16 + `compile` | dense (exact) | **14.6 ms** |
| FlashIM, bf16 eager | dense (exact) | 19.9 ms |
| FlashIM, fp32 eager (TF32 convs) | dense (exact) | 57.3 ms |

On the same (exact) algorithm FlashIM is now ~20% faster than their TensorRT-fp16 engine (their number is engine-only GPU
time; FlashIM's includes Python launch overhead). FlashIM's `boundary` ranker is not implemented, so the 13.8 ms number
still has no like-for-like counterpart. The port started at 88.8 ms; what moved it, in order of size:

- **Fused multi-scale kernel** (`ops.multiscale_fuse`): upsample x3 + concat + the score head's first layer (1x1 conv 128->8
  + SELU, FP32-accurate `tf32x3`) + L2 normalize in one pass, never materializing the 512 MB FP32 concat. The first cut was
  2.3 ms; Nsight Compute showed L1/TEX at 93% (DRAM 5%) because the bilinear gathers read NCHW maps, so each warp
  touched 32 cache lines. Making the coarse maps channels-last: **2.3 -> 0.41 ms** (0.57 ms for ALIKED's descriptor path).
- **BatchNorm folding** (exact) and a **bf16 detector trunk + ranker** under `compile`.
- **Fused ranker blocks**: each residual block is one kernel (input read once, output written once, intermediate kept on
  chip) plus a fused 5x5 reflect head. Three implementations of the same block, all agreeing to ~1e-5 relative, per
  12-channel block at 1024x1024: unfused torch/Inductor ~0.33 ms, Triton 0.237 ms (`ops/_triton/raco.py`; per-tile
  L1/L2 scratch + barriers), **CUDA C++ 0.175 ms** (`ops/_cuda/raco_resblock.cu`, used by default). The CUDA kernel is
  an implicit GEMM with `mma.sync.m16n8k16`: A-fragments are `ldmatrix`-ed straight from the NHWC input/intermediate
  tile in shared memory (no im2col), 28 KB of shared memory so several blocks fit per SM, shortcut as a tenth tap. It is
  JIT-built with `nvcc` on first use (cached in `~/.cache/flashim/cuda`, pure CUDA with no torch headers) and falls back
  to the Triton kernel when there is no toolchain (`RaCo.use_fused_ranker(backend="auto"|"cuda"|"triton")`). Ranker:
  3.0 -> 2.3 -> ~1.7 ms per image, and closer to FP32 than the compiled bf16 path (99.8% of keypoints unchanged vs 99.6%).
  A TileLang version (correct, but 0.43 ms: one 4-warp block per SM from a 51 KB im2col buffer) was tried and dropped.
  Padded channels/halo make ~2x the MACs wasted in every version; that, not the language, is the remaining ceiling.

Accuracy of the whole fast path against the exact FP32 reference on the test pair: 98.5-98.9% of the keypoints are the
same pixels, RANSAC inliers are within run-to-run noise (73-81 across FP32/TF32/bf16 variants), and bf16 descriptors give
~9% fewer matches (155 vs 170) at the same inlier count.

Overlapping those CNNs on CUDA streams (`RaCoALIKED.concurrent`, off by default) gave no speedup: the GPU is already
saturated at 1024x1024. It also exposed that **`torchvision.ops.deform_conv2d` ignores the current CUDA stream** and
returns wrong values when its input was produced on a side stream (ALIKED's `block3`/`block4`), so ALIKED's encoder is
kept on the caller's stream.

INT8/FP8: LightGlue's quantized presets do not help at 1024 keypoints (bf16 13.8 ms vs int8 14.6 ms vs fp8 15.3 ms in
the plain-ALIKED pipeline) and use a calibration table computed on RDD descriptors. LightGlue-ONNX's own ModelOpt FP8
engines are not comparable: they also quantize RaCo (56-59 matches vs ~165 in FP16) and were calibrated on 3 pairs.
Their INT8 quantization was not completed.

Getting their engines running needed `onnxslim` (constant-folds an axis the exporter leaves as a `Reshape`, which
TensorRT 11's stricter parser rejects) and, for FP16, exporting with `--fp16` despite the tool's stale warning
(TensorRT >= 10 is strongly typed: precision comes from the graph's dtypes). ONNX Runtime's TensorRT execution
provider wants their pinned `libnvinfer.so.10`, so engines were built and validated with `trtexec` directly.

## Layout

```
flashim/
  platforms/   device + feature detection (triton, fp8, mxfp8 scaled_mm, ...)
  ops/         every op = torch reference + optional per-platform impls (Triton, cuBLAS, fused_local_corr)
               files by computation (attention, conv, correlation, density, gemm, quantize);
               model-specific ops in a file named after the model (romav2.py); _triton/ mirrors it
  quant/       formats (MX), QuantizedTensor, schemes (compressed-tensors vocabulary), calibration,
               methods (StaticMXFP8Linear), kernels (cuBLAS MXFP8 > Triton FP8 > emulation), fusion passes
  engines/     region torch.compile, experimental CUDA-graph runner
  models/      romav2 (DINOv3 ViT-L/16, MV-ViT, DPT head, conv refiners, presets, calibration)
               rdd (ConvNeXt, deformable encoder, detector, sparse keypoints / matching, presets, calibration)
               aliked (deformable-conv encoder, SDDH descriptor head, DKD keypoints / matching, presets)
  geometry/    pose estimation + AUC metrics
benchmarks/    speed/forward.py, accuracy/run.py (MegaDepth-1500, ScanNet-1500), accuracy/run_sparse.py (RDD)
scripts/       calibrate_romav2.py, calibrate_rdd.py
tests/         unit, ops (impl vs reference), quant, parity (vs the RoMaV2 package, ~/work/RDD, ~/work/ALIKED)
```

Op overrides: `FLASHIM_OPS="rope_qk:reference,-local_corr"`; `FLASHIM_STRICT_OPS=1` raises
if a GPU call falls back to a torch reference.

## Platforms

FP8 presets run natively on Blackwell (sm_100 / sm_120) through `F.scaled_mm`
block-scaled MXFP8. Elsewhere they fall back to a Triton FP8 GEMM (sm_89+) or to
emulation kernels (any device, including CPU) — same numerics contract, slower.

## Development

```bash
uv sync --extra eval                     # torch 2.11 (cu130), triton 3.6, fused-local-corr
uv run pytest -m "not slow"              # unit / ops / quant
uv run pytest tests/parity               # bit-exactness vs ../RoMaV2 (needs weights + GPU)
uv run python benchmarks/speed/forward.py --preset fp8 --engine compile
uv run python benchmarks/accuracy/run.py --dataset mega1500 --preset fp8 --seed 17
```

The DINOv3 weights inside the RoMaV2 checkpoint are covered by the DINOv3 License.
