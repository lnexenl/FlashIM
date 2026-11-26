# Changelog

## Unreleased

### Changed

- PyTorch is no longer pinned to 2.11: `torch>=2.10`, `torchvision>=0.25`, and no explicit `triton` requirement (CUDA
  torch wheels pin their own; Jetson keeps Triton 3.6 in `requirements-jetson.txt`). The floor is measured: torch 2.9's
  Triton 3.5 fails a FlashIM kernel on `sm_100`. On Blackwell, `fp8` with torch < 2.10 builds the nearest preset
  (`build_report.fallbacks` names the installed version). Measured numbers remain those of the torch 2.11 setup.

### Added

- CI compiles every CUDA C++ kernel with nvcc 12.6 / 12.8 / newest 13.x, every recorded Triton specialization
  (`ci/triton.jsonl`) and the TileLang kernels for each supported NVIDIA architecture, without a GPU; CPU tests run on
  torch 2.10 / 2.11 / newest, plus a weekly nightly-torch job.

- `flashim.model_info(name)`: code and weights license IDs of a model's default configuration (every component of a
  pipeline) and `commercial_ok` (only Apache-2.0 / BSD-3-Clause / MIT; unknown or restricted terms are `False`).
  [docs/licenses.md](docs/licenses.md) is generated from it. Weights published in the upstream repository inherit its
  license, with the basis recorded in each `weights.py`.
- A one-time warning per model family before downloading non-commercial weights (Magic Leap SuperPoint / SuperGlue,
  Fraunhofer academic RIPE / RIPE++, CC-BY-NC-SA MASt3R / DUSt3R / UFM).

### Fixed

- `engine="compile"` on model / preset combinations known to crash in Dynamo or Inductor (INT8 / FP8 `alike`, `xfeat`,
  `dedode*`, `dad`, `ripe*`, `loma-*`, `xoftr`; BF16 `ripe*`) no longer fails at the first `match` / `extract`: the
  build runs `eager` with a warning and a `<path>.engine` entry in `build_report.fallbacks`, or raises with
  `strict=True`. The failures are declared in each model's `spec.py` (`ModelSpec.compile_failures`), measured on torch
  2.11 / `sm_120`, and apply on every CUDA architecture and torch version until re-measured.
- Warnings and errors (backend load failures, missing `nvcc`, preset and engine fallbacks) reach stderr when the
  application has not configured logging: the `NullHandler` on the `flashim` logger is gone.
- `romav2` honours `config.checkpoint` (`overrides={"checkpoint": path}`) like every other model; it used to load the
  default checkpoint regardless.

## 0.1.1 (2026-11-26)

### Added

- RIPE++ (`ripepp`, `ripepp-aachen`, `ripepp-scared`, `ripepp-lightglue`; Fraunhofer HHI, LIMIT@ECCV 2026; non-commercial
  academic-use weights): native PyTorch port reusing RIPE's VGG19 + ConvRefiner backbone and exact `sm_120` INT8/BF16
  fusion passes, adding ImageNet normalization, 3×3 softmax subpixel refinement, `align_corners=False` hypercolumn
  sampling, pixel-center keypoint conversion (`matcher_keypoint_offset=0.5`), and the published weakly-supervised
  LightGlue matcher (`ripe++lightglue.tar`). `ripepp` ships `fp32` / `bf16` (default) / `int8` (31/35 convs INT8,
  `net.decoder.layers.{1,2}.*` kept BF16 for subpixel keypoint accuracy) / `fp8` (28/35 convs MXFP8,
  `net.encoder.layers.{3,7,10}` + `net.decoder.layers.{1,2}.*` BF16); `ripepp-aachen` and `ripepp-scared` ship `fp32` /
  `bf16`; `ripepp-lightglue` ships `fp32` / `bf16` (default) / `int8` (`ripepp` `int8` + `lightglue` `bf16`, `fp8 ->
  int8` via `build.NEAREST`). Verified on RTX 5070 Ti (`sm_120`, `eager`, 1200 px / 4096 kp): `ripepp int8` extract
  `20.4 ms` and `fp8` `28.8 ms` vs `bf16` `33.7 ms` (1.65x / 1.17x), `ripepp-lightglue int8` pair `52.6 ms` vs `bf16`
  `79.4 ms` (1.51x); MegaDepth-1500 / ScanNet-1500 paired-bootstrap gate (3 seeds, PoseLib 1.0 px, vs own `bf16`, 2 PASS
  / 4 BORDERLINE / 0 FAIL): `ripepp` `bf16` `0.5877 / 0.1620`, `int8` `-0.0040 / +0.0020`, `fp8` `+0.0004 / -0.0012`;
  `ripepp-lightglue` `bf16` `0.6057 / 0.1495`, `int8` `-0.0011 / -0.0009`.
- FlexGlue (`flexglue`, cvg/FlexGlue v0.1): sparse-to-dense matcher with a DINOv3-ConvNeXt descriptor.
  `match_keypoints` matches any extractor's keypoints, `match` samples the dense warp with upstream's gates. Presets
  `fp32` / `bf16` / `int8` / `fp8` (default: MXFP8 transformer linears + INT8 descriptor MLP); `int8` / `fp8` run
  SageAttention (pass `sage_attention`: `sage` / `sage-fp8-fast` on sm_89 and sm_120, `sage` also on sm_80-87;
  torch SDPA on Hopper, where the sm_89 kernels are slower); opt-in early stopping and `arch.num_layers` truncation.
- `optimize.attention="sage-fp8-fast"`: `sage-fp8` with FP16 PV accumulation (SageAttention2++), sm_89+.
- Quant configs may mix MXFP8 and INT8 schemes: `quantize_model` takes one table of each kind, a spec's default
  calibration may be a tuple of files, and a table passed with `calibration=` replaces only the default of its kind.
- Ten new image-matching model families (all verified against upstream parity oracles with `fp32` / `bf16` / `int8` / `fp8` presets and custom Triton / CUDA C++ kernels):
  - Dense matchers: `roma`, `roma-indoor`, `tiny-roma`, `ufm`, `mast3r`, `dust3r` (`mast3r` and `dust3r` use FP16 compute under `bf16`/`int8`/`fp8` to avoid BF16 ULP loss on CroCo's ~344–649 residual peaks).
  - Detector-free / cross-modal matchers: `eloftr`, `xoftr`, `minima-loftr`, `minima-roma`.
  - Sparse matchers & pipelines: `superglue`, `superpoint-superglue`, `omniglue`, `omniglue-extractor`, `omniglue-matcher` (adding SuperPoint `sp_v6` variant).
- Shared ops & quantization infrastructure:
  - Unified register-resident + L2-streaming CUDA Sinkhorn library (`csrc/sinkhorn.cu`, `ops.log_optimal_transport` and `ops.sinkhorn_mutual`); CUDA BVH 3D reciprocal NN (`csrc/reciprocal_nn.cu`, `ops.reciprocal_nn`); Triton ops for 2D/interleaved RoPE, ELU linear attention, bilinear upsampling (+ INT8 quantize), 2D box NMS, ReLU-affine, window gather, and fused LayerNorm/GELU/rotation per-row dynamic INT8 quantization (`layernorm_orthogonal`, `signed_hadamard`, FP16 `DynamicInt8Linear` / `DynamicInt8TransformerBlock`).
  - Dynamic-only quantization configs skip calibration table loading unless explicitly supplied or registered (`calibration_metadata = None`).
  - `ModelSpec.quant_configs` per-variant `<variant>/<quant>` lookup (mirroring `ModelSpec.calibrations`, used by `loma-b128` `fp8`), `extractor_spec` `fp8_ignore` and `cuda_graph` options, and resumable pre-2026-11 accuracy matrix tooling (`benchmarks/accuracy/perf_matrix.sh` with `SKIP_SPECS`, `perf_protocol.py`, `perf_report.py`).

### Changed

- **Intended output improvement (ALIKED family)**: keeping SDDH pixel-space keypoint coordinates in FP32 under `bf16` fixes 2–8 px BF16 ULP quantization on `kpts_wh`, lifting MegaDepth-1500 / ScanNet-1500 pose AUC@5 (RTX 5070 Ti `sm_120`, `eager`, 3 seeds, PoseLib 1.0 px) on `aliked bf16` (`+0.0559` / `+0.0068`), `raco-aliked bf16` (`+0.0397` / `+0.0056`), `aliked-lightglue bf16` (`+0.0126` / `+0.0073`), and `raco-aliked-lightglue bf16` (`+0.0186` / `+0.0122`).
- **Preset fallbacks via `build.NEAREST`** (presets that lost pose AUC or were Pareto-dominated by the fallback are omitted and recorded in `model.build_report.fallbacks`):
  - `fp8 -> int8`: `dedode` (`dedode-g` keeps `fp8`), `aliked-lightglue`, `superpoint-lightglue` (runs SuperPoint in `bf16` + LightGlue in rotated `int8`), `disk-lightglue`, `superpoint-superglue`, `superglue`, `omniglue`, and `omniglue-matcher`.
  - `int8` / `fp8 -> bf16`: `xfeat-lightglue`.

### Fixed

- `engine="compile"` crashed for `disk` and `disk-lightglue` (`bf16` / `int8` / `fp8`): torch 2.11 Inductor passes non-constexpr Python float arguments of Triton kernels as fp64, so the InstanceNorm+PReLU kernels (`eps`) failed with `Expected dtype ['fp32'] but got fp64`. The float scalars of the Triton kernels in `norm.py`, `resample.py`, `correlation.py`, `quantize.py` and `_layernorm_stats.py` are cast to fp32 inside the kernel (eager PTX unchanged; `rgb_mean_instance_norm`, `upsample_concat_int8` and `local_corr` also crashed under compile, ten more kernels were silently promoted to fp64). `ripe` `engine="compile"` still fails (`KeyError` in the Inductor scheduler, also on the previous release).
- `decode_jpeg_cuda`: fixed stream/allocator race in nvJPEG GPU decoding causing intermittent pixel corruption (`fix/jpeg-decode`).
- `tests/ops/test_depthwise_cuda.py` on `sm_120`: swept and enabled `PRODUCER_TILE` / `DWPW_TILE` tables and `cuda_composite` (`51/51` passing, bit-identical to Triton; unmeasured `sm_100` / `sm_110` keep fallbacks).
- Bounded cuBLAS MXFP8 `_InputScales.by_rows` (16-entry LRU, pinning CUDA-graph-captured row counts) and LoFTR `_pe` / `_inner_cache` (fixes LoFTR `fp8` MegaDepth-1500 OOM at pair 1417).
- Guarded LightGlue / `SparseExtractor` CUDA graph captures (`hits >= 2` repeat-shape gate, `max(N0, N1) <= 8192` guard, pool release outside `except` before eager fallback) and added `sdpa_graph_safe` probe so unpatched SageAttention never captures corrupt graphs.
- Quantization accuracy & numerics fixes (verified on RTX 5070 Ti `sm_120` MegaDepth-1500 & ScanNet-1500, 3 seeds, paired bootstrap gate `-0.005` AUC@5, **84 PASS, 42 BORDERLINE, 0 FAIL**):
  - Targeted BF16 layer exclusions: `xfeat` protects `keypoint_head.*` and `block2.*` (`int8`/`fp8`); `superpoint` keeps `conv1b` in BF16 (`fp8`); `dedode` excludes `descriptor.encoder.layers.{3,7,10}` (`int8`); `ripe` excludes `net.encoder.layers.{3,7,10}` (`fp8`); `loma-b128` keeps `*.decoder.*` and matcher `transformers.{4,5}.*` in BF16 (`fp8`).
  - Restored intermediate BF16 round in `ops.instance_norm_prelu`, exact 9-tap FMA in `ops.conv3x3_small_cin_relu` (`Cin=1`), IEEE FP32 in `ops.mutual_nn_match` for `M*N <= 8192^2`, FP32 similarity in `DeDoDe.match_features`, and exact-only INT8 passes on RIPE.

### Performance

- **Broad `sm_120` (RTX 5070 Ti) kernel, fusion, and dispatch sweep** across all 26 pre-2026-11 models (measured on RTX 5070 Ti `sm_120`, `eager`; non-`sm_120` CUDA C++ / Triton kernels compile-checked only):
  - **Shared runtime & ops**: cached per-platform dispatch and `BoundOp._usable` in `ops/_registry.py`; enabled `sm_120` for `depthwise.cu` (`PRODUCER_TILE`, `DWPW_TILE`, `cuda_composite`), `upsample_concat_int8` / `upsample_add` `triton_rows`, `conv2d_small_cin_quantize_int8`, `patch_embed_tokens` cuBLAS GEMM, `layernorm_quantize_int8_rows`, `layernorm_gelu_quantize_int8`, 2-pass `dual_softmax_mutual`, and INT8/MXFP8 conv chaining (`QUANT_OUT`); added exact 2x BF16 `upsample_concat` `triton_rows`, `ops.quantize_int8_rows_heads` / `ops.gelu_quantize_int8_rows` + zero-copy NHD FlashAttention layout, CUDA C++ `ops.simple_nms`, `ops.aliked_score_head`, `ops.raco_score_head`, and Triton `ops.hamming_distances`, `ops.hamming_ratio_mutual_match`, `ops.ratio_mutual_match`, `ops.conv3x3_small_cin_relu`, `ops.int8_conv2d_maxpool_only`, `ops.multiscale_score`, `ops.instance_norm_prelu`, `ops.rgb_mean_instance_norm`, `ops.conv2d_at_points`, `ops.strict_nms_mask`, `ops.bilinear_sample_points`, `ops.mutual_nn_match`.
  - **RoMaV2 (RTX 5070 Ti `sm_120`, `eager`, `mega1500` 800² + 1024², both directions, B=1)**: preserved `rope_impl` through quantization (`42/42` Triton RoPE), enabled `fold_fine_features`, `fold_layer_scale`, `fused_block_epilogues`, `batched_matcher` (reusing `logits_ba = logits_ab^T` + batched B=2 `DPTHead` in channels-last), `sage_attention`, and `depthwise.cu` (`fp8` 184.07 -> 113.94 ms [1.62x], `int8` 139.75 -> 110.83 ms [1.26x]).
  - **LoMa (`loma-*`, RTX 5070 Ti `sm_120`, `eager`)**: enabled all DeDoDe/LoMa extractor fusions + DINOv2 `sage-fp8-warp` (`int8` / `fp8`) + matcher `cuda_graph`, `fused_bias` (`fp8`), and `sage-fp8` (`dim >= 512`): `loma-b fp8` matcher (N=2048) 7.21 -> 3.52 ms (2.05x), `loma-l int8`/`fp8` matcher (N=4096) 20.32 -> 12.97 ms / 26.22 -> 18.61 ms, `loma-g int8`/`fp8` matcher (N=4096) 43.57 -> 29.25 ms / 65.58 -> 39.30 ms, `loma-b128` 1024 px pair (4096 kp) `fp8` 40.11 ms vs `bf16` 51.17 ms (1.28x).
  - **LightGlue (RTX 5070 Ti `sm_120`, `eager`)**: repeat-shape `cuda_graph`, `low_precision_similarity`, `fused_bias` (`fp8`), batched `CrossBlock.to_out`, `rotate_residual_stream` (Paley-12 Hadamard for `dim=96`) + SmoothQuant `ffn.3` + `fused_int8_glue` (`rdd` N=1024 `fp8` 7.09 -> 1.56 ms, `int8` 6.29 -> 1.48 ms; `aliked` N=2048 `fp8` 7.07 -> 3.44 ms, `int8` 6.03 -> 3.36 ms; `rdd` N=4096 `fp8` 13.01 -> 10.24 ms, `int8` 12.60 -> 9.90 ms).
  - **LoFTR (RTX 5070 Ti `sm_120`, `eager`, diff pair 896x1200 / 880x1200)**: unified 2-kernel Triton `ops.elu_linear_attention` (`N*H` on `gridDim.x`), `low_precision_similarity`, `quantize_int8_multi` QKV, fused MLP ReLU, same-shape self-attention batching, indexed 5x5 fine gather (-214 MB VRAM): `bf16` 94.03 -> 79.62 ms, `int8` 60.73 -> 37.99 ms (1.60x), `fp8` 57.91 -> 45.81 ms (1.26x).
  - **Sparse extractors (RTX 5070 Ti `sm_120`, `eager`)**: `orb` `fp32` match (4096x4096) 34.06 -> 0.31 ms (109.1x), `sift` `fp32` match (4096x3629) 1.34 -> 0.85 ms; `superpoint` extract (856x1200, 4096 kp) `bf16` 8.15 -> 4.42 ms, `int8` 6.03 -> 1.35 ms, `fp8` 6.09 -> 3.82 ms; `alike` extract (832x1184, 4096 kp) `bf16`/`int8`/`fp8` 7.17–7.59 -> 1.78–2.09 ms (3.44–4.11x); `aliked` extract (832x1184, 4096 kp) `bf16` 5.42 -> 3.47 ms, `fp32` 6.36 -> 5.59 ms; `raco` `bf16` `extract_keypoints` (1024x1024) 6.92 -> 3.67 ms (1.88x); `disk` extract (1200 px, 4096 kp) `bf16`/`int8`/`fp8` 12.73–15.40 -> 5.75–7.47 ms (1.74–2.68x); `xfeat` extract (1200 px, 4096 kp) `bf16`/`int8`/`fp8` 2.01–2.44 -> 1.13–1.42 ms (1.53–1.99x); `rdd` match (4096x4096, `bf16`) 5.76 -> 0.75 ms (7.67x) and extract (1600 px) `bf16` 62.72 -> 56.92 ms, `fp8` 53.59 -> 47.76 ms; `dedode` extract (1200 px, 4096 kp) `int8` 30.37 vs `bf16` 55.94 ms (1.84x); `dedode-g` extract (1200 px) `int8` 101.55 -> 76.14 ms, `fp8` 116.36 -> 101.22 ms; `dad` extract (1200 px) `int8` 136.5 -> 78.47 ms, `fp8` 102.79 -> 85.97 ms; `ripe` extract (1200 px) `int8` 42.28 -> 19.19 ms (`_describe` 4.8x, -492 MB VRAM) and `fp32` match (32320x23731x256) 22.54 -> 9.81 ms (-2.97 GB VRAM).
  - **FlexGlue (RTX 5070 Ti `sm_120`, `eager`, 512x512 descriptor, MegaDepth pair `505127/505128`, 2-round interleaved best-of-2)**: added Triton `ops.flexglue_entmax_readout` (zero-sort safeguarded Newton + active-set quadratic), `ops.qkv_norm_rope_interleaved`, `ops.qk_norm_swap_heads`, `ops.quantize_mxfp8_heads`, 3D `ops.kde_density`, `padding_mode="border"` on `ops.bilinear_sample_points`, `input_bias` on `ops.layernorm_cat_quantize_mxfp8`, and low-D `triton_tiled` on `ops.reciprocal_nn`; added ConvNeXt `fold_convnext_gamma`, `channels_last`, `fuse_convnext_int8`, `fuse_transformer_glue` (MXFP8 cuBLAS fused bias + fused head/LN-cat/FFN producers), dense `MatchHead` reuse (`logits_10 = logits_01^T`), and fixed-shape descriptor `cuda_graph`: `match_keypoints` (2048 RDD kp) `fp8` 32.30 -> 18.58 ms (1.74x), `int8` 35.47 -> 22.97 ms (1.54x), `bf16` 45.76 -> 37.75 ms (1.21x), `fp32` 163.42 -> 153.86 ms (1.06x); dense `match()` (5000 samples) `fp8` 36.85 -> 19.50 ms (1.89x), `int8` 40.35 -> 25.67 ms (1.57x), `bf16` 50.45 -> 38.78 ms (1.30x), `fp32` 168.08 -> 154.86 ms (1.09x). MegaDepth-1500 / ScanNet-1500 accuracy gate (3 seeds, PoseLib 1.0 px, paired bootstrap vs `75077dc bf16`, 5 PASS, 1 BORDERLINE, 0 FAIL): `bf16` `-0.0005 / -0.0002`, `int8` `-0.0012 / -0.0010`, `fp8` `-0.0019 / +0.0009`.

- LoMa `int8` (`loma-*` pipelines, sm_89 / sm_90), from the LoMa tuning prototypes:
  - extractor: the DINOv2 ViT as per-token dynamic INT8 and every DeDoDe int8 pass (decoder fusions, encoder stem,
    fused ViT glue, patch-embedding GEMM, SageAttention on ViT blocks 2-23); `loma-l` keeps the bit-exact stem and runs
    SageAttention on blocks 4-23 only;
  - matcher: SageAttention (pass `sage_attention`: `sage-fp8` on sm_89, `sage-fp8-sm90` on sm_90), the INT8
    transformer glue fused (pass `fused_int8_glue`, `models/lightglue/int8_glue.py`, bit-identical: one residual stream
    for both images, one-read multi-quantize, RoPE / head scatter and the image swap in the GEMM epilogues, the attention
    output quantized from its head layout, the message quantized into the FFN input, LayerNorm + GELU + INT8 in one
    kernel; new ops `quantize_int8_multi`, `quantize_int8_heads`, `layernorm_gelu_quantize_int8`,
    `int8_linear_quantize`, `lightglue_qkv_rope`, `lightglue_swapped_linear`) and the transformer + assignment head
    replayed from one CUDA graph per keypoint-count pair (pass `cuda_graph`, eager engine; it falls back to eager with a
    warning if capture fails, e.g. with a SageAttention build that launches on the legacy default stream).
  Pair (2 extracts from the JPEG at 1024 px, 4096 keypoints, + match; eager, interleaved best of 3), int8 before ->
  after: L20 b / l / g / r / b128 115.5 / 123.1 / 148.2 / 115.3 / 49.2 -> 74.0 / 82.5 / 99.0 / 74.1 / 35.3 ms
  (1.68-1.77x faster than bf16); H20 102.4 / 106.6 / 126.8 / 101.7 / 46.8 -> 69.9 / 77.0 / 91.4 / 69.8 / 33.0 ms
  (1.53-1.65x). MegaDepth-1500 AUC@5 vs bf16 (OpenCV RANSAC 0.5 px, 10 seeds, paired bootstrap): L20 -0.22 / -0.30 /
  -0.10 / -0.25 / -0.27, H20 -0.09 / +0.23 / -0.03 / +0.12 / -0.05.
- `CudaGraphRunner.capture` replays the graph once before returning: its outputs held nothing (capture only records
  the kernels).

- DeDoDe `int8` (ported from the LoMa experiments): `dedode-g` quantizes its DINOv2 ViT with per-token dynamic INT8
  (`DynamicInt8Linear`: `ops.quantize_int8_rows` + `ops.int8_linear` with per-row scales) instead of leaving it in BF16,
  and both models fuse their ConvRefiner decoders (depthwise 5x5 + ReLU as the INT8 producer of the 1x1 conv,
  upsample-concat writing the INT8 refiner input, the residual's 1/1.4 folded into `out_conv`). Extract at 1200 px on
  an RTX 5070 Ti (eager): `dedode` 46.4 -> 33.9 ms, `dedode-g` 139.9 -> 87.1 ms (on par with its `fp8`); MegaDepth-1500
  AUC@5 vs `bf16` +0.0004 / +0.0025.
  `extractor_spec(int8_dynamic=...)` and the `INT8_TOKEN_ACTIVATIONS` scheme make per-token INT8 available to other
  models; `Op.supports(...)` tells whether a call would run a kernel rather than the reference.
- `dedode-g` `int8`: the ViT's glue ops are fused, bit-identically (`DynamicInt8TransformerBlock`, pass
  `fused_dynamic_vit_blocks`): LayerNorm and the per-row INT8 quantize in one kernel (`ops.layernorm_quantize_int8_rows`,
  whose statistics replicate ATen's LayerNorm reduction tree), GELU in `fc1`'s GEMM epilogue and the residual adds in
  `proj`'s / `fc2`'s (`ops.int8_linear(gelu=True, residual=...)`; GELU only where a `libdevice` with torch's `erf` is
  installed, probed once per process). 16 kernels per block become 9. Extract at 1200 px (eager, interleaved): L20
  104.9 -> 102.6 ms, H20 93.5 -> 90.1 ms (with the next item); ViT alone in a CUDA graph 49.6 -> 47.4 ms (L20),
  40.1 -> 38.4 ms (H20). The layer-norm kernel was initially registered for sm_89 / sm_90 only (subsequently enabled on
  sm_120 in the sweep above); other GPUs run the unfused modules.
- INT8 GEMM on sm_90: 64 x 64 / 4-warp tiles for the decoders' small-N 1x1 convolutions (`n < 512`), exact under INT32
  accumulation; dedode-g's decoder GEMMs take 0.82 ms less per extract on an H20.
- Decoder upsampling at the memory roofline on sm_89 / sm_90 (bit-identical): `ops.upsample_concat_int8` has a second
  Triton impl, `triton_rows` (one program per pair of output rows, constexpr channel counts, vectorized loads, shared taps
  and horizontal lerps), chosen for exact 2x upsampling of BF16 maps; `ops.upsample_add` has a row kernel for wide
  channels-last maps. Other ratios (the DINOv2 descriptor's 56 -> 98 map), FP32 taps, narrow or NCHW maps and other
  architectures keep the tile kernel. Per extract at 1200 px, 4096 keypoints (CUDA graph, DRAM-bound), `dedode`:
  L20 upsample-concat 2.97 -> 1.54 ms, upsample-add 1.12 -> 0.89 ms; H20 3.76 -> 0.44 ms and 0.29 -> 0.25 ms; int8 pair
  (2 x extract + match) L20 68.8 -> 66.4 ms, H20 62.0 -> 57.5 ms.
- DeDoDe `int8` encoders (bit-identical): each `INT8 conv -> ReLU -> MaxPool2d -> INT8 conv` runs as one kernel that
  also writes the pooled map's INT8 input (`ops.int8_conv2d_maxpool`, pass `fused_pool_epilogue`), the BF16 first
  conv's bias + ReLU + quantize are one pass after cuDNN (`ops.bias_relu_quantize_int8`, pass
  `fused_first_conv_epilogue`), and `blocks.VGG` no longer runs its last max pool, whose output is never read (every
  model built on it). Extract at 1200 px, 4096 keypoints (eager, interleaved): L20 `dedode` 33.9 -> 30.5 ms,
  `dedode-g` 83.9 -> 80.6 ms; H20 30.5 -> 29.2 ms, 71.6 -> 70.4 ms.
- CUDA C++ depthwise 5x5 kernels on sm_89 / sm_90 (`backends/cuda/csrc/depthwise.cu`): the INT8 producer of the
  fused ConvRefiner blocks (`ops.depthwise5x5_relu_int8`, any `C % 64 == 0`, so the 256 / 768-channel blocks fuse on
  Ada too) and the fused BF16 depthwise + pointwise block (`ops.depthwise5x5_relu_pointwise`, 32 / 64 / 128 channels),
  bit-identical to the Triton kernels they replace. Extract at 1200 px, 4096 keypoints (eager, interleaved): `dedode`
  int8 L20 33.9 -> 31.2 ms, H20 30.5 -> 25.7 ms; `dedode-g` int8 L20 84.0 -> 81.0 ms, H20 71.4 -> 66.4 ms; bf16
  (fused block only) L20 -0.6 ms, H20 -4 ms.
  Outputs are bit-identical except `dedode` / `dedode-g` `int8` on the L20, where the 256 / 768-channel blocks were
  unfused before: MegaDepth-1500 AUC@5 there vs `bf16` +0.19 / -0.27 points (-0.06 / -0.11 vs the previous `int8`).
- `dedode-g` `int8` on sm_89 / sm_90: SageAttention (QK INT8 with per-warp scales, PV FP8) on DINOv2 blocks 2-23 (pass
  `vit_sage_attention`; `optimize.options.vit_attention = {"layers": "2-23", "impl": "auto"}` picks the layers and the
  `ops.sdpa` impl, blocks 0 and 1 stay on torch SDPA because the ViT output is most sensitive there). New `ops.sdpa` impls
  `sage-fp8-warp` and, for the H20, `sage-fp8-sm90` / `sage-fp8-sm90-warp` (the WGMMA kernel; `sage-fp8` keeps running the
  sm_89 kernel everywhere). Extract at 1200 px (eager): L20 105.0 -> 93.7 ms, H20 93.5 -> 87.8 ms; MegaDepth-1500 AUC@5 vs
  `bf16` -0.12 (L20) / +0.23 (H20) points, +0.05 / +0.07 vs the unchanged `int8`.
- DeDoDe `int8` on sm_89 / sm_90 (L20 / H20; numerics, MegaDepth-1500 AUC@5 within 0.5 of `bf16`): the 3 -> 64 first
  conv of both VGG encoders runs on tensor cores with the bias / ReLU / quantize epilogue
  (`ops.conv2d_small_cin_quantize_int8`, pass `tensor_core_first_conv`; 1.46 -> 0.06 ms per encoder on an L20) and
  `dedode-g`'s DINOv2 patch embedding is one GEMM into the token buffer (`ops.patch_embed_tokens`, pass
  `patch_embed_gemm`). Extract at 1200 px: L20 `dedode` 30.5 -> 29.2 ms, `dedode-g` 80.8 -> 79.4 ms; H20 29.3 -> 28.4 ms,
  70.4 -> 69.0 ms.
- Triton INT8 depthwise producer (`ops.depthwise5x5_relu_int8`) accepts any channel count `% 32 == 0` (was 32 / 128 /
  512), so the 256 / 768-channel ConvRefiner blocks of the DeDoDe `int8` presets fuse on architectures whose only
  producer is Triton (no TileLang / TMA, no CUDA C++ impl). Measured on L20 / H20 against the unfused block only; draft.

## 0.1.0 (2026-10-28)

First public release.

### Interface

- One loader: `flashim.load(name, preset, *, device, engine, strict, calibration, config, overrides)`. Model-specific
  settings are config keys (`overrides={"arch.setting": "mega1500"}`); the per-model `load_*` functions are gone.
- One matching call: `model.match(a, b, *, resize, max_matches, threshold)` returns `flashim.Matches` for every model
  (dense, detector-free, detector + descriptor, LightGlue pipelines). Dense models attach the full prediction as
  `Matches.dense`; RoMaV2's `match_dense` returns only that. Per-call defaults come from the config's `inference`
  section.
- Four presets: `fp32`, `bf16`, `int8`, `fp8`. A model without the asked preset, and `fp8` on a GPU without FP8,
  builds the nearest one (`fp8` -> `int8` -> `bf16` -> `fp32`) and records it in `model.build_report.fallbacks`. A
  pipeline's preset applies to each of its components.
- Keypoints of every model are `(x, y)` pixel-center coordinates of the original image, RaCo's included (RaCo's own
  pixel-edge convention, pixel index + 0.5, is only what LightGlue sees in the `raco-*-lightglue` pipelines).
- `flashim.resolve` returns the config a build would use; `model.run_config` and `model.build_report` say what was
  asked for and what was applied.

### Models

- Dense: RoMa v2 (`bf16`, `int8`, `fp8`; `compile-full` engine).
- Sparse: RDD, ALIKED, RaCo (+ ALIKED or any descriptor model), SuperPoint, ALIKE, DISK, XFeat, DeDoDe(-G), DaD,
  RIPE, SIFT, ORB; semi-dense LoFTR; LightGlue pipelines for RDD, ALIKED, RaCo + ALIKED, SuperPoint, DISK and XFeat.
- Every learned model is checked against its vendored upstream code (`tests/parity`), and checkpoints download from
  their upstream releases, pinned by SHA-256.

### Packaging

- Python 3.10 or newer (3.10 for JetPack 6's torch); torch 2.11.x, torchvision 0.26.x, Triton 3.6.x.
- Extras: `cv` (OpenCV, for SIFT / ORB), `eval` (accuracy benchmarks), `tilelang`. A model that needs a missing
  extra says which one at load time.
- The wheel ships the calibration tables, the CUDA C++ sources and the third-party license texts.

### Performance

- MXFP8 and INT8 quantization with shipped calibration tables, fused Triton / TileLang / CUDA C++ kernels and
  `torch.compile`; accuracy accepted by pose AUC on MegaDepth-1500 / ScanNet-1500 and HardMatch
  ([docs/models.md](docs/models.md), [docs/benchmarks.md](docs/benchmarks.md)).
