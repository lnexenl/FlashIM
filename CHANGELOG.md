# Changelog

## Unreleased

### Models

- FlexGlue (`flexglue`, cvg/FlexGlue v0.1): sparse-to-dense matcher with a DINOv3-ConvNeXt descriptor.
  `match_keypoints` matches any extractor's keypoints, `match` samples the dense warp with upstream's gates. Presets
  `fp32` / `bf16` / `int8` / `fp8` (default: MXFP8 transformer linears + INT8 descriptor MLP); opt-in early stopping
  and `arch.num_layers` truncation. MegaDepth-1500 AUC@5 (dense, 5000 matches, 3 seeds): bf16 0.6397, fp8 0.6369,
  int8 0.6370; compiled `match_keypoints` at 2048 keypoints on an RTX 5070 Ti: bf16 32.5 ms, fp8 27.5 ms, fp8 +
  `optimize.attention=sage-fp8-fast` 19.4 ms.

### Interface

- `optimize.attention="sage-fp8-fast"`: `sage-fp8` with FP16 PV accumulation (SageAttention2++), sm_89+, opt-in.
- Quant configs may mix MXFP8 and INT8 schemes: `quantize_model` takes one table of each kind, a spec's default
  calibration may be a tuple of files, and a table passed with `calibration=` replaces only the default of its kind.

### Performance

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
  40.1 -> 38.4 ms (H20). The layer-norm kernel is registered for sm_89 / sm_90 only (the architectures it was checked
  on); other GPUs run the unfused modules.
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
