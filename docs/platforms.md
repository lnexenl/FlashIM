# Platforms

FlashIM detects the device's vendor, dispatch key, architecture and ISA features, and picks kernels and presets from
them. NVIDIA GPUs are supported today. Other vendors (ROCm, NPU, XPU) are planned: the platform layer already separates
the vendor, the dispatch key and the ISA features.

## NVIDIA architectures

| arch | GPU | status |
|---|---|---|
| sm_120 | RTX 5070 Ti | primary target: tuned, benchmarked, full test suite and accuracy runs |
| sm_100 | datacenter Blackwell | kernels compile-checked only; no tuning, not run |
| sm_90 | H20 | tuned (Triton FP8 GEMM and chain-MLP, TileLang GEMM); not re-tuned on H100 / H800 |
| sm_89 | L20 | tuned (Triton FP8 GEMM, correlation, conv) |
| sm_87 | Jetson AGX Orin | tuned (INT8 path; FP8 presets fall back to INT8) |
| sm_80 / sm_86 | A100, RTX 30xx | not run or tuned; they share the `mma.sync` path validated on sm_87 (correctness, not performance) |

Other architectures (sm_103, sm_110, sm_121) use the default configs.

## Test status by hardware

What was actually run where, per model. The table above says what is tuned; this one says what is verified.

| mark | meaning |
|---|---|
| ✅ | tested on this hardware and passes: the parity test and the unit / op tests, and for quantized presets the task-level AUC gate had no FAIL |
| ❌ | tested on this hardware and fails |
| 🔍 | needs further inspection: only compile-checked, tuned but without a recorded per-model validation, or the test was inconclusive |
| ➖ | not tested |

The only GPU that runs the full test suite and the accuracy runs is the sm_120 one (RTX 5070 Ti). Recorded results on other
GPUs are LoMa's accuracy gate on L20 and H20 ([loma.md](models/loma.md)) and the INT8 / FP8-fallback runs on Jetson AGX Orin;
everything else on them is compile-checked or tuned, not validated.

### Dense and semi-dense models

| model | sm_120 RTX 5070 Ti | sm_100 | sm_90 H20 | sm_89 L20 | sm_87 Orin | sm_80 / sm_86 |
|---|---|---|---|---|---|---|
| `romav2` | ✅ | 🔍 | 🔍 | 🔍 | 🔍 | ➖ |
| `roma`, `roma-indoor`, `minima-roma` | ✅ | 🔍 | 🔍 | 🔍 | ➖ | ➖ |
| `tiny-roma` | ✅ | 🔍 | 🔍 | 🔍 | ➖ | ➖ |
| `mast3r`, `dust3r` | ✅ | 🔍 | 🔍 | 🔍 | ➖ | ➖ |
| `ufm` | ✅ | 🔍 | 🔍 | 🔍 | ➖ | ➖ |
| `loftr` | ✅ | 🔍 | 🔍 | 🔍 | 🔍 | ➖ |
| `eloftr`, `xoftr`, `minima-loftr` | ✅ | 🔍 | 🔍 | 🔍 | ➖ | ➖ |

### Sparse extractors

| model | sm_120 RTX 5070 Ti | sm_100 | sm_90 H20 | sm_89 L20 | sm_87 Orin | sm_80 / sm_86 |
|---|---|---|---|---|---|---|
| `rdd` | ✅ | 🔍 | 🔍 | 🔍 | 🔍 | ➖ |
| `aliked`, `raco`, `raco-aliked` | ✅ | 🔍 | 🔍 | 🔍 | 🔍 | ➖ |
| `superpoint`, `alike`, `disk`, `xfeat`, `dedode`, `dedode-g`, `dad`, `ripe`, `ripepp`, `ripepp-aachen`, `ripepp-scared` | ✅ | 🔍 | 🔍 | 🔍 | 🔍 | ➖ |
| `sift`, `orb` | ✅ | ➖ | ➖ | ➖ | ➖ | ➖ |

### Sparse matchers and pipelines

| model | sm_120 RTX 5070 Ti | sm_100 | sm_90 H20 | sm_89 L20 | sm_87 Orin | sm_80 / sm_86 |
|---|---|---|---|---|---|---|
| `rdd-lightglue`, `aliked-lightglue`, `raco-aliked-lightglue`, `superpoint-lightglue`, `disk-lightglue`, `xfeat-lightglue`, `ripepp-lightglue` | ✅ | 🔍 | 🔍 | 🔍 | 🔍 | ➖ |
| `loma-*` | ✅ | 🔍 | ✅ | ✅ | 🔍 | ➖ |
| `superglue`, `superpoint-superglue` | ✅ | 🔍 | 🔍 | 🔍 | ➖ | ➖ |
| `omniglue` | ✅ | 🔍 | 🔍 | 🔍 | ➖ | ➖ |

### Sparse-to-dense

| model | sm_120 RTX 5070 Ti | sm_100 | sm_90 H20 | sm_89 L20 | sm_87 Orin | sm_80 / sm_86 |
|---|---|---|---|---|---|---|
| `flexglue` | ✅ | 🔍 | 🔍 | 🔍 | 🔍 | ➖ |

- **sm_120**: parity test against the upstream code, unit and op tests, and the accuracy gate on MegaDepth-1500 and
  ScanNet-1500 (1500 pairs each; 0 FAIL, details in the tables below and in [benchmarks.md](benchmarks.md#accuracy)).
- **sm_80 / 89 / 90 / 100 (except LoMa on sm_89 / sm_90, below)**: the Triton / CUDA kernels were compiled for these architectures, not run.
- **LoMa on sm_89 / sm_90**: the INT8 pipeline passes the gate (AUC@5 vs `bf16` -0.30 to -0.10 on L20, -0.09 to +0.23 on H20
  over the five variants, gate 0.5 points on 1500 pairs; `loma-l` needs a per-variant extractor overlay on L20).
- **Jetson AGX Orin (sm_87)**: the INT8 path runs and `fp8` builds `int8`; no per-model accuracy runs are recorded.
- **sm_80 / sm_86**: not run; they share the `mma.sync` path that runs on sm_87.

### Accuracy gate on sm_120

AUC@5 of each preset on MegaDepth-1500 and ScanNet-1500 (1500 pairs each, 3 seeds 17/29/42, poselib 1.0 px, paired
bootstrap 95% CI, `-0.005` gate; `romav2` uses `engine="compile"`, all others `eager`): ✅ every cell passes (95% CI above
-0.005), 🔍 inconclusive on at least one dataset (the CI straddles -0.005; 1500 pairs resolve about +-0.005), ❌ fails. **84
PASS, 42 BORDERLINE (🔍), 0 FAIL** across the 126 `(model, preset, dataset)` cells of the models with a reference `bf16`.
`➡️ int8` / `➡️ bf16` mark presets omitted and resolved via `build.NEAREST` (a pipeline whose extractor loses accuracy under
MXFP8 has no `fp8` preset and builds `int8`); `—` means the model does not define that preset, or that `bf16` is the
reference the other columns are measured against. The numbers are in [benchmarks.md](benchmarks.md#accuracy).

#### Dense and semi-dense models

| model | bf16 | int8 | fp8 |
|---|---|---|---|
| `romav2` | ✅ | 🔍 | 🔍 |
| `roma, roma-indoor` | — | ✅ | ✅ |
| `minima-roma` | — | ✅ | 🔍 |
| `tiny-roma` | — | 🔍 | 🔍 |
| `mast3r` | — | 🔍 | ✅ |
| `dust3r` | — | 🔍 | 🔍 |
| `ufm` | — | 🔍 | 🔍 |
| `loftr` | ✅ | ✅ | 🔍 |
| `eloftr` | — | ✅ | ✅ |
| `xoftr` | — | ✅ | ✅ |
| `minima-loftr` | — | ✅ | 🔍 |

#### Sparse extractors

| model | bf16 | int8 | fp8 |
|---|---|---|---|
| `rdd` | ✅ | — | ✅ |
| `aliked`, `raco-aliked` | ✅ | — | — |
| `superpoint` | 🔍 | 🔍 | 🔍 |
| `xfeat` | ✅ | 🔍 | ✅ |
| `alike` | ✅ | ✅ | 🔍 |
| `disk` | ✅ | ✅ | 🔍 |
| `dedode` | ✅ | 🔍 | ➡️ int8 |
| `dedode-g` | ✅ | 🔍 | 🔍 |
| `dad` | ✅ | ✅ | 🔍 |
| `ripe` | ✅ | 🔍 | 🔍 |
| `ripepp` | — | 🔍 | 🔍 |
| `orb`, `sift` (`fp32`) | ✅ | — | — |

#### Sparse matchers and pipelines

| model | bf16 | int8 | fp8 |
|---|---|---|---|
| `rdd-lightglue` | ✅ | 🔍 | 🔍 |
| `aliked-lightglue` | ✅ | ✅ | ➡️ int8 |
| `raco-aliked-lightglue` | ✅ | ✅ | ✅ |
| `superpoint-lightglue` | ✅ | 🔍 | ➡️ int8 |
| `disk-lightglue` | 🔍 | 🔍 | ➡️ int8 |
| `xfeat-lightglue` | ✅ | ➡️ bf16 | ➡️ bf16 |
| `loma-b128`, `loma-b`, `loma-l`, `loma-g`, `loma-r` | ✅¹ | 🔍 | 🔍 |
| `superpoint-superglue` | — | 🔍 | ➡️ int8 |
| `omniglue` | — | 🔍 | ➡️ int8 |
| `ripepp-lightglue` | — | 🔍 | ➡️ int8 |

¹ `bf16` of every LoMa variant is bit-identical to the unoptimized code: `loma-b128` and `loma-g` on both full datasets, the
other variants on every pair checked (test pair, 3 MegaDepth-1500 and 3 ScanNet-1500 pairs).

#### Sparse-to-dense

| model | bf16 | int8 | fp8 |
|---|---|---|---|
| `flexglue` | ✅ | ✅ | 🔍 |

### Known failing checks on sm_120

- 🔍 `tests/parity/test_lightglue_parity.py::test_reduced_precision_tracks_fp32[fp8]` is an expected failure (MXFP8 noise at its
  0.9 threshold).

On RTX 5070 Ti (`sm_120`, `FLASHIM_IMAGE_DECODER=pil`), the full test suite passes all 96 test files (**1,842 passed, 24
skipped, 1 expected xfail, 0 failed**), including `tests/ops/test_depthwise_cuda.py` (`51/51` passing after sweeping
`sm_120` tile tables). All 44 `(model, preset)` configs of `roma`, `tiny-roma`, `minima-roma`, `mast3r`, `dust3r`, `ufm`, `eloftr`, `xoftr`, `minima-loftr`, `superpoint-superglue` and `omniglue` were also checked against the base tree
via `benchmarks/profile/output_hashes.py`: 31 bit-identical, 8 keypoint-identical with `<= 1.88e-6` FP32 confidence diff
from 2-pass `ops.dual_softmax_mutual`, and 5 with `<= 1` BF16 ULP reduction-order differences from `cublas`
`patch_embed_tokens` / Triton `conv3x3_small_cin_relu`.

### Kernels enabled on sm_120 (and compile-checked architectures)

- **Enabled and verified on `sm_120` (RTX 5070 Ti)**:
  - Previously `sm_89`/`sm_90`-only kernels now measured and enabled on `sm_120`: CUDA C++ `depthwise.cu`
    (`PRODUCER_TILE`, `DWPW_TILE`, `cuda_composite`), `upsample_concat_int8` / `upsample_add` `triton_rows`,
    `conv2d_small_cin_quantize_int8`, cuBLAS `patch_embed_tokens`, exact `layernorm_quantize_int8_rows` /
    `layernorm_gelu_quantize_int8`, 2-pass `dual_softmax_mutual`, INT8 / MXFP8 conv chaining (`QUANT_OUT`), and DINOv2 /
    matcher SageAttention (`sage-fp8`, `sage-fp8-warp`).
  - New CUDA C++ (`ops.simple_nms`, `ops.aliked_score_head`, `ops.raco_score_head`) and Triton (`ops.hamming_distances`,
    `ops.hamming_ratio_mutual_match`, `ops.ratio_mutual_match`, `ops.conv3x3_small_cin_relu`,
    `ops.int8_conv2d_maxpool_only`, `ops.multiscale_score`, `ops.instance_norm_prelu`, `ops.rgb_mean_instance_norm`,
    `ops.conv2d_at_points`, `ops.strict_nms_mask`, `ops.bilinear_sample_points`, `ops.mutual_nn_match`,
    `ops.quantize_int8_rows_heads`, `ops.gelu_quantize_int8_rows`, exact 2x BF16 `upsample_concat` `triton_rows`,
    `ops.aliked_sample_patches`, `ops.aliked_bilinear_sample`, `ops.elu_linear_attention`,
    `ops.flexglue_entmax_readout`, `ops.qkv_norm_rope_interleaved`, `ops.qk_norm_swap_heads`,
    `ops.quantize_mxfp8_heads`, 3D `ops.kde_density`, border-padded `ops.bilinear_sample_points`,
    `input_bias` on `ops.layernorm_cat_quantize_mxfp8`, and tiled low-D `ops.reciprocal_nn` (`triton_tiled`) kernels.
- **Compile-checked only**: the new CUDA C++ kernels, `ops.elu_linear_attention`, and the new FlexGlue Triton kernels
  above (`sm_89+` for FP8 E4M3 kernels) were compile-checked for `sm_80`, `sm_89`, `sm_90a` / `sm_90`, and `sm_100a` /
  `sm_100` without running on hardware. Capability-tuned ops explicitly gate on measured capabilities (`(8, 9)`, `(9, 0)`,
  `(12, 0)`), so unmeasured architectures (`sm_100`, `sm_110`) intentionally keep safe fallbacks.

## FP8 presets

| arch | how an FP8 preset runs |
|---|---|
| sm_120 | natively, as block-scaled MXFP8 through `F.scaled_mm` |
| sm_100 | the same path, compile-checked only |
| sm_89 / sm_90 | natively, through a Triton FP8 GEMM |
| sm_80 / sm_86 / sm_87 | no FP8: the preset's fallback is built instead |

### GPUs without FP8

A GPU without FP8 (Ampere, Jetson Orin) builds the nearest other preset: `fp8` → `int8`. RDD has no INT8 preset,
so its `fp8` becomes `bf16`. A warning says so, and `model.preset` and `model.build_report.fallbacks` record it.

The reason is speed. RoMaV2 on a Jetson AGX Orin, eager, per pair:

| preset | time |
|---|---|
| `fp8`, emulated | 3.3 s |
| `bf16` | 2.0 s |
| `int8` (measured before the fast refiner kernels it has today) | 1.1 s |

### Keeping FP8

The FP8 preset is kept, and runs on emulation kernels (same numerics contract, slower):

- on the CPU;
- when a calibration table is passed with the preset;
- when the config is resolved without a platform and built directly:
  `flashim.build_config(flashim.resolve(name, preset))`.

`optimize.strict=true` raises instead of replacing or emulating.
