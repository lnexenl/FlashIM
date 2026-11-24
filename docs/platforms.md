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

| model | sm_120 RTX 5070 Ti | sm_100 | sm_90 H20 | sm_89 L20 | sm_87 Orin | sm_80 / sm_86 |
|---|---|---|---|---|---|---|
| **added 2026-11** | | | | | | |
| `roma`, `roma-indoor`, `minima-roma` | ✅ | 🔍 | 🔍 | 🔍 | ➖ | ➖ |
| `tiny-roma` | ✅ | 🔍 | 🔍 | 🔍 | ➖ | ➖ |
| `mast3r`, `dust3r` | ✅ | 🔍 | 🔍 | 🔍 | ➖ | ➖ |
| `ufm` | ✅ | 🔍 | 🔍 | 🔍 | ➖ | ➖ |
| `eloftr`, `xoftr`, `minima-loftr` | ✅ | 🔍 | 🔍 | 🔍 | ➖ | ➖ |
| `superglue`, `superpoint-superglue` | ✅ | 🔍 | 🔍 | 🔍 | ➖ | ➖ |
| `omniglue` | ✅ | 🔍 | 🔍 | 🔍 | ➖ | ➖ |
| **before 2026-11** | | | | | | |
| `romav2` | ✅ | 🔍 | 🔍 | 🔍 | 🔍 | ➖ |
| `rdd`, `rdd-lightglue` | ✅ | 🔍 | 🔍 | 🔍 | 🔍 | ➖ |
| `aliked`, `raco`, `raco-aliked`, `aliked-lightglue`, `raco-aliked-lightglue` | ✅ | 🔍 | 🔍 | 🔍 | 🔍 | ➖ |
| `superpoint`, `alike`, `disk`, `xfeat`, `dedode`, `dedode-g`, `dad`, `ripe`, `superpoint-lightglue`, `disk-lightglue`, `xfeat-lightglue` | ✅ | 🔍 | 🔍 | 🔍 | 🔍 | ➖ |
| `loftr` | ✅ | 🔍 | 🔍 | 🔍 | 🔍 | ➖ |
| `loma-*` | ✅ | 🔍 | ✅ | ✅ | 🔍 | ➖ |
| `sift`, `orb` | ✅ | ➖ | ➖ | ➖ | ➖ | ➖ |

- **Models added 2026-11 on sm_120**: parity test against the upstream code, unit and op tests, and the accuracy gate on
  MegaDepth-1500 and ScanNet-1500 (1500 pairs each; 0 FAIL, details in the next table and in [benchmarks.md](benchmarks.md#accuracy-of-the-2026-11-additions)).
- **Models added 2026-11 on sm_80 / 89 / 90 / 100**: their new Triton / CUDA kernels were compiled for these architectures, not run.
- **LoMa on sm_89 / sm_90**: the INT8 pipeline passes the gate (AUC@5 vs `bf16` -0.30 to -0.10 on L20, -0.09 to +0.23 on H20
  over the five variants, gate 0.5 points on 1500 pairs; `loma-l` needs a per-variant extractor overlay on L20).
- **Jetson AGX Orin (sm_87)**: the INT8 path runs and `fp8` builds `int8`; no per-model accuracy runs are recorded.
- **sm_80 / sm_86**: not run; they share the `mma.sync` path that runs on sm_87.

### Accuracy gate of the 2026-11 models on sm_120

AUC@5 of the quantized preset against the same model's `bf16`, on both datasets: ✅ every cell passes (95% CI above -0.005),
🔍 inconclusive on at least one dataset (the CI straddles -0.005; 1500 pairs resolve about +-0.005), ❌ fails. No cell fails.
`➡️ int8`: the pipeline has no `fp8` preset and builds `int8`, because MXFP8 lost accuracy in its extractor.

| model | int8 | fp8 |
|---|---|---|
| `roma, roma-indoor` | ✅ | ✅ |
| `tiny-roma` | 🔍 | 🔍 |
| `minima-roma` | ✅ | 🔍 |
| `mast3r` | 🔍 | ✅ |
| `dust3r` | 🔍 | 🔍 |
| `ufm` | 🔍 | 🔍 |
| `eloftr` | ✅ | ✅ |
| `xoftr` | ✅ | ✅ |
| `minima-loftr` | ✅ | 🔍 |
| `superpoint-superglue` | 🔍 | ➡️ int8 |
| `omniglue` | 🔍 | ➡️ int8 |

### Known failing checks on sm_120

- ❌ `tests/ops/test_depthwise_cuda.py`: 21 failures (bit-identity against the Triton implementation), present before the
  2026-11 models.
- 🔍 `tests/parity/test_lightglue_parity.py::test_reduced_precision_tracks_fp32[fp8]` is an expected failure (MXFP8 noise at its
  0.9 threshold).
- ❌ presets that fail the AUC gate on MegaDepth-1500 ([benchmarks.md](benchmarks.md#accuracy), 2026-10-05; they still build):
  `xfeat` `int8` / `fp8`, `dedode` `fp8`, `superpoint-lightglue` `int8` / `fp8`, `xfeat-lightglue` `int8` / `fp8`.

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
