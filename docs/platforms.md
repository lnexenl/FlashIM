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

## FP8 presets

| arch | how an FP8 preset runs |
|---|---|
| sm_120 | natively, as block-scaled MXFP8 through `F.scaled_mm` |
| sm_100 | the same path, compile-checked only |
| sm_89 / sm_90 | natively, through a Triton FP8 GEMM |
| sm_80 / sm_86 / sm_87 | no FP8: the preset's fallback is built instead |

### GPUs without FP8

A GPU without FP8 (Ampere, Jetson Orin) builds the preset's INT8 counterpart: `fp8` → `int8`. RDD has no INT8 preset,
so its FP8 presets become `bf16`. A warning says so, and `model.preset` and `model.build_report.fallbacks` record it.

The reason is speed. RoMaV2 on a Jetson AGX Orin, eager, per pair:

| preset | time |
|---|---|
| `fp8`, emulated | 3.3 s |
| `bf16` | 2.0 s |
| `int8-full` (an earlier preset; today's `int8` adds the fast refiner kernels) | 1.1 s |

### Keeping FP8

The FP8 preset is kept, and runs on emulation kernels (same numerics contract, slower):

- on the CPU;
- when a calibration table is passed with the preset;
- when the config is resolved without a platform and built directly:
  `flashim.build_config(flashim.resolve(name, preset))`.

`optimize.strict=true` raises instead of replacing or emulating.
