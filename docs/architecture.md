# Architecture

How the package is laid out and how an op call reaches a kernel. The contributor rules (naming, how to add an op, a
backend, a model or a platform, and how changes are verified) are in [AGENTS.md](../AGENTS.md).

## Layout

| path | what it holds |
|---|---|
| `flashim/platforms/` | Device detection: vendor, dispatch key, arch, ISA features (TMA, WGMMA, TCGEN05, FP8, MXFP8 `scaled_mm`, ...) and toolchains (Triton, TileLang, nvcc). Pluggable per device type |
| `flashim/ops/` | Op declarations: a signature and a torch reference. Files are named by computation (`attention`, `conv`, `correlation`, `density`, `gemm`, `quantize`). An op only one model can use lives in a file named after that model (`romav2.py`, `aliked.py`) |
| `flashim/backends/` | Optimized implementations, one package per kernel language or library, with the same file names as `ops/`: `triton/`, `tilelang/`, `cuda/` (JIT CUDA C++), `torch_native/` (cuBLAS), `external/` (sageattention). Out-of-tree backends register through the `flashim.backends` entry point |
| `flashim/layers/` | Building blocks the passes recognize by type (`TransformerBlock`, `SeparableConvBlock`, `DeformableEncoder`, ...). A model built from them gets quantization and fused kernels without model-specific code |
| `flashim/quant/` | Formats (MX), `QuantizedTensor`, schemes (compressed-tensors vocabulary), calibration, methods (`StaticMXFP8Linear`, `StaticInt8Linear`) and fusion passes |
| `flashim/config.py` | `RunConfig`: the typed, layered, serializable run configuration ([configuration.md](configuration.md)) |
| `flashim/build.py` | `ModelSpec` (presets, passes, engines), the generic builder every model goes through and `flashim.load`, the one loader |
| `flashim/engines/` | Region `torch.compile` and an experimental CUDA-graph runner |
| `flashim/models/` | One package per model, typically `model.py` (a pure `nn.Module` that only calls ops), `config.py`, `weights.py`, `quant.py`, `spec.py`. `base.py` holds `ImageMatcher`, the `match(a, b) -> Matches` interface every image-pair model implements; `sparse.py` and `sparse_spec.py` are the base class and spec factory shared by the sparse extractors |
| `flashim/geometry/` | Pose estimation and AUC metrics |
| `flashim/log.py` | Logging (`FLASHIM_LOG`), dispatch tracing, profiler and NVTX ranges |
| `benchmarks/` | `speed/`, `accuracy/` (MegaDepth-1500, ScanNet-1500, HardMatch), `profile/` |
| `scripts/` | Calibration: `calibrate_romav2.py`, `calibrate_rdd.py`, `calibrate_extractor.py`, `calibrate_lightglue.py` |
| `tests/` | `unit/`, `ops/` (implementation vs reference), `quant/`, `parity/` (vs the upstream code vendored in `tests/oracles/`) |

Dependencies point one way: `platforms < quant.formats < ops < backends`, and `layers < quant.passes < models`.
`ops/` never imports a backend, and neither do models: they call `ops.*`.

## Op dispatch

Every op has a torch reference. It is the correctness oracle and the fallback that runs everywhere.

An optimized implementation registers for:

- a **device**: a platform dispatch key (`"cuda"` is NVIDIA only; ROCm is `"rocm"`), optionally with an NVIDIA
  compute-capability range;
- the platform **features** it needs (for example TMA or FP8);
- optionally a per-call **predicate** on the arguments.

Among the implementations that match a call, dispatch picks the highest priority, then the narrowest capability range,
then the first registered. An implementation with a negative priority is opt-in: only a pin or an override selects it.

Ops with constant weights let each implementation repack them once, at load time. `StaticMXFP8Linear`, for example, is
`ops.mxfp8_linear` bound to its weight; normal dispatch then picks cuBLAS MXFP8, a Triton FP8 GEMM or the emulated
reference, in that order.

## Inspecting and overriding

| tool | effect |
|---|---|
| `op.explain(*args)` | says why each implementation was or was not chosen for these arguments |
| `ops.call_counts()` | how many times each implementation of each op ran |
| `FLASHIM_LOG=info,ops=debug` | logs the first use of every implementation, and reference fallbacks on accelerators |
| `FLASHIM_OPS="rope_qk:reference,-local_corr"` | comma-separated overrides: `name:impl` forces an implementation, `-name` disables the optimized ones |
| `FLASHIM_STRICT_OPS=1` | raises when a GPU call falls back to a torch reference |
| `FLASHIM_NVTX=1` | wraps op calls in NVTX ranges for `nsys` / `ncu --nvtx` (eager only) |
