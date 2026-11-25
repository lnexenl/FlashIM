# Architecture

How the package is laid out and how an op call reaches a kernel. The contributor rules (naming, how to add an op, a
backend, a model or a platform, and how changes are verified) are in [AGENTS.md](../AGENTS.md).

## Layout

| path | what it holds |
|---|---|
| `flashim/platforms/` | Device detection: vendor, dispatch key, arch, ISA features (TMA, WGMMA, TCGEN05, FP8, MXFP8 `scaled_mm`, ...) and toolchains (Triton, TileLang, nvcc). Pluggable per device type |
| `flashim/ops/` | Op declarations: a signature and a torch reference. Files are named by computation (`activation`, `assignment`, `attention`, `conv`, `correlation`, `density`, `gemm`, `image`, `matching`, `nms`, `norm`, `patchify`, `quantize`, `resample`). An op only one model can use lives in a file named after that model (`aliked.py`, `dust3r.py`, `eloftr.py`, `lightglue.py`, `raco.py`, `roma.py`, `romav2.py`, `tiny_roma.py`) |
| `flashim/backends/` | Optimized implementations, one package per kernel language or library, mirroring `ops/`: `triton/` (including `point_conv.py`, `stem_conv.py`), `tilelang/`, `cuda/` (JIT CUDA C++ in `csrc/`: `depthwise.cu`, `raco_resblock.cu`, `raco_score_head.cu`, `aliked_score_head.cu`, `simple_nms.cu`, `sinkhorn.cu`, `reciprocal_nn.cu`, `raco_resblock_helpers.cuh`), `torch_native/` (cuBLAS `gemm.py`, `patchify.py`), `external/` (sageattention). Out-of-tree backends register through the `flashim.backends` entry point |
| `flashim/layers/` | Building blocks the passes recognize by type (`TransformerBlock`, `SeparableConvBlock`, `DeformableEncoder`, ...). A model built from them gets quantization and fused kernels without model-specific code |
| `flashim/quant/` | Formats (MX), `QuantizedTensor`, schemes (compressed-tensors vocabulary), calibration, methods (`StaticMXFP8Linear`, `StaticInt8Linear`, per-token `DynamicInt8Linear`), orthogonal / Hadamard rotations folded into weights (`rotation.py`) and fusion passes |
| `flashim/config.py` | `RunConfig`: the typed, layered, serializable run configuration ([configuration.md](configuration.md)) |
| `flashim/build.py` | `ModelSpec` (presets, passes, engines, `quant_configs` with per-variant `<variant>/<quant>` lookup mirroring `calibrations`), the generic builder every model goes through and `flashim.load`, the one loader |
| `flashim/engines/` | Region `torch.compile` and the CUDA-graph runner (`CudaGraphRunner`, guarded by repeat-shape hits, keypoint caps, pool release on fallback, and `sdpa_graph_safe` probing) |
| `flashim/models/` | One package per model, typically `model.py` (a pure `nn.Module` that only calls ops), `config.py`, `weights.py`, `quant.py`, `spec.py`. `base.py` holds `ImageMatcher`, the `match(a, b) -> Matches` interface every image-pair model implements; `sparse.py` and `sparse_spec.py` (`extractor_spec`, with `int8_ignore`, `fp8_ignore`, `int8_dynamic`, `cuda_graph`) are the base class and spec factory shared by the sparse extractors |
| `flashim/geometry/` | Pose estimation and AUC metrics |
| `flashim/image_io.py` | GPU (nvJPEG) decoding of JPEG paths for `load_image`, with libjpeg's chroma upsampling (`ops.jpeg_fancy_upsample`). The decode runs on a private stream ordered against the caller's (torchvision's decoder allocates on the caller's stream but decodes on its own, which corrupted pixels next to GPU work) |
| `flashim/log.py` | Logging (`FLASHIM_LOG`), dispatch tracing, profiler and NVTX ranges |
| `benchmarks/` | `speed/`, `accuracy/` (MegaDepth-1500, ScanNet-1500, HardMatch), `profile/` |
| `scripts/` | Calibration: `calibrate_romav2.py`, `calibrate_rdd.py`, `calibrate_extractor.py`, `calibrate_lightglue.py`, and one per newer model (`calibrate_roma.py`, `calibrate_mast3r.py`, `calibrate_dust3r.py`, `calibrate_ufm.py`, `calibrate_superglue.py`); `convert_omniglue_weights.py` converts OmniGlue's TF SavedModels |
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
reference, in that order. `flashim/ops/_registry.py` caches the ordered candidate list per platform (and `BoundOp._usable`
per bound op) to keep eager dispatch overhead low.

Capability-tuned kernels (for example `depthwise.cu` tile tables, `triton_rows` upsampling, `conv2d_small_cin_quantize_int8`,
cuBLAS `patch_embed_tokens`, exact `layernorm_quantize_int8_rows`, 2-pass `dual_softmax_mutual`, and INT8/MXFP8 conv
chaining) register only for explicitly measured compute capabilities (`(8, 9)`, `(9, 0)`, `(12, 0)`), so unmeasured
architectures (`sm_100`, `sm_110`) fall back safely rather than running untuned tiles ([platforms.md](platforms.md)).

## Inspecting and overriding

| tool | effect |
|---|---|
| `op.explain(*args)` | says why each implementation was or was not chosen for these arguments |
| `ops.call_counts()` | how many times each implementation of each op ran |
| `FLASHIM_LOG=info,ops=debug` | logs the first use of every implementation, and reference fallbacks on accelerators |
| `FLASHIM_OPS="rope_qk:reference,-local_corr"` | comma-separated overrides: `name:impl` forces an implementation, `-name` disables the optimized ones |
| `FLASHIM_STRICT_OPS=1` | raises when a GPU call falls back to a torch reference |
| `FLASHIM_NVTX=1` | wraps op calls in NVTX ranges for `nsys` / `ncu --nvtx` (eager only) |
| `FLASHIM_IMAGE_DECODER=pil` | decoder of image *paths* (`load_image`): `nvjpeg` (default; JPEGs on CUDA decoded on the GPU, chroma upsampled libjpeg's way), `pil`, `nvjpeg-raw` (nvJPEG's own chroma) |
