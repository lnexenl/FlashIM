# AGENTS.md

Guidance for coding agents working on FlashIM. Read this before changing code; `README.md` covers usage and
results.

## What FlashIM is

An inference framework for image matching (dense: RoMaV2; sparse: RDD, ALIKED, RaCo, LightGlue) whose goals are
**custom kernels as first-class citizens**, **multiple platforms** (NVIDIA archs today; ROCm / NPU / XPU planned)
and **quantization** (MXFP8, INT8). Models are native PyTorch ports verified against the upstream implementations
("oracles"); speedups come from quantization, fused kernels and `torch.compile`, never at the cost of unmeasured
accuracy.

## Layout and layering

```
flashim/
  platforms/   device detection: Platform(vendor, dispatch key, arch, features). "cuda" = NVIDIA only,
               ROCm is "rocm". ISA features (TMA, WGMMA, TCGEN05, MMA_SYNC, FP8_E4M3, MXFP8_SCALED_MM)
               and toolchains (TRITON, TILELANG, NVCC). register_detector() / entry point flashim.platforms
  log.py       stdlib logging under "flashim"; FLASHIM_LOG=info,ops=debug; warn_once; profiler/NVTX ranges
  ops/         op DECLARATIONS only: Op(name, torch reference[, weights=n, composite=]) + docstring.
               _registry.py: dispatch, overrides (FLASHIM_OPS), strict mode, explain(), bind()/BoundOp
  backends/    optimized impls, one package per kernel language/library, files mirror ops/:
               triton/ tilelang/ cuda/ (JIT CUDA C++, csrc/) torch_native/ (cuBLAS) external/ (optional pkgs)
               _loader.py lists them; out-of-tree backends via entry point flashim.backends
  layers/      building blocks the passes recognize by TYPE (TransformerBlock, SeparableConvBlock,
               DeformableEncoder, ...)
  quant/       scheme.py (compressed-tensors vocabulary), formats/, tensor.py, calibration.py,
               methods/ (StaticMXFP8Linear = ops.mxfp8_linear bound to its weight, StaticInt8Linear),
               passes/ (structural fusions matched by layer type), apply.py (quantize_model)
  config.py    RunConfig: arch / precision / optimize / engine / inference / components; layered and hashable
  build.py     ModelSpec + the generic builder every model goes through; flashim.load (the one loader), PRECISIONS
  models/base.py
               ImageMatcher / FeatureMatcher: the one match(a, b, *, resize, max_matches, threshold) -> Matches
  models/<m>/  model.py (pure nn.Module, calls ops only), config.py, weights.py, quant.py (presets and their
               accuracy evidence), spec.py (ModelSpec)
  models/sparse.py, sparse_spec.py, weights_util.py
               shared by the sparse extractors (superpoint, alike, disk, xfeat, dedode(-g), dad, ripe, loftr,
               sift, orb): `SparseExtractor` base (resize, coordinate mapping, MNN matching, BN folding, compute
               dtype), `extractor_spec()` (the fp32 / bf16 / int8 / fp8 presets from a list of quantizable globs),
               checkpoint lookup (`checkpoint` config key / env var / torch hub folder, else download of the release
               pinned by URL + SHA-256 in each model's `weights.py`)
  engines/     region torch.compile, CUDA-graph runner
benchmarks/    speed/, accuracy/ (MegaDepth-1500, ScanNet-1500, HardMatch), profile/
scripts/       calibration
tests/         unit/, ops/, quant/, parity/ (vs upstream oracles)
```

Dependencies point one way: `platforms < quant.formats < ops < backends`; `layers < quant.passes < models`.
`ops/` never imports a backend; models never import a backend (they call `ops.*`).

## Rules

- **Naming** (user rule): `ops/`, `backends/*/` and `quant/` files are named by *what they compute*
  (`attention`, `conv`, `correlation`, `density`, `gemm`, `quantize`), never by the model part using them. An op
  only one model can use goes in `ops/<model>.py` with a model-prefixed name (`romav2_refiner_input`,
  `raco_resblock`). Quant passes are named by the structural pattern (`transformer_block`, `separable_conv_block`).
- **Every op has a torch reference** (correctness oracle and universal fallback). Impls register in
  `backends/<backend>/<category>.py` next to their kernels with `device=`, `requires=` (features, not capability
  ranges when the feature is not monotonic in the capability), `when=` (per-call eligibility), optionally
  `prepare=` / `accepts=` for ops with bound weights (`Op(weights=n)`; use `op.bind(...).warmup()` at load time).
- **Kernels opaque to `torch.compile`** (CUDA C++ via ctypes, TileLang, vendor libs) are wrapped in
  `torch.library.custom_op` with a fake. Triton stays visible to Inductor. Heavy JIT frameworks keep builders in
  a private `_*_kernels.py` imported on first call (backend loading must stay cheap).
- **Models are built by the builder.** Add behaviour as a named `Pass` in the model's `spec.py` (stage
  `pre_quant` / `post_quant`, `requires=` a platform predicate, `exact=`), never as a flag flipped by hand in a
  loader. Presets are config overlays (`extends` allowed) named only `fp32` / `bf16` / `int8` / `fp8`
  (`build.PRECISIONS`; `ModelSpec` rejects anything else): no `int8-full`-style variants. Configs kept for parity
  or experiments are internal overlays (RoMaV2's `ORACLE_OVERLAYS`), not presets. A model without the asked
  preset, and `fp8` on a GPU without FP8, build the nearest one (`build.NEAREST`: fp8 -> int8 -> bf16 -> fp32);
  a pipeline's preset is every component's, each degrading on its own. `flashim.load` applies the FP8 fallback,
  `flashim.resolve` without `platform=` does not. The resolved config is `model.run_config` (`model.config` is
  the architecture config in several models); what happened is `model.build_report` (`fallbacks` included).
- **One interface.** Load with `flashim.load(name, preset, *, device, engine, strict, calibration, config,
  overrides)`; anything model-specific is a config key, never a new loader keyword or a per-model `load_*`.
  Every image-pair model is an `ImageMatcher` (implement `_match`; detector/descriptor models get it from
  `FeatureMatcher` via `extract` + `match_features`) and returns `flashim.Matches`; per-call defaults come from the
  config's `inference` section (`match_defaults`).
- **New models**: build from `flashim.layers` where the pattern fits so quantization/fusion apply for free; keep
  upstream parameter names so checkpoints load with `strict=True`; add a parity test against the upstream code.
- Keep comments/docstrings at the density of the surrounding code; record *measured* reasons for design choices
  (e.g. "channels_last measured slower here") in docstrings, not guesses.

## Recipes

- **Optimize / write a kernel** (any NVIDIA arch, Ampere to Blackwell): use the `optimize-kernel` skill
  (`.agents/skills/optimize-kernel`, linked from `.claude/skills/`). It is the KDA loop (research, implement,
  verify, profile, iterate, promote) on top of the recipes below, with `kernelwiki` and `ncu-report-skill` as
  helpers and `references/nvidia-archs.md` for per-arch features. Only sm_120 can be run locally; other archs are
  compile-checked and must be reported as such. `kernelwiki` / `ncu-report-skill` are third-party (hash-locked in
  `skills-lock.json`): extend arch coverage in `optimize-kernel`, don't edit them.

- **Add an op**: declaration + reference in `ops/<category>.py`, export in `ops/__init__.py`; impl + registration in
  `backends/<backend>/<category>.py`; test impl vs reference (`tests/ops/`), check `op.explain(*args)`.
- **Add a backend**: new package under `backends/`, add a `Backend(...)` entry in `backends/_loader.py` (Python
  packages it needs), impls register for the platform's dispatch key and features.
- **Add a platform**: detector in `platforms/` (or `register_detector`) setting a distinct dispatch key and only
  features that some kernel has been validated with on that vendor.
- **Add a sparse extractor**: subclass `SparseExtractor` (implement `detect`, `batchnorm_pairs`, keep upstream
  parameter names), build the spec with `extractor_spec(quantized=..., protected=...)`, register it in
  `flashim/registry.py`, calibrate (`scripts/calibrate_extractor.py --model m --format fp8 --format int8`; LightGlue pairings:
  `scripts/calibrate_lightglue.py --extractor m`), add `tests/parity/<m>/test_parity.py`. Convolutions are quantized as
  im2col + GEMM (`quant/methods/conv2d.py`; groups == 1 only): measured *slower* than BF16 on the small nets here.
- **Add a model**: `models/<m>/` with `model.py`, `weights.py`, `quant.py`, `spec.py` (register it in
  `flashim/registry.py` `SPECS`), the model class an `ImageMatcher` / `FeatureMatcher`, CPU unit tests and a
  parity test against the oracle. `weights.py` pins each release as `Checkpoint(filename, url, sha256)` and loads it
  through `find_weights` (auto-download; add the module to `tests/unit/test_weights.py`).

## Verifying changes

- Tests: `.venv/bin/python -m pytest -q -p no:cacheprovider tests` (~3.5 min on the RTX 5070 Ti; without a GPU:
  `-m "not cuda"`). `tests/parity/test_lightglue_parity.py::test_reduced_precision_tracks_fp32[fp8]` is an xfail
  (MXFP8 noise at its 0.9 threshold). The vendored LightGlue oracles turn off flash SDPA and set
  `cudnn.deterministic` for the whole process; `tests/conftest.py` restores both after every test, so results do not
  depend on test order. Markers: `cuda`, `sm120`, `weights`, `oracle`, `slow`.
- Lint: `.venv/bin/ruff check flashim tests` (line length 110). **Only `--fix` files you changed**
  (`ruff check --fix` on a directory rewrote the user's uncommitted files once); preview with `--diff`.
- **Refactors must be bit-identical**: run the same inference in a `git worktree` of the previous commit and in the
  working tree (`PYTHONPATH=<tree>`), compare output hashes and `ops.call_counts()` for several configs (RoMaV2
  fp8 / int8 / bf16, RDD+LightGlue fp8 / int8, ALIKED+LightGlue, RaCo+ALIKED+LightGlue bf16). Make sure the
  path under test actually runs (check call counts and `model.build_report.fallbacks`: a component without the
  pipeline's preset builds another one).
- **Accuracy changes** are accepted by task-level AUC, not tensor allclose: MegaDepth-1500 / ScanNet-1500, 3 seeds
  (17, 29, 42), AUC@5 drop < 0.005 vs the BF16 compile baseline (`benchmarks/accuracy/matrix.sh`); HardMatch
  mAA@10px for hard pairs. Compare numerics in eager in one process: `compile-full` is not reproducible across
  processes (max-autotune), and two compiled instances in one process share Dynamo graphs.
- **Speed**: compiled RoMaV2 varies 15-25% between processes. Compare best-of-N *interleaved* runs of old and new
  trees (keep N small, ~2 rounds; long GPU runs need the user's OK). Profile before explaining a slowdown
  (`torch.profiler` CUDA activities, `ncu` on the suspect kernel: read the limiter, e.g. L1/TEX% for gathers).

## Environment gotchas

- Use `.venv/bin/python` (torch 2.11.0+cu130, triton 3.6). Never run Python from inside `flashim/`
  (`flashim/types.py` shadows the stdlib `types`).
- `uv sync` removes anything not in the lockfile (sageattention, tilelang, hand-built extras):
  use `uv sync --extra eval --extra tilelang --group dev`; sageattention must be rebuilt by the user (GCC 16 breaks torch headers:
  `CC=clang CXX=clang++ CUDAHOSTCXX=clang++ NVCC_APPEND_FLAGS=-allow-unsupported-compiler`, needs libomp).
- Jetson (AGX Orin, JetPack 6.2, sm_87, aarch64): no `uv sync` there. NVIDIA's L4T torch exists for Python 3.10
  only: `requirements-jetson.txt` (its header has the commands, including the `.pth` that stands in for the
  editable install), so keep the code importable on 3.10 (`typing.Self`, `tomllib`,
  `logging.getLevelNamesMapping` are 3.11+). Triton comes from PyPI (the L4T wheel has no `ptxas`). CUDA needs the
  user in the `video` *and* `render` groups (`/dev/nvmap`, `/dev/dri/renderD*`; without `render`: "Error 801").
  No FP8 below sm_89: `fp8` is replaced by the nearest other preset (`build.NEAREST`). CPU results differ from x86 in the
  last ulp (OpenBLAS / FMA). Test suite there: 34 min cold (INT8 conv autotuning is 3-4 min per shape on its
  CPU), 2 min warm, the same failures as on x86. No kernel profiler there (CUPTI needs privileges): time ops with
  CUDA events (`results/orin/op_breakdown.py`).
- `/tmp` is a 16 GB tmpfs: set `TORCHINDUCTOR_CACHE_DIR=~/.cache/flashim/inductor` for compile benchmarks with many
  shapes. CUDA C++ kernels JIT-build into `~/.cache/flashim/cuda` (`nvcc` from `/opt/cuda`, `-allow-unsupported-compiler`,
  no torch headers).
- Long jobs: `setsid nohup benchmarks/accuracy/matrix.sh ... &` (plain background shells die with the session);
  scripts are resumable. Don't edit a running shell script in place (write a new file and `mv`).
  To find benchmark processes use `pgrep -f '^\.venv/bin/python benchmarks/accuracy'` (a bare pattern matches the
  calling shell).
- zsh: unquoted `$cfg` does not word-split (use `${=cfg}`).
- `git clone` of GitHub repos fails in the sandbox proxy; download the archive zip instead.
- The upstream reference code of the parity tests is vendored under `tests/oracles/<model>/` (only the files the tests
  import, each with a license header and the upstream LICENSE next to it; `tests/oracles/README.md`). Checkpoints come
  from FlashIM's own weight lookup. DISK and LoFTR are checked against `kornia` (dev dep). The oracles' own imports
  (`timm`, `yacs`, `pyyaml` for RDD; `kornia`, `cv2`) are in the dev group.
  Parity-test pitfalls: put upstream modules in `.eval()`; disable `cudnn.allow_tf32` (BN folding changes TF32
  rounding); `torch.cdist` on pixel coordinates needs `compute_mode="donot_use_mm_for_euclid_dist"`; compare
  keypoints as sets, not by index (a few threshold-borderline points flip and shift every later index).
- Nothing reads from a RoMaV2 (or other upstream) checkout: the test / benchmark images are in `assets/` (copied from
  RoMaV2, MegaDepth scene 0015 and LightGlue-ONNX; `assets/LICENSE-RoMaV2`), the datasets and calibration manifests under `data/` (or `$FLASHIM_DATA`; `flashim/paths.py`; locally symlinks to
  `~/work/RoMaV2/data/{megadepth,scannet,quantization}`). `data/` and `results/` are gitignored, and so is the
  third-party `kernelwiki` skill.
- The autotuned conv kernels (`backends/triton/conv.py`) keep their timings in Triton's cache, so a pick made on a
  busy GPU sticks until the kernel source changes: `FLASHIM_AUTOTUNE_CACHE=0` re-times in every process (do that
  when comparing tiles). Editing that file shifts line numbers, which recompiles every kernel in it.
- TileLang 0.1.14: no `from __future__ import annotations` in TileLang files; width <= 16 miscompiles;
  `T.no_set_max_nreg()`; an explicit `fence.proxy.async` is needed before TMA refills a stage read by plain loads.
- `torchvision.ops.deform_conv2d` ignores the current CUDA stream: keep ALIKED on the caller's stream.

## Git

- Work on a branch cut from `main`; commit only when asked.
- Commit messages: what changed and why, plus how it was verified (bit-identical configs, test counts, timings).
