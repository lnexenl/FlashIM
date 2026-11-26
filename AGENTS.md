# AGENTS.md

Rules for coding agents on FlashIM. This file is short on purpose: workflows live in skills (`.agents/skills/`, linked from
`.claude/skills/`), structure in `docs/architecture.md`, usage and results in `README.md` and `docs/`. A rule that a test can check is a test (`tests/unit/test_architecture.py`), not a sentence here.

## What FlashIM is

An inference framework for image matching (dense, semi-dense, sparse; `docs/models.md` lists them) whose goals are
**custom kernels as first-class citizens**, **multiple platforms** (NVIDIA Ampere to Blackwell today; ROCm / NPU / XPU
planned) and **quantization** (MXFP8, INT8). Models are native PyTorch ports verified against the upstream implementations
("oracles"). Speedups come from quantization, fused kernels and `torch.compile`, never at the cost of unmeasured accuracy.

## Commands

- Python is `.venv/bin/python` (torch 2.11, triton 3.6). **Never run Python from inside `flashim/`**: `flashim/types.py`
  shadows the stdlib `types`. A worktree without a `.venv` runs the main checkout's interpreter with `PYTHONPATH=<worktree>`.
- Tests: `.venv/bin/python -m pytest -q -p no:cacheprovider tests` (several minutes with a GPU; without one add
  `-m "not cuda"`). Markers: `cuda`, `sm120`, `weights`, `oracle`, `slow`. Known xfails (2026-11):
  `tests/parity/test_lightglue_parity.py::test_reduced_precision_tracks_fp32[fp8]` is an expected xfail
  (`tests/ops/test_depthwise_cuda.py` passes all 51 tests on `sm_120`).
- Lint: `.venv/bin/ruff check <files you changed>` (line length 110). **Only `--fix` files you changed**: a directory `--fix`
  once rewrote the user's uncommitted files.
- Use a model: `flashim.load(name, preset)` then `model.match(a, b)`; `flashim.available_models()` lists the names.

## Invariants

1. **Layering** (checked by `tests/unit/test_architecture.py`): `platforms < quant.formats < ops < backends` and
   `layers < quant.passes < models`. `ops/` and models never import a backend: models call `ops.*`. Shared helpers go
   down into `flashim/utils.py` or `ops/`, never up.
2. **Every op has a torch reference** (the oracle and the universal fallback). Optimized implementations register in
   `backends/<backend>/<category>.py`. Kernels opaque to `torch.compile` are wrapped in `torch.library.custom_op`.
3. **Names say what is computed**, not which model uses it (`attention`, `gemm`, `quantize`); an op only one model can use is
   `ops/<model>.py` with a model-prefixed name. Quant passes are named by the structural pattern.
4. **Models are built by the builder** (`build.py`): behaviour is a named `Pass` in the model's `spec.py`, never a flag
   flipped in a loader. Presets are named only `fp32` / `bf16` / `int8` / `fp8` (`ModelSpec` rejects anything else); a model
   without the asked preset builds the nearest (`build.NEAREST`: fp8 -> int8 -> bf16 -> fp32). The resolved config is
   `model.run_config`, what happened is `model.build_report` (`fallbacks` included).
5. **One interface**: `flashim.load(name, preset, *, device, engine, strict, calibration, config, overrides)`; model-specific
   options are config keys. Every image-pair model is an `ImageMatcher` and returns `flashim.Matches`.
6. **Accuracy is measured, not assumed.** A quantized preset ships only if it passes the task-level gate (AUC@5 drop < 0.005
   vs BF16, paired bootstrap; `verify-change`); when it loses accuracy, change the scheme before dropping the preset
   (`quantization-accuracy`). Record *measured* reasons for design choices (`channels_last measured slower here`) in
   docstrings and `quant.py`, not guesses; keep comments at the density of the surrounding code.
7. **Refactors are bit-identical** (`verify-change`: `benchmarks/profile/output_hashes.py`).
8. **Licenses**: record code and weights licenses in each model's `weights.py` and flag non-commercial weights in the README.

## Skills: open the one that matches the task

| task | skill |
|---|---|
| add a model, extractor, pipeline or weights variant | `add-model` |
| add or change an op, a backend implementation, a backend package or a platform | `add-op` |
| make a kernel faster (Triton / TileLang / CUDA C++ / cuBLAS, Ampere to Blackwell) | `optimize-kernel` |
| int8 / fp8 loses accuracy, or choosing how to quantize | `quantization-accuracy` |
| check a change: bit-identity, accuracy gate, speed, profiling | `verify-change` |

`kernelwiki` and `ncu-report-skill` are third-party (hash-locked in `skills-lock.json`): extend arch coverage in
`optimize-kernel`, do not edit them.

## Verifying changes

Before you call a change done: the tests above pass, `ruff` is clean on your files, and the PR says what was verified and what
was not. Refactor -> bit-identical outputs and call counts. Numerics or presets -> the AUC gate. A speed claim -> interleaved
best-of-N old vs new tree, with GPU / arch, model, preset, engine and shape. Only sm_120 can be run locally: other
architectures are compile-checked and reported as such. **Every change that can change a model's latency (a model, op,
backend kernel, quantization pass, preset or default) re-measures the affected rows of the latency tables with
`benchmarks/speed/speed_table.sh` / `speed_table_models.sh` and updates `docs/benchmarks.md` (and the README / model page
numbers that quote it) in the same PR; a new model adds its rows (`tests/unit/test_speed_table_coverage.py` fails
without them); the PR lists the rows it re-measured.** Long GPU jobs (one command over about 10 minutes, or a whole dataset) need the user's OK. Details: `verify-change`.

## Gotchas

- `uv sync` removes everything outside the lockfile (sageattention, tilelang): use `uv sync --extra eval --extra tilelang --group dev`.
- `data/` (or `$FLASHIM_DATA`, `flashim/paths.py`) and `results/` are gitignored; test images are in `assets/`; nothing reads from
  an upstream checkout.
- `compile-full` is not reproducible across processes; compare numerics in eager in one process. Compiled RoMaV2 speed
  varies 15-25% between processes.
- The autotuned Triton conv kernels cache their tile picks: `FLASHIM_AUTOTUNE_CACHE=0` when comparing tiles or trees.
- GPU JPEG decode runs on a private stream (`flashim/image_io.py`): torchvision's nvJPEG decoder raced with GPU work and
  corrupted pixels. `FLASHIM_IMAGE_DECODER=pil` is the reference decoder if a result looks odd.
- `mast3r` / `dust3r` compute in FP16 under `bf16` / `int8` / `fp8` (BF16 cannot resolve the CroCo residual peaks); judge them by
  pose AUC / epipolar error, their 2 px match agreement is below one working pixel.
- `torchvision.ops.deform_conv2d` ignores the current CUDA stream: keep ALIKED on the caller's stream.
- `engine="compile"` fails on quantized models with autotuned INT8 convs (torch 2.11 Dynamo bug): they run eager.
- Keep the code importable on Python 3.10 (JetPack 6); `test_architecture.py` checks it.

## Working with other agents

- One task, one branch cut from `dev` (`model/<name>`, `fix/<name>`), one `git worktree`. Do not `git stash` (shared with
  other worktrees): commit a WIP instead. Do not touch files you did not change; other worktrees can have uncommitted work.
- One GPU shared by several agents: serialize CUDA commands with a lock file (`flock`) and keep each locked command short.
- Shared code (`flashim/ops`, `quant/`, `layers/`, backends) is used by every model: a change there needs the bit-identity
  check of the models it affects, not only of yours.

## Git

- **`main` only receives merges from `dev`.** Work branches are cut from `dev` and their PRs target `dev`; never open a PR from a
  work branch straight to `main`, and never push to `main` or `dev` directly.
- Commit and push only when asked. Commit messages say what changed and why, plus how it was verified (bit-identical
  configs, test counts, timings).
- Pull requests: fill `.github/pull_request_template.md`, keep the sections the change needs, and say what was not verified.
