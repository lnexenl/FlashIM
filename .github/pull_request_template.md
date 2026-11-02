<!--
Title: what changed, as in the git log ("RaCo models return pixel-center keypoints like every other model").
Fill the sections your change needs and delete the rest (the boxes under "Kind of change" say which).
Every number names its GPU / arch, model, preset, engine and shape. The rules are in AGENTS.md ("Verifying changes").
-->

## What

<!-- One bullet per effect a user or a reviewer would notice. Name the ops, backends, models and presets touched. -->

-

## Why

<!-- The problem, or the measured gain. Link the issue. -->

## Kind of change

- [ ] Docs, benchmarks or scripts only: nothing under `flashim/` behaves differently
- [ ] Refactor, outputs unchanged → **Bit-identical**
- [ ] Kernel, op or backend → **Kernels**; **Numerics** if numbers can change, **Speed** if it claims a gain
- [ ] Quantization, calibration table, preset or default → **Numerics**
- [ ] Intended output change (convention, bug fix) → **Numerics**
- [ ] New model, extractor or pipeline → **Models**, **Numerics**
- [ ] Platform detection or ISA features → **Where it ran**
- [ ] Public interface or config (`flashim.load`, `match`, `Matches`, config keys, presets) → **Interface**
- [ ] Packaging or dependencies

Base commit for every comparison below: `<sha>`

## Bit-identical

<!--
benchmarks/profile/output_hashes.py in a `git worktree` of the base commit vs this tree (commands in its docstring).
Check that the path under test ran: same ops.call_counts(), and no unexpected model.build_report.fallbacks.
Run the base tree twice: configs that differ between two base runs (Triton autotune) are "not reproducible", not passes.
-->

- Identical outputs and `ops.call_counts()`:
  <!-- e.g. romav2 bf16/int8/fp8, rdd-lightglue fp8/int8, aliked-lightglue int8, raco-aliked-lightglue bf16 -->
- Not reproducible on the base tree itself:
- Different (and why that is expected):

## Numerics

<!--
Accepted by task metrics, not tensor allclose: MegaDepth-1500 and ScanNet-1500, seeds 17 / 29 / 42
(benchmarks/accuracy/matrix.sh), AUC@5 drop < 0.005 vs the BF16 compile baseline; HardMatch mAA@10px for hard pairs.
Compare numerics in eager, in one process (compile-full is not reproducible across processes).
For an intended output change, state exactly how outputs differ ("keypoints == old - 0.5 exactly, indices identical").
-->

| model | preset | dataset | protocol | base AUC@5 / 10 / 20 | new AUC@5 / 10 / 20 | Δ AUC@5 |
|---|---|---|---|---|---|---|
| | | mega1500 | <!-- engine, resolution, keypoints, RANSAC --> | | | |

## Speed

<!--
Real call-site shapes. Old and new trees interleaved, best-of-N, two rounds (compiled RoMaV2 varies 15-25% between
processes). To explain a speed-up or a slowdown, cite the profile (torch.profiler CUDA activities, ncu limiter).
-->

GPU (arch), torch, engine:

| what | shape | preset | base ms | new ms | speed-up |
|---|---|---|---|---|---|
| | | | | | |

## Kernels

- Op and impl: <!-- ops/<category>.py declaration; backends/<backend>/<category>.py registration -->
- Registration: <!-- device=, requires= (features), when=, priority=; what op.explain(...) picks at each call site -->
- Correctness: <!-- tests/ops test vs the torch reference; shapes (odd, non-tile-multiple, call-site); tolerance and why -->
- Run on: <!-- arch(s) where the tests and timings ran -->
- Compiled, not validated, on: <!-- archs that were only compile-checked (nvcc -cubin / triton.compile) -->
- [ ] Opaque kernels (CUDA C++, TileLang, vendor libraries) are a `torch.library.custom_op` with a fake
- [ ] JIT builders load on first call (private `_*_kernels.py`), so importing the backend stays cheap
- [ ] Design choices backed by a measurement are recorded in the docstring

## Models

- Upstream and version: <!-- repo + commit / release the port follows -->
- Parity: <!-- tests/parity/<m>/ against the vendored oracle in tests/oracles/<m>/ (license header + upstream LICENSE) -->
- Weights: <!-- Checkpoint(filename, url, sha256) in weights.py, module added to tests/unit/test_weights.py, weight license -->
- Presets and calibration: <!-- presets offered, scripts/calibrate_*.py command, accuracy of each preset in Numerics -->
- Registered in `flashim/registry.py`; row in `docs/models.md`

## Interface

<!--
What user code sees differently, whether it breaks existing code, and the migration. Model-specific settings are config
keys, never new load() keywords; presets stay fp32 / bf16 / int8 / fp8. Say if RunConfig.digest() changes.
-->

## Tests and lint

- `pytest tests`: <!-- N passed, N skipped, N xfailed; or the subset you ran and why -->
- New or changed tests:
- `ruff check flashim tests`: <!-- clean; --fix only on files this PR changes -->

### Where it ran

| device | arch | Python | what ran |
|---|---|---|---|
| | | | |

<!-- e.g. RTX 5070 Ti sm_120 / 3.12 / full suite; CPU only / -m "not cuda"; Jetson AGX Orin sm_87 / 3.10 / unit + parity -->

## Not verified

<!--
What was skipped and why: accuracy not gated yet, archs only compiled, docs not previewed on GitHub, a failing or flaky
test (and whether it fails the same way on the base commit). Write "Nothing" if everything above was run.
-->

## Checklist

- [ ] Layering holds: `ops/` and models import no backend
- [ ] New files and ops are named by what they compute; model-only ops live in `ops/<model>.py` with the model prefix
- [ ] Every new op has a torch reference
- [ ] New model behaviour is a named `Pass` in the model's `spec.py`, not a flag set in a loader
- [ ] Importable on Python 3.10 (no `typing.Self`, `tomllib`, `logging.getLevelNamesMapping`)
- [ ] New dependencies sit in the right extra or group and `uv lock --check` passes; new data files (calibration
      tables, `csrc/`) are in the wheel (`uv build`)
- [ ] Ported upstream code keeps its license header and is listed in `THIRD_PARTY_NOTICES.md`
- [ ] Docs match the change (`docs/`, `README.md`) and user-visible changes have a `CHANGELOG.md` entry
