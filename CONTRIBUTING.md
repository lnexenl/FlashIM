# Contributing to FlashIM

Thanks for helping improve FlashIM. It is an inference framework for image matching models, with native PyTorch ports,
custom kernels, and quantized presets. Please read the [Code of Conduct](CODE_OF_CONDUCT.md) before participating.

## Development setup

Use Python 3.10 or newer. From the repository root, install the development environment with the extras used by the
project:

```bash
uv sync --extra eval --extra tilelang --group dev
```

The project interpreter is `.venv/bin/python`; use it directly when running commands. Do not run Python from inside
`flashim/`, because `flashim/types.py` shadows the standard-library `types` module.

## Tests and lint

The CPU suite is what CI runs across Python 3.10, 3.12, and 3.13 with several PyTorch versions:

```bash
.venv/bin/python -m pytest -q -p no:cacheprovider \
  -m "not cuda and not sm120 and not weights and not oracle and not slow" tests
```

For local checks, run the unit suite without CUDA:

```bash
.venv/bin/python -m pytest -q -p no:cacheprovider -m "not cuda" tests/unit
```

Tests marked `cuda`, `sm120`, `weights`, `oracle`, or `slow` need the corresponding GPU, checkpoint, vendored oracle,
or longer-running environment. CI also compiles CUDA C++, Triton, and TileLang kernels across the supported NVIDIA
architectures; it does not provide a GPU for runtime tests.

CI lints the whole repository with Ruff; run the same check locally, and apply `--fix` only to files you changed:

```bash
.venv/bin/ruff check .
.venv/bin/ruff check --fix path/to/changed_file.py
```

## Branches and pull requests

Create a branch from `dev` and target pull requests at `dev`. The `main` branch receives merges from `dev`; do not push
directly to either protected branch. Fill in the [pull request template](.github/pull_request_template.md), including the
commands and environments used for verification and anything you could not run.

Verification depends on the change: refactors need bit-identical outputs and call counts, numerical changes need the
pose-AUC gate, and latency changes need the interleaved best-of-N speed protocol with the affected benchmark rows
updated. See [AGENTS.md](AGENTS.md), [architecture documentation](docs/architecture.md), and the relevant detailed
workflow skill ([add-model](.agents/skills/add-model/SKILL.md), [add-op](.agents/skills/add-op/SKILL.md),
[optimize-kernel](.agents/skills/optimize-kernel/SKILL.md), [quantization-accuracy](.agents/skills/quantization-accuracy/SKILL.md),
or [verify-change](.agents/skills/verify-change/SKILL.md)) for the invariants and procedures.

For security reports, see [SECURITY.md](SECURITY.md). For changes made with coding agents, [AGENTS.md](AGENTS.md) describes
the repository rules; human contributors should use this guide as their starting point.
