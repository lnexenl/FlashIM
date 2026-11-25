"""Per-model evaluation protocol for the integration/perf-all MegaDepth-1500 + ScanNet-1500 accuracy gate.

Shared by ``perf_matrix.sh`` (which executes the runs) and ``perf_report.py`` (which aggregates and gates them).
Every touched model from ``perf-dense-matchers``, ``perf-learned-extractors``, ``perf-classic-extractors``, and
``perf-loma`` is evaluated on MegaDepth-1500 and ScanNet-1500 across 3 RANSAC seeds (17, 29, 42):

- **RoMaV2** (``runner="dense"``): ``benchmarks/accuracy/run.py`` (compile engine, OpenCV 0.5 px, 5 samples/shuffles per
  pair, ``arch.setting=mega1500|scannet1500``), gating ``bf16`` / ``int8`` / ``fp8`` against the ``dev`` (``f9ce273``)
  ``bf16`` compile baseline.
- **All sparse / semi-dense / LightGlue / LoMa models** (``runner="sparse"``): ``benchmarks/accuracy/run_sparse.py``
  (eager engine, poselib 1.0 px, seeds 17 29 42; MegaDepth 1200 px or LoMa's 1024 px, ScanNet 640x480), gating every
  distinct numerics-changing preset (``bf16``, ``int8``, ``fp8``; ``fp32`` for ``orb``/``sift``) against the ``dev``
  (``f9ce273``) ``bf16`` (or ``fp32`` for ``orb``/``sift``) baseline.

CLI helper for ``perf_matrix.sh``:
    python benchmarks/accuracy/perf_protocol.py --list-jobs [--datasets ...] [--models ...]
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass


@dataclass(frozen=True)
class ModelProtocol:
    """One ``(dataset, model)`` evaluation specification."""

    key: str  # logical key used for filtering
    dataset: str  # "mega1500" | "scannet1500"
    model: str  # FlashIM registry name loaded via flashim.load
    runner: str  # "dense" (run.py) | "sparse" (run_sparse.py)
    resize: int  # long side fed to model.match (0 = model default; unused by dense runner)
    top_k: int  # max_matches (0 = all; unused by dense runner)
    base_preset: str = "bf16"  # dev baseline preset ("bf16" or "fp32" for weightless orb/sift)
    presets: tuple[str, ...] = ("bf16", "int8", "fp8")  # integration presets to gate
    overrides: tuple[str, ...] = ()
    note: str = ""


def _sparse_pair(
    model: str,
    *,
    mega_resize: int = 1200,
    scan_resize: int = 640,
    top_k: int = 4096,
    base_preset: str = "bf16",
    presets: tuple[str, ...] = ("bf16", "int8", "fp8"),
    note: str = "",
) -> tuple[ModelProtocol, ModelProtocol]:
    return (
        ModelProtocol(model, "mega1500", model, "sparse", mega_resize, top_k, base_preset, presets, (), note),
        ModelProtocol(model, "scannet1500", model, "sparse", scan_resize, top_k, base_preset, presets, (), note),
    )


# Ordered with fast classic extractors first so a fresh launch completes its first several jobs quickly for
# verification, followed by learned extractors, LightGlue pairings, LoFTR, LoMa variants, and RoMaV2 compile.
_ORDERED_PAIRS: tuple[tuple[ModelProtocol, ModelProtocol], ...] = (
    # Classic extractors
    _sparse_pair("orb", base_preset="fp32", presets=("fp32",), note="weightless (fp32 only)"),
    _sparse_pair("sift", base_preset="fp32", presets=("fp32",), note="weightless (fp32 only)"),
    _sparse_pair("superpoint"),
    _sparse_pair("xfeat"),
    _sparse_pair("alike"),
    _sparse_pair("aliked", presets=("bf16",), note="no int8/fp8 preset"),
    _sparse_pair("disk"),
    # Learned extractors (raco is detector-only; raco-aliked exercises RaCo detector + ranker)
    _sparse_pair("raco-aliked", presets=("bf16",), note="RaCo detector+ranker + ALIKED descriptor"),
    _sparse_pair("rdd", presets=("bf16", "fp8"), note="no int8 preset"),
    _sparse_pair("dedode"),
    _sparse_pair("dedode-g"),
    _sparse_pair("dad"),
    _sparse_pair("ripe"),
    # LightGlue pairings
    _sparse_pair("superpoint-lightglue"),
    _sparse_pair("xfeat-lightglue"),
    _sparse_pair("disk-lightglue"),
    _sparse_pair("aliked-lightglue"),
    _sparse_pair("raco-aliked-lightglue"),
    _sparse_pair("rdd-lightglue"),
    # Semi-dense matcher
    _sparse_pair("loftr", top_k=0, note="semi-dense, all coarse matches"),
    # LoMa variants (upstream 1024 px on MegaDepth, 640x480 on ScanNet)
    _sparse_pair("loma-b128", mega_resize=1024, note="LoMa-B128"),
    _sparse_pair("loma-b", mega_resize=1024, note="LoMa-B"),
    _sparse_pair("loma-r", mega_resize=1024, note="LoMa-R"),
    _sparse_pair("loma-l", mega_resize=1024, note="LoMa-L"),
    _sparse_pair("loma-g", mega_resize=1024, note="LoMa-G"),
    # Dense matcher (RoMaV2 compile protocol via run.py)
    (
        ModelProtocol("romav2", "mega1500", "romav2", "dense", 0, 5000, "bf16", ("bf16", "int8", "fp8"), (), "compile"),
        ModelProtocol(
            "romav2", "scannet1500", "romav2", "dense", 0, 5000, "bf16", ("bf16", "int8", "fp8"), (), "compile"
        ),
    ),
    # Sparse-to-dense matcher (FlexGlue dense match() at native 512x512, 5000 KDE-sampled matches)
    (
        ModelProtocol(
            "flexglue", "mega1500", "flexglue", "sparse", 0, 5000, "bf16", ("fp32", "bf16", "int8", "fp8"), (),
            "dense 512x512",
        ),
        ModelProtocol(
            "flexglue", "scannet1500", "flexglue", "sparse", 0, 5000, "bf16", ("fp32", "bf16", "int8", "fp8"), (),
            "dense 512x512",
        ),
    ),
)

PROTOCOLS: tuple[ModelProtocol, ...] = tuple(p for pair in _ORDERED_PAIRS for p in pair)


def resolve_distinct_presets(proto: ModelProtocol) -> tuple[list[tuple[str, str]], list[tuple[str, str]]]:
    """Deduplicate ``proto.presets`` on the current platform via ``RunConfig.digest()`` without loading weights."""
    import flashim
    from flashim.platforms import current_platform

    platform = current_platform()
    engine_ov = "engine.kind=compile" if proto.runner == "dense" else "engine.kind=eager"
    seen: dict[str, str] = {}
    kept: list[tuple[str, str]] = []
    skipped: list[tuple[str, str]] = []
    for preset in proto.presets:
        cfg = flashim.resolve(
            proto.model, preset, overrides=[engine_ov, *proto.overrides], platform=platform
        )
        digest = cfg.digest()
        if digest in seen:
            skipped.append((preset, f"duplicate of {seen[digest]} (fallback)"))
            continue
        seen[digest] = preset
        notes: list[str] = []
        if cfg.preset != preset:
            notes.append(f"preset:{preset}->{cfg.preset}")
        for cname, ccfg in sorted(cfg.components.items()):
            if ccfg.preset != preset:
                notes.append(f"{cname}:{preset}->{ccfg.preset}")
        kept.append((preset, ",".join(notes)))
    return kept, skipped


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--list-jobs", action="store_true", help="print pipe-delimited job lines for perf_matrix.sh")
    parser.add_argument("--datasets", nargs="+", default=["mega1500", "scannet1500"])
    parser.add_argument("--models", nargs="*", default=[])
    parser.add_argument("--no-dev", action="store_true", help="skip dev baseline jobs")
    parser.add_argument(
        "--skip", nargs="*", default=[],
        help="skip new jobs matching '<model>:<preset>' or '<dataset>:<model>:<preset>' (e.g. proven bit-identical)",
    )
    args = parser.parse_args()

    wanted_datasets = set(args.datasets)
    wanted_models = set(args.models)
    skip_specs = set(args.skip)

    if not args.list_jobs:
        for proto in PROTOCOLS:
            if proto.dataset in wanted_datasets and (not wanted_models or proto.model in wanted_models):
                print(
                    f"{proto.dataset:11s} {proto.model:22s} {proto.runner:6s} "
                    f"base={proto.base_preset:4s} presets={','.join(proto.presets)}"
                )
        return

    for proto in PROTOCOLS:
        if proto.dataset not in wanted_datasets:
            continue
        if wanted_models and proto.key not in wanted_models and proto.model not in wanted_models:
            continue
        ov_str = " ".join(proto.overrides)
        if not args.no_dev:
            print(
                f"RUN|dev|{proto.runner}|{proto.dataset}|{proto.model}|{proto.base_preset}"
                f"|{proto.resize}|{proto.top_k}|{ov_str}|dev baseline"
            )
        kept, skipped = resolve_distinct_presets(proto)
        for preset, reason in skipped:
            print(f"# SKIP|new|{proto.dataset}|{proto.model}|{preset}|{reason}")
        for preset, fb_note in kept:
            if f"{proto.model}:{preset}" in skip_specs or f"{proto.dataset}:{proto.model}:{preset}" in skip_specs:
                print(f"# SKIP|new|{proto.dataset}|{proto.model}|{preset}|skipped via --skip")
                continue
            print(
                f"RUN|new|{proto.runner}|{proto.dataset}|{proto.model}|{preset}"
                f"|{proto.resize}|{proto.top_k}|{ov_str}|{fb_note}"
            )


if __name__ == "__main__":
    main()
