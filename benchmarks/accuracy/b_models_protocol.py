"""Per-model evaluation protocol for the 10 B-models on MegaDepth-1500 and ScanNet-1500.

Shared by ``b_models_matrix.sh`` (which executes the runs) and ``b_models_report.py`` (which aggregates and gates
them). Every preset of a model uses the exact same resolution, match budget, overrides, and RANSAC protocol so
``int8`` and ``fp8`` are compared apples-to-apples against the ``bf16`` baseline under ``engine=eager`` and
poselib LO-RANSAC (1.0 px, seeds 17 29 42).

CLI helper for ``b_models_matrix.sh``:
    python benchmarks/accuracy/b_models_protocol.py --list-jobs [--datasets ...] [--presets ...] [--models ...]
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass


@dataclass(frozen=True)
class ModelProtocol:
    """One ``(dataset, model)`` evaluation specification."""

    key: str  # logical family key used for filtering (e.g. "roma", "superpoint-superglue")
    dataset: str  # "mega1500" | "scannet1500"
    model: str  # FlashIM registry name loaded via flashim.load
    resize: int  # long side fed to model.match (0 = None, i.e. model's native fixed resolution)
    top_k: int  # max_matches passed to model.match (0 = None, i.e. all matches)
    overrides: tuple[str, ...] = ()
    note: str = ""


# Per-model native/upstream protocol on MegaDepth-1500 and ScanNet-1500 (640x480 evaluation images).
# Dense models keep their native resolution (resize=0 for RoMa/UFM, 1024/640 for Tiny-RoMa) with 5000 balanced
# samples; 3D matchers use their native 512 px; semi-dense models use 840/640 px (XoFTR 640); sparse pipelines use
# 1024/640 px and 1024 keypoints (upstream SuperGlue / OmniGlue default). ScanNet uses indoor weights where available
# (roma-indoor; superpoint-superglue with components.matcher.variant=indoor).
PROTOCOLS: tuple[ModelProtocol, ...] = (
    # -- MegaDepth-1500 --
    ModelProtocol("roma", "mega1500", "roma", 0, 5000, (), "native 560->864 outdoor"),
    ModelProtocol("tiny-roma", "mega1500", "tiny-roma", 1024, 5000, (), "calibrated 1024px"),
    ModelProtocol("eloftr", "mega1500", "eloftr", 840, 0, (), "840px (floors to 832x32)"),
    ModelProtocol("xoftr", "mega1500", "xoftr", 640, 0, (), "native 640px"),
    ModelProtocol("superpoint-superglue", "mega1500", "superpoint-superglue", 1024, 1024, (), "outdoor, 1024 kpts"),
    ModelProtocol("mast3r", "mega1500", "mast3r", 512, 0, (), "native 512px (fp16 compute)"),
    ModelProtocol("dust3r", "mega1500", "dust3r", 512, 0, (), "native 512px, all reciprocal NNs (fp16 compute)"),
    ModelProtocol("ufm", "mega1500", "ufm", 0, 5000, (), "native 560x420"),
    ModelProtocol("minima-loftr", "mega1500", "minima-loftr", 840, 0, (), "840px"),
    ModelProtocol("minima-roma", "mega1500", "minima-roma", 0, 5000, (), "native 560->864"),
    ModelProtocol("omniglue", "mega1500", "omniglue", 1024, 1024, (), "1024px, 1024 kpts"),
    ModelProtocol("ripepp", "mega1500", "ripepp", 1200, 4096, (), "1200px, 4096 kpts (same as ripe)"),
    ModelProtocol("ripepp-lightglue", "mega1500", "ripepp-lightglue", 1200, 4096, (), "1200px, 4096 kpts"),
    # -- ScanNet-1500 (640x480) --
    ModelProtocol("roma", "scannet1500", "roma-indoor", 0, 5000, (), "native 560->864 indoor"),
    ModelProtocol("tiny-roma", "scannet1500", "tiny-roma", 640, 5000, (), "640x480"),
    ModelProtocol("eloftr", "scannet1500", "eloftr", 640, 0, (), "640x480"),
    ModelProtocol("xoftr", "scannet1500", "xoftr", 640, 0, (), "native 640x480"),
    ModelProtocol(
        "superpoint-superglue",
        "scannet1500",
        "superpoint-superglue",
        640,
        1024,
        ("components.matcher.variant=indoor",),
        "indoor weights, 640x480, 1024 kpts",
    ),
    ModelProtocol("mast3r", "scannet1500", "mast3r", 512, 0, (), "native 512px (fp16 compute)"),
    ModelProtocol("dust3r", "scannet1500", "dust3r", 512, 0, (), "native 512px, all reciprocal NNs (fp16 compute)"),
    ModelProtocol("ufm", "scannet1500", "ufm", 0, 5000, (), "native 560x420"),
    ModelProtocol("minima-loftr", "scannet1500", "minima-loftr", 640, 0, (), "640x480"),
    ModelProtocol("minima-roma", "scannet1500", "minima-roma", 0, 5000, (), "native 560->864"),
    ModelProtocol("omniglue", "scannet1500", "omniglue", 640, 1024, (), "640x480, 1024 kpts"),
    ModelProtocol("ripepp", "scannet1500", "ripepp", 640, 4096, (), "640x480, 4096 kpts (same as ripe)"),
    ModelProtocol("ripepp-lightglue", "scannet1500", "ripepp-lightglue", 640, 4096, (), "640x480, 4096 kpts"),
)

DEFAULT_PRESETS = ("fp32", "bf16", "int8", "fp8")


def resolve_distinct_presets(
    proto: ModelProtocol, presets: tuple[str, ...] = DEFAULT_PRESETS
) -> tuple[list[tuple[str, str]], list[tuple[str, str]]]:
    """Resolve ``presets`` against the current platform without loading weights.

    Returns ``(kept, skipped)`` where ``kept`` is ``[(preset, fallback_note), ...]`` for distinct resolved configs
    (deduplicated by ``RunConfig.digest()`` so presets that fall back to another preset are not run twice) and
    ``skipped`` is ``[(preset, reason), ...]``.
    """
    import flashim
    from flashim.platforms import current_platform

    platform = current_platform()
    seen_digests: dict[str, str] = {}
    kept: list[tuple[str, str]] = []
    skipped: list[tuple[str, str]] = []
    for preset in presets:
        cfg = flashim.resolve(
            proto.model, preset, overrides=["engine.kind=eager", *proto.overrides], platform=platform
        )
        digest = cfg.digest()
        if digest in seen_digests:
            skipped.append((preset, f"duplicate of {seen_digests[digest]} (fallback)"))
            continue
        seen_digests[digest] = preset
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
    parser.add_argument("--list-jobs", action="store_true", help="print TSV job lines for b_models_matrix.sh")
    parser.add_argument("--datasets", nargs="+", default=["mega1500", "scannet1500"])
    parser.add_argument("--presets", nargs="+", default=list(DEFAULT_PRESETS))
    parser.add_argument("--models", nargs="*", default=[])
    args = parser.parse_args()

    if not args.list_jobs:
        for proto in PROTOCOLS:
            print(f"{proto.dataset:12s} {proto.model:22s} resize={proto.resize:4d} top_k={proto.top_k:4d} {proto.overrides}")
        return

    wanted_datasets = set(args.datasets)
    wanted_models = set(args.models)
    presets = tuple(args.presets)
    for proto in PROTOCOLS:
        if proto.dataset not in wanted_datasets:
            continue
        if wanted_models and proto.key not in wanted_models and proto.model not in wanted_models:
            continue
        kept, skipped = resolve_distinct_presets(proto, presets)
        for preset, reason in skipped:
            print(f"# SKIP|{proto.dataset}|{proto.model}|{preset}|{reason}")
        for preset, fb_note in kept:
            ov_str = " ".join(proto.overrides)
            print(f"RUN|{proto.dataset}|{proto.model}|{preset}|{proto.resize}|{proto.top_k}|{ov_str}|{fb_note}")


if __name__ == "__main__":
    main()
