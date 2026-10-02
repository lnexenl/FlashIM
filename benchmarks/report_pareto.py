"""Plot the MegaDepth-1500 AUC@5 vs. forward latency Pareto frontier of the sparse / semi-dense models.

    python benchmarks/report_pareto.py [--out results/accuracy/pareto_auc5_vs_ms.png]

Reads the same two result files as ``report_sparse.py``:
``results/accuracy/sparse_<model>_<preset>.json`` (AUC) and ``results/speed/sparse_speed.jsonl`` (latency).
Forward ms is made comparable across models by always covering the same span -- two images in, matched
keypoints out:

- standalone extractors (MNN / ratio-test matching) and ``*-lightglue`` pipelines:
  ``2 * extract_median_ms + match_median_ms`` (both images' extraction + the match)
- LoFTR (detector-free, no separate extract stage): ``match_median_ms`` of the pair (already covers both
  images, since LoFTR has no per-image extract stage to double)

A point is on the Pareto frontier if no other point has both >= AUC@5 and <= forward ms (with at least one
strictly better); frontier points are connected in latency order and labelled ``model/preset``.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PRESETS = ("fp32", "bf16", "int8", "fp8")
MODELS = [
    "superpoint", "superpoint-lightglue", "disk", "disk-lightglue", "alike", "xfeat", "xfeat-lightglue",
    "dedode", "dedode-g", "dad", "ripe", "loftr", "sift", "orb",
]

# RoMaV2 (dense matcher): (preset, engine, forward_ms, auc_5). Forward ms is already "two images in, dense
# correspondences out" (RoMaV2's own compiled forward benchmark, no separate extract/match split to combine).
# Sourced from the curated Notion Results database rather than results/speed/*.json, whose RoMaV2 entries are
# a mix of ad-hoc dumps from before the current benchmarks/ naming convention (abc_, step1_, int8opt_, ...) --
# these 7 are the ones with both a Forward ms and a matching Mega1500 AUC@5 recorded.
ROMAV2 = [
    ("bf16", "compile", 344, 0.6210),
    ("fp8", "compile", 204, 0.6173),
    ("int8", "compile", 324, 0.6241),
    ("int8-full", "compile", 194, 0.6201),
    ("fp8", "compile-full", 165, 0.6194),
    ("fp8-fast", "compile-full", 110, 0.6187),
    ("int8-fast", "compile-full", 103, 0.6222),
]

# unified-protocol results (sparse_matrix.sh); older runs (OpenCV / per-model resize) stay in results/accuracy/
ACCURACY_DIR = ROOT / "results" / "accuracy" / os.environ.get("FLASHIM_ACCURACY_SUBDIR", "unified")


def accuracy() -> dict[tuple[str, str], dict]:
    out = {}
    for path in ACCURACY_DIR.glob("sparse_*.json"):
        data = json.loads(path.read_text())
        out[(data["model"], data["preset"])] = data
    return out


def speed() -> dict[tuple[str, str], dict]:
    """Keyed by ``(model, preset)`` with the base preset (``"fp32+fp32"`` -> ``"fp32"`` for *-lightglue rows,
    whose own ``preset`` field is ``"<extractor>+<matcher>"``, always equal since both run at the same precision)."""
    path = ROOT / "results" / "speed" / os.environ.get("FLASHIM_ACCURACY_SUBDIR", "unified") / "sparse_speed.jsonl"
    out = {}
    if not path.exists():
        return out
    for line in path.read_text().splitlines():
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if row.get("engine", "eager") != "eager":
            continue
        out[(row["model"], row["preset"].split("+")[0])] = row
    return out


def forward_ms(model: str, row: dict | None) -> float | None:
    """Two images in, matched keypoints out -- comparable across detector-based and detector-free models."""
    if row is None:
        return None
    if model == "loftr":  # pair-level: match_median_ms already covers both images, nothing to double
        return row.get("match_median_ms")
    extract, match = row.get("extract_median_ms"), row.get("match_median_ms")
    return None if extract is None or match is None else 2 * extract + match


def pareto_frontier(points: list[tuple[float, float, str]]) -> list[tuple[float, float, str]]:
    """``points`` as ``(ms, auc5, label)``; keeps those with no other point both <= ms and >= auc5 (one strict)."""
    frontier = []
    for p in points:
        dominated = any(
            q is not p and q[0] <= p[0] and q[1] >= p[1] and (q[0] < p[0] or q[1] > p[1]) for q in points
        )
        if not dominated:
            frontier.append(p)
    return sorted(frontier)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", default=str(ROOT / "results" / "accuracy" / "pareto_auc5_vs_ms.png"))
    parser.add_argument("--label-all", action="store_true", help="label every point, not just the frontier")
    args = parser.parse_args()

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    acc, spd = accuracy(), speed()
    marker = {"fp32": "o", "bf16": "s", "int8": "^", "fp8": "D"}
    colors = plt.get_cmap("tab20").colors
    color = {model: colors[i % len(colors)] for i, model in enumerate(MODELS)}

    fig, ax = plt.subplots(figsize=(10, 7))
    points: list[tuple[float, float, str]] = []
    seen_models: list[str] = []
    for model in MODELS:
        for preset in PRESETS:
            a, s = acc.get((model, preset)), spd.get((model, preset))
            ms = forward_ms(model, s)
            if a is None or ms is None:
                continue
            auc5 = 100 * a["auc_5"]
            label = f"{model}/{preset}"
            points.append((ms, auc5, label))
            first = model not in seen_models
            if first:
                seen_models.append(model)
            ax.scatter(
                ms, auc5, marker=marker[preset], color=color[model], s=70, zorder=3,
                edgecolors="black", linewidths=0.4, label=model if first else None,
            )
            if args.label_all:
                ax.annotate(label, (ms, auc5), fontsize=6, xytext=(4, 3), textcoords="offset points")

    for i, (preset, engine, ms, auc5_frac) in enumerate(ROMAV2):
        auc5 = 100 * auc5_frac
        label = f"romav2/{preset}-{engine}"
        points.append((ms, auc5, label))
        ax.scatter(
            ms, auc5, marker="*", color="black", s=160, zorder=4,
            edgecolors="black", linewidths=0.4, label="romav2 (dense)" if i == 0 else None,
        )
        if args.label_all:
            ax.annotate(label, (ms, auc5), fontsize=6, xytext=(4, 3), textcoords="offset points")

    frontier = pareto_frontier(points)
    if frontier:
        fx, fy, flabels = zip(*frontier)
        ax.plot(fx, fy, color="black", linewidth=1.2, linestyle="--", zorder=2, label="Pareto frontier")
        if not args.label_all:
            for x, y, label in frontier:
                ax.annotate(label, (x, y), fontsize=7, xytext=(4, 3), textcoords="offset points")

    ax.set_xscale("log")
    ax.set_xlabel("Forward latency, ms (log scale; B=1, RTX 5070 Ti, eager)")
    ax.set_ylabel("MegaDepth-1500 pose AUC@5 (%)")
    ax.set_title("Sparse / semi-dense models: accuracy vs. speed")
    ax.grid(True, which="both", linestyle=":", linewidth=0.5, alpha=0.6)

    # Two legends: model -> color (marker shape irrelevant), preset -> marker shape (color irrelevant).
    model_legend = ax.legend(loc="lower right", fontsize=7, title="model", ncol=2)
    ax.add_artist(model_legend)
    preset_handles = [
        plt.Line2D([], [], marker=m, color="gray", linestyle="", markeredgecolor="black", markeredgewidth=0.4)
        for m in marker.values()
    ]
    ax.legend(preset_handles, marker.keys(), loc="lower left", fontsize=7, title="preset")

    fig.tight_layout()
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    print(f"{len(points)} points, {len(frontier)} on the Pareto frontier -> {out_path}")
    for x, y, label in frontier:
        print(f"  {label}: AUC@5={y:.1f}%  {x:.2f} ms")


if __name__ == "__main__":
    main()
