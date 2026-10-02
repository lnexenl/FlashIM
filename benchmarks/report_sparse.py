"""Collect the sparse-model benchmark results into one Markdown report.

    python benchmarks/report_sparse.py > results/sparse_models.md

Reads ``results/accuracy/sparse_<model>_<preset>.json`` (MegaDepth-1500 AUC from ``benchmarks/accuracy/sparse_matrix.sh``)
and ``results/speed/sparse_speed.jsonl`` (``benchmarks/speed/sparse_matrix.sh``); missing entries print as ``-``.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PRESETS = ("fp32", "bf16", "int8", "fp8")
MODELS = [
    "superpoint", "superpoint-lightglue", "disk", "disk-lightglue", "alike", "xfeat", "xfeat-lightglue",
    "dedode", "dedode-g", "dad", "ripe", "loftr", "sift", "orb",
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
    path = ROOT / "results" / "speed" / os.environ.get("FLASHIM_ACCURACY_SUBDIR", "unified") / "sparse_speed.jsonl"
    out = {}
    if path.exists():
        for line in path.read_text().splitlines():
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if row.get("engine", "eager") == "eager":
                out[(row["model"], row["preset"])] = row
    return out


def latency(row: dict | None) -> float | None:
    if row is None:
        return None
    return row.get("extract_median_ms", row.get("match_median_ms"))


def fmt(value: float | None, digits: int = 1, scale: float = 1.0) -> str:
    return "-" if value is None else f"{value * scale:.{digits}f}"


def main() -> None:
    acc, spd = accuracy(), speed()
    print("# Sparse / semi-dense models: MegaDepth-1500 pose AUC and extraction latency\n")
    print("AUC@5 / @10 / @20 in percent (relative pose, unified protocol: 1200 px, poselib LO-RANSAC at 1.0 px, see benchmarks/accuracy/run_sparse.py); latency is the median")
    print("`extract` time (LoFTR: `match`) of one fixed MegaDepth image in ms (RTX 5070 Ti, eager). `x` is the speed-up over the")
    print("model's bf16 row. SIFT / ORB have no weights: fp32 only.\n")
    print("| model | preset | AUC@5 | AUC@10 | AUC@20 | pairs | ms | x vs bf16 |")
    print("|---|---|---|---|---|---|---|---|")
    for model in MODELS:
        base = latency(spd.get((model, "bf16")))
        for preset in PRESETS:
            a, s = acc.get((model, preset)), spd.get((model, preset))
            if a is None and s is None:
                continue
            ms = latency(s)
            ratio = None if ms is None or base is None else base / ms
            print(
                f"| {model} | {preset} | {fmt(a and a['auc_5'], 1, 100)} | {fmt(a and a['auc_10'], 1, 100)} | "
                f"{fmt(a and a['auc_20'], 1, 100)} | {a['pairs'] if a else '-'} pairs | {fmt(ms, 2)} | {fmt(ratio, 2)} |"
            )


def fidelity() -> None:
    path = ROOT / "results" / "accuracy" / "fidelity.jsonl"
    if not path.exists():
        return
    print("\n## Fidelity to the model's own fp32 output (held-out MegaDepth-1500 pairs; proxy, not task accuracy)\n")
    print("`kp` = fraction of fp32 keypoints also detected within 1.5 px; `cos` = median descriptor cosine at those; `agree` =")
    print("fraction of fp32 matches reproduced within 2 px on both sides; `ratio` = match count over fp32's.\n")
    print("| model | preset | kp | cos | agree | ratio |")
    print("|---|---|---|---|---|---|")
    for line in path.read_text().splitlines():
        r = json.loads(line)
        print(
            f"| {r['model']} | {r['preset']} | {fmt(r['kp_overlap'], 3)} | {fmt(r['desc_cos'], 3)} | "
            f"{fmt(r['match_agree'], 3)} | {fmt(r['match_ratio'], 3)} |"
        )


if __name__ == "__main__":
    main()
    fidelity()
