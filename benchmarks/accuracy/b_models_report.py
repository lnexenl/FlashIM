"""Report MegaDepth-1500 and ScanNet-1500 AUC gate results for the 10 B-models.

Reads ``<dir>/<dataset>_<model>_<preset>.json`` and ``.npy`` (per-pair pose errors ``[seed, pair]`` saved by
``run_sparse.py --save-errors``) and emits a Markdown table with:

- ``AUC@5 / 10 / 20`` per preset (mean over RANSAC seeds, in ``[0, 1]`` units, matching ``AGENTS.md``);
- ``Δ AUC@5 vs bf16`` with a **95% paired bootstrap CI** over pairs (resampling pair indices with replacement across
  all seeds jointly, preserving the exact pair alignment between candidate and ``bf16``);
- **Gate status** for quantized presets (``int8``, ``fp8``) against the threshold ``-gate`` (default ``-0.005``,
  i.e. AUC@5 drop < 0.005 vs ``bf16``):
  - ``PASS``: ``ci_lo > -gate`` (the entire 95% CI lies strictly above ``-0.005``);
  - ``FAIL``: ``ci_hi < -gate`` (the entire 95% CI lies strictly below ``-0.005``);
  - ``BORDERLINE``: ``ci_lo <= -gate <= ci_hi`` (the 95% CI straddles ``-0.005``, statistically inconclusive;
    check the point estimate ``delta`` and sample size);
  - ``BASE`` for ``bf16`` and ``REF`` for ``fp32``.

    python benchmarks/accuracy/b_models_report.py --dir results/accuracy/b_models
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from b_models_protocol import DEFAULT_PRESETS, PROTOCOLS


def _bootstrap_auc5_delta(
    base_err: np.ndarray, cand_err: np.ndarray, *, threshold: float = 5.0, n_boot: int = 10000, seed: int = 0
) -> tuple[float, float, float]:
    """Exact paired bootstrap of ``mean_s AUC@threshold(cand) - mean_s AUC@threshold(base)`` over pairs.

    Uses the exact identity for ``flashim.geometry.pose.pose_auc``: for sorted errors ``e_1 <= ... <= e_N`` with
    ``m`` errors strictly below ``t``, ``AUC@t = (sum_{e_i < t} (t - e_i) + 0.5 * max_{e_i < t} e_i) / (N * t)``.
    Returns ``(point_delta, ci_lo, ci_hi)`` at 95% (2.5th / 97.5th percentiles).
    """
    if base_err.shape != cand_err.shape:
        raise ValueError(f"Mismatched error shapes: base {base_err.shape} vs cand {cand_err.shape}")
    if base_err.ndim == 1:
        base_err = base_err[None, :]
        cand_err = cand_err[None, :]
    n_seeds, n_pairs = base_err.shape

    def exact_auc5_weighted(err_row: np.ndarray, weights: np.ndarray) -> np.ndarray:
        """Vectorized exact ``pose_auc`` at ``threshold`` for ``weights`` ``[B, N]`` (row sums = ``N``)."""
        below = err_row < threshold
        contrib = np.where(below, threshold - err_row, 0.0)  # [N]
        linear = weights @ contrib  # [B]
        # Exact boundary term: 0.5 * max_{i: e_i < t and w_i > 0} e_i.
        below_idx = np.flatnonzero(below)
        if len(below_idx) == 0:
            return np.zeros(weights.shape[0], dtype=np.float64)
        order = below_idx[np.argsort(err_row[below_idx])]
        # Reverse order so argmax on boolean mask picks the largest error < t present in each resample.
        rev = order[::-1]
        present = weights[:, rev] > 0  # [B, M]
        any_present = present.any(axis=1)
        first = present.argmax(axis=1)
        max_e = np.where(any_present, err_row[rev[first]], 0.0)
        return (linear + 0.5 * max_e) / (n_pairs * threshold)

    unit = np.ones((1, n_pairs), dtype=np.float64)
    base_point = float(np.mean([exact_auc5_weighted(base_err[s], unit)[0] for s in range(n_seeds)]))
    cand_point = float(np.mean([exact_auc5_weighted(cand_err[s], unit)[0] for s in range(n_seeds)]))
    point_delta = cand_point - base_point

    rng = np.random.default_rng(seed)
    probs = np.full(n_pairs, 1.0 / n_pairs, dtype=np.float64)
    # Chunk bootstrap matrix to bound peak RAM (e.g. 2000 x 1500 float64 ~= 24 MB).
    chunk = 2000
    deltas = np.empty(n_boot, dtype=np.float64)
    for start in range(0, n_boot, chunk):
        b = min(chunk, n_boot - start)
        w = rng.multinomial(n_pairs, probs, size=b).astype(np.float64)
        diff = np.zeros(b, dtype=np.float64)
        for s in range(n_seeds):
            diff += exact_auc5_weighted(cand_err[s], w) - exact_auc5_weighted(base_err[s], w)
        deltas[start : start + b] = diff / n_seeds
    ci_lo, ci_hi = np.percentile(deltas, [2.5, 97.5])
    return point_delta, float(ci_lo), float(ci_hi)


def decide_gate(preset: str, ci_lo: float, ci_hi: float, gate: float) -> str:
    """3-way gate decision on 95% paired bootstrap CI of ``AUC@5(preset) - AUC@5(bf16)`` vs ``-gate``."""
    if preset == "bf16":
        return "BASE"
    if preset == "fp32":
        return "REF"
    if ci_lo > -gate:
        return "PASS"
    if ci_hi < -gate:
        return "FAIL"
    return "BORDERLINE"


def _fmt_delta(delta: float, lo: float, hi: float) -> str:
    return f"{delta:+.4f} [{lo:+.4f}, {hi:+.4f}]"


def _fmt_fallbacks(meta: dict) -> str:
    fb = meta.get("fallbacks") or {}
    if not fb:
        return ""
    return "; ".join(f"{k}: {v}" for k, v in sorted(fb.items()))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dir", type=Path, default=Path("results/accuracy/b_models"))
    parser.add_argument("--datasets", nargs="+", default=["mega1500", "scannet1500"])
    parser.add_argument("--presets", nargs="+", default=list(DEFAULT_PRESETS))
    parser.add_argument("--gate", type=float, default=0.005, help="max allowed AUC@5 drop vs bf16 (in [0, 1])")
    parser.add_argument("--bootstraps", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    print(
        f"# B-Models Accuracy Gate Report (`{args.dir}`)\n\n"
        f"- **Gate criterion**: `Δ = AUC@5(preset) - AUC@5(bf16)` with 95% paired bootstrap CI (`B={args.bootstraps}` "
        f"over pairs, mean across seeds).\n"
        f"- **Decision rule** (`gate = -{args.gate:.4f}`): "
        f"`PASS` if `CI_lo > -{args.gate:.4f}`; `FAIL` if `CI_hi < -{args.gate:.4f}`; "
        f"`BORDERLINE` if CI straddles `-{args.gate:.4f}`; `BASE` = `bf16`, `REF` = `fp32`.\n"
    )

    counts = {"PASS": 0, "BORDERLINE": 0, "FAIL": 0}
    for dataset in args.datasets:
        protos = [p for p in PROTOCOLS if p.dataset == dataset]
        print(f"## {dataset}\n")
        print(
            "| model | preset | pairs | AUC@5 | AUC@10 | AUC@20 | Δ AUC@5 vs bf16 (95% CI) | gate | sec | notes |"
        )
        print("|---|---|---:|---:|---:|---:|---|---|---:|---|")
        for proto in protos:
            base_json = args.dir / f"{dataset}_{proto.model}_bf16.json"
            base_npy = args.dir / f"{dataset}_{proto.model}_bf16.npy"
            base_err = np.load(base_npy) if base_npy.exists() else None
            base_meta = json.loads(base_json.read_text()) if base_json.exists() else None

            any_row = False
            for preset in args.presets:
                jpath = args.dir / f"{dataset}_{proto.model}_{preset}.json"
                epath = args.dir / f"{dataset}_{proto.model}_{preset}.npy"
                if not jpath.exists():
                    continue
                any_row = True
                meta = json.loads(jpath.read_text())
                err = np.load(epath) if epath.exists() else None
                auc5, auc10, auc20 = meta["auc_5"], meta["auc_10"], meta["auc_20"]
                pairs = meta["pairs"]
                sec = meta.get("seconds", 0.0)
                notes = _fmt_fallbacks(meta)

                if preset == "bf16":
                    delta_str, gate_str = "—", "BASE"
                elif base_err is not None and err is not None:
                    delta, lo, hi = _bootstrap_auc5_delta(
                        base_err, err, threshold=5.0, n_boot=args.bootstraps, seed=args.seed
                    )
                    delta_str = _fmt_delta(delta, lo, hi)
                    gate_str = decide_gate(preset, lo, hi, args.gate)
                elif base_meta is not None:
                    delta = auc5 - base_meta["auc_5"]
                    delta_str = f"{delta:+.4f} (no .npy)"
                    gate_str = "REF" if preset == "fp32" else ("PASS" if delta > -args.gate else "FAIL")
                else:
                    delta_str, gate_str = "no bf16", "MISSING_BASE"

                if gate_str in counts:
                    counts[gate_str] += 1
                print(
                    f"| `{proto.model}` | `{preset}` | {pairs} | {auc5:.4f} | {auc10:.4f} | {auc20:.4f} "
                    f"| {delta_str} | **{gate_str}** | {sec:.1f} | {notes} |"
                )
            if not any_row:
                print(f"| `{proto.model}` | — | — | — | — | — | missing | — | — | {proto.note} |")
        print()

    print(
        f"**Summary (quantized presets)**: {counts['PASS']} PASS, {counts['BORDERLINE']} BORDERLINE, {counts['FAIL']} FAIL."
    )


if __name__ == "__main__":
    main()
