"""Report MegaDepth-1500 and ScanNet-1500 AUC gate results for integration/perf-all vs dev (f9ce273).

Reads ``<dir>/<tree>_<dataset>_<model>_<preset>.json`` / ``.npy`` (sparse/semi-dense/LoMa) and
``<dir>/<tree>_<dataset>_<model>_<preset>_compile_seed<S>.json`` / ``.npy`` (RoMaV2 compile) and emits a Markdown
table with:

- ``AUC@5 / 10 / 20`` (mean over seeds 17, 29, 42, in ``[0, 1]`` units, matching ``AGENTS.md``);
- ``Δ AUC@5 vs dev base`` with a **95% paired bootstrap CI** over pairs (resampling pair indices with replacement
  across all seeds jointly, preserving exact pair alignment between candidate and the ``dev`` ``bf16`` baseline, or
  ``dev`` ``fp32`` for weightless ``orb``/``sift``);
- ``Δ AUC@5 vs new bf16`` (point estimate for quantized ``int8``/``fp8`` presets against ``integration/perf-all``'s own
  ``bf16``);
- **Gate status** against ``-gate`` (default ``-0.005``, i.e. AUC@5 drop < 0.005 vs ``dev`` baseline):
  - ``PASS``: ``ci_lo > -gate`` (entire 95% CI strictly above ``-0.005``);
  - ``FAIL``: ``ci_hi < -gate`` (entire 95% CI strictly below ``-0.005``);
  - ``BORDERLINE``: ``ci_lo <= -gate <= ci_hi`` (95% CI straddles ``-0.005``);
  - ``BASE`` for the ``dev`` baseline row.

    python benchmarks/accuracy/perf_report.py --dir results/accuracy/perf
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from perf_protocol import PROTOCOLS, ModelProtocol


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
        below = err_row < threshold
        contrib = np.where(below, threshold - err_row, 0.0)
        linear = weights @ contrib
        below_idx = np.flatnonzero(below)
        if len(below_idx) == 0:
            return np.zeros(weights.shape[0], dtype=np.float64)
        order = below_idx[np.argsort(err_row[below_idx])]
        rev = order[::-1]
        present = weights[:, rev] > 0
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


def decide_gate(ci_lo: float, ci_hi: float, gate: float) -> str:
    if ci_lo > -gate:
        return "PASS"
    if ci_hi < -gate:
        return "FAIL"
    return "BORDERLINE"


def _load_sparse(root: Path, tree: str, proto: ModelProtocol, preset: str) -> tuple[dict | None, np.ndarray | None]:
    stem = f"{tree}_{proto.dataset}_{proto.model}_{preset}"
    jpath, epath = root / f"{stem}.json", root / f"{stem}.npy"
    if not jpath.exists():
        return None, None
    meta = json.loads(jpath.read_text())
    err = np.load(epath) if epath.exists() else None
    return meta, err


def _load_dense(
    root: Path, tree: str, proto: ModelProtocol, preset: str, seeds: list[int]
) -> tuple[dict | None, np.ndarray | None]:
    metas, errs = [], []
    for s in seeds:
        stem = f"{tree}_{proto.dataset}_{proto.model}_{preset}_compile_seed{s}"
        jpath, epath = root / f"{stem}.json", root / f"{stem}.npy"
        if not jpath.exists():
            return None, None
        metas.append(json.loads(jpath.read_text()))
        if epath.exists():
            errs.append(np.load(epath))
    meta = {
        "pairs": metas[0]["pairs"],
        "auc_5": float(np.mean([m["auc_5"] for m in metas])),
        "auc_10": float(np.mean([m["auc_10"] for m in metas])),
        "auc_20": float(np.mean([m["auc_20"] for m in metas])),
        "seconds": float(sum(m.get("seconds", 0.0) for m in metas)),
        "fallbacks": metas[0].get("fallbacks"),
    }
    err = np.stack(errs, axis=0) if len(errs) == len(seeds) else None
    return meta, err


def _load_entry(
    root: Path, tree: str, proto: ModelProtocol, preset: str, seeds: list[int]
) -> tuple[dict | None, np.ndarray | None]:
    if proto.runner == "dense":
        return _load_dense(root, tree, proto, preset, seeds)
    return _load_sparse(root, tree, proto, preset)


def _fmt_fallbacks(meta: dict) -> str:
    fb = meta.get("fallbacks") or {}
    return "; ".join(f"{k}: {v}" for k, v in sorted(fb.items())) if fb else ""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dir", type=Path, default=Path("results/accuracy/perf"))
    parser.add_argument("--datasets", nargs="+", default=["mega1500", "scannet1500"])
    parser.add_argument("--seeds", type=int, nargs="+", default=[17, 29, 42])
    parser.add_argument("--gate", type=float, default=0.005, help="max allowed AUC@5 drop vs dev baseline (in [0, 1])")
    parser.add_argument("--bootstraps", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    done_path = args.dir / "DONE"
    status = "COMPLETE" if done_path.exists() else "IN PROGRESS (no DONE marker)"
    print(
        f"# Integration Accuracy Gate Report (`{args.dir}`, {status})\n\n"
        f"- **Gate criterion**: `Δ = AUC@5(new preset) - AUC@5(dev base)` with 95% paired bootstrap CI "
        f"(`B={args.bootstraps}` over pairs, mean across seeds {args.seeds}).\n"
        f"- **Decision rule** (`gate = -{args.gate:.4f}`): `PASS` if `CI_lo > -{args.gate:.4f}`; "
        f"`FAIL` if `CI_hi < -{args.gate:.4f}`; `BORDERLINE` if CI straddles `-{args.gate:.4f}`.\n"
    )

    counts = {"PASS": 0, "BORDERLINE": 0, "FAIL": 0, "MISSING_BASE": 0}
    for dataset in args.datasets:
        protos = [p for p in PROTOCOLS if p.dataset == dataset]
        print(f"## {dataset}\n")
        print(
            "| model | tree | preset | pairs | AUC@5 | AUC@10 | AUC@20 | Δ AUC@5 vs dev (95% CI) | Δ vs new bf16 | gate | sec | notes |"
        )
        print("|---|---|---|---:|---:|---:|---:|---|---:|---|---:|---|")
        for proto in protos:
            base_meta, base_err = _load_entry(args.dir, "dev", proto, proto.base_preset, args.seeds)
            new_bf16_meta, _ = _load_entry(args.dir, "new", proto, proto.base_preset, args.seeds)
            any_row = False
            if base_meta is not None:
                any_row = True
                print(
                    f"| `{proto.model}` | `dev` | `{proto.base_preset}` | {base_meta['pairs']} "
                    f"| {base_meta['auc_5']:.4f} | {base_meta['auc_10']:.4f} | {base_meta['auc_20']:.4f} "
                    f"| — | — | **BASE** | {base_meta.get('seconds', 0.0):.1f} | {proto.note} |"
                )
            for preset in proto.presets:
                meta, err = _load_entry(args.dir, "new", proto, preset, args.seeds)
                if meta is None:
                    continue
                any_row = True
                auc5, auc10, auc20 = meta["auc_5"], meta["auc_10"], meta["auc_20"]
                sec = meta.get("seconds", 0.0)
                notes = _fmt_fallbacks(meta)
                if base_err is not None and err is not None:
                    delta, lo, hi = _bootstrap_auc5_delta(
                        base_err, err, threshold=5.0, n_boot=args.bootstraps, seed=args.seed
                    )
                    delta_str = f"{delta:+.4f} [{lo:+.4f}, {hi:+.4f}]"
                    gate_str = decide_gate(lo, hi, args.gate)
                elif base_meta is not None:
                    delta = auc5 - base_meta["auc_5"]
                    delta_str = f"{delta:+.4f} (no .npy)"
                    gate_str = "PASS" if delta > -args.gate else "FAIL"
                else:
                    delta_str, gate_str = "no dev base", "MISSING_BASE"
                if preset != proto.base_preset and new_bf16_meta is not None:
                    d_new = f"{auc5 - new_bf16_meta['auc_5']:+.4f}"
                else:
                    d_new = "—"
                if gate_str in counts:
                    counts[gate_str] += 1
                print(
                    f"| `{proto.model}` | `new` | `{preset}` | {meta['pairs']} "
                    f"| {auc5:.4f} | {auc10:.4f} | {auc20:.4f} "
                    f"| {delta_str} | {d_new} | **{gate_str}** | {sec:.1f} | {notes} |"
                )
            if not any_row:
                print(f"| `{proto.model}` | — | — | — | — | — | — | missing | — | — | — | {proto.note} |")
        print()

    print(
        f"**Summary (new vs dev)**: {counts['PASS']} PASS, {counts['BORDERLINE']} BORDERLINE, "
        f"{counts['FAIL']} FAIL, {counts['MISSING_BASE']} MISSING_BASE."
    )


if __name__ == "__main__":
    main()
