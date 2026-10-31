"""Paired comparison of MegaDepth-1500 runs from ``run_sparse.py --save-errors`` files ([seed, pair] pose errors).

AUC per seed, averaged over seeds (= run_sparse's number); the difference ``b - a`` with a 95% interval from a paired
bootstrap over the 1500 pairs (the same resampled pairs for both runs), so it covers both the RANSAC noise and how
much the pairs a quantized model gets right or wrong vary.

    python benchmarks/experimental/loma/paired_auc.py results/loma/matches/errors_loma-b_bf16_off0.npy \
        results/loma/matches/errors_loma-b_int8_off0.npy
"""
import argparse
import json

import numpy as np

THRESHOLDS = (5, 10, 20)


def auc_exact(errors_1d: np.ndarray) -> list[float]:
    from flashim.geometry.pose import pose_auc

    return pose_auc(errors_1d, list(THRESHOLDS))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("a")
    parser.add_argument("b")
    parser.add_argument("--reps", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    a, b = np.load(args.a).astype(np.float64), np.load(args.b).astype(np.float64)
    assert a.shape[1] == b.shape[1], (a.shape, b.shape)
    pairs = a.shape[1]
    point_a = np.mean([auc_exact(s) for s in a], 0) * 100
    point_b = np.mean([auc_exact(s) for s in b], 0) * 100
    rng = np.random.default_rng(args.seed)
    deltas = []
    for _ in range(args.reps):
        idx = rng.integers(0, pairs, pairs)
        deltas.append(np.mean([auc_exact(s[idx]) for s in b], 0) - np.mean([auc_exact(s[idx]) for s in a], 0))
    deltas = np.asarray(deltas) * 100
    lo, hi = np.percentile(deltas, [2.5, 97.5], axis=0)
    print(json.dumps({
        "a": args.a, "b": args.b, "seeds": [a.shape[0], b.shape[0]], "pairs": pairs,
        "auc_a": np.round(point_a, 2).tolist(), "auc_b": np.round(point_b, 2).tolist(),
        "delta": np.round(point_b - point_a, 2).tolist(), "ci95_lo": np.round(lo, 2).tolist(),
        "ci95_hi": np.round(hi, 2).tolist(),
    }))


if __name__ == "__main__":
    main()
