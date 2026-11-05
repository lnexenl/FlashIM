"""FlexGlue latency on one fixed MegaDepth pair: ``match_keypoints`` (RDD keypoints) and dense ``match()``.

Keypoints are extracted once (RDD fp8, 2048 per image); ``warmups`` untimed calls (including compilation), then
``iterations`` calls timed with CUDA events. Run each configuration in its own process.

    python benchmarks/speed/flexglue.py --preset fp8 --engine compile [--attention sage-fp8-fast]
"""

from __future__ import annotations

import argparse
import json
import statistics

import torch

import flashim
from flashim.paths import IMAGE_A, IMAGE_B


def timed(fn, warmups: int, iterations: int) -> tuple[list[float], object]:
    for _ in range(warmups):
        out = fn()
    times = []
    for _ in range(iterations):
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        out = fn()
        end.record()
        end.synchronize()
        times.append(start.elapsed_time(end))
    return times, out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--preset", default="fp8")
    parser.add_argument("--engine", default="eager")
    parser.add_argument("--attention", default=None, help="optimize.attention override (e.g. sage, sage-fp8-fast)")
    parser.add_argument("--top-k", type=int, default=2048)
    parser.add_argument("--max-matches", type=int, default=5000)
    parser.add_argument("--warmups", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--tag", default="")
    args = parser.parse_args()

    rdd = flashim.load("rdd", "fp8")
    kpts = [rdd.extract(str(p), top_k=args.top_k).keypoints.reshape(-1, 2) for p in (IMAGE_A, IMAGE_B)]
    del rdd
    overrides = {"optimize.attention": args.attention} if args.attention else {}
    model = flashim.load("flexglue", args.preset, engine=args.engine, overrides=overrides)

    kp_times, kp_result = timed(
        lambda: model.match_keypoints(str(IMAGE_A), kpts[0], str(IMAGE_B), kpts[1]), args.warmups, args.iterations)
    dense_times, dense_result = timed(
        lambda: model.match(str(IMAGE_A), str(IMAGE_B), max_matches=args.max_matches), args.warmups, args.iterations)
    print(json.dumps({
        "tag": args.tag, "preset": model.preset, "engine": args.engine, "attention": args.attention,
        "top_k": args.top_k,
        "keypoints_median_ms": round(statistics.median(kp_times), 3), "keypoints_min_ms": round(min(kp_times), 3),
        "keypoints_matches": len(kp_result.confidence),
        "dense_median_ms": round(statistics.median(dense_times), 3), "dense_min_ms": round(min(dense_times), 3),
        "dense_matches": len(dense_result.confidence),
        "fallbacks": list(getattr(model.build_report, "fallbacks", [])),
        "gpu": torch.cuda.get_device_name(),
    }))


if __name__ == "__main__":
    main()
