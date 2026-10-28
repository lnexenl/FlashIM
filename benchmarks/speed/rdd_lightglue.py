"""Full RDD + LightGlue pipeline latency: extract(A) + extract(B) + match, on one fixed MegaDepth pair.

LightGlue is never used alone; the detector/descriptor extraction is part of every real call. Unlike
``benchmarks/speed/lightglue.py`` (matcher only, features precomputed outside the timed loop), this times
:meth:`SparsePipeline.match` end to end, and also breaks out extract-A / extract-B / match so the split is
visible. ``warmups`` untimed calls (including compilation), then ``iterations`` timed with CUDA events.
Run each configuration in its own process.

    python benchmarks/speed/rdd_lightglue.py --preset fp8 --rdd-preset fp8 --engine compile
"""

from __future__ import annotations

import argparse
import json
import statistics

import torch

import flashim
from flashim.paths import IMAGE_A, IMAGE_B


def timed(fn) -> float:
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record()
    result = fn()
    end.record()
    end.synchronize()
    return start.elapsed_time(end), result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--preset", default="fp8", help="LightGlue preset")
    parser.add_argument("--rdd-preset", default="fp8", help="RDD preset")
    parser.add_argument("--engine", default="eager")
    parser.add_argument("--attention", default="sdpa")
    parser.add_argument("--resize", type=int, default=1600)
    parser.add_argument("--top-k", type=int, default=4096)
    parser.add_argument("--warmups", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--extra", default="", help="JSON dict of dotted config overrides")
    args = parser.parse_args()

    overrides = {
        "components.extractor.preset": args.rdd_preset,
        "components.matcher.optimize.attention": args.attention,
        **(json.loads(args.extra) if args.extra else {}),
    }
    pipeline = flashim.load("rdd-lightglue", args.preset, engine=args.engine, overrides=overrides)

    def full():
        return pipeline.match(str(IMAGE_A), str(IMAGE_B), resize=args.resize, max_matches=args.top_k)

    def parts():
        a = pipeline.extract(str(IMAGE_A), resize=args.resize, top_k=args.top_k)
        b = pipeline.extract(str(IMAGE_B), resize=args.resize, top_k=args.top_k)
        r = pipeline.match_features(a, b)
        return a, b, r

    for _ in range(args.warmups):
        full()
    total_times, result = [], None
    for _ in range(args.iterations):
        ms, result = timed(full)
        total_times.append(ms)

    for _ in range(args.warmups):
        parts()
    extract_a_times, extract_b_times, match_times = [], [], []
    for _ in range(args.iterations):
        start_a, end_a = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start_b, end_b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start_m, end_m = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start_a.record()
        a = pipeline.extract(str(IMAGE_A), resize=args.resize, top_k=args.top_k)
        end_a.record()
        start_b.record()
        b = pipeline.extract(str(IMAGE_B), resize=args.resize, top_k=args.top_k)
        end_b.record()
        start_m.record()
        pipeline.match_features(a, b)
        end_m.record()
        end_m.synchronize()
        extract_a_times.append(start_a.elapsed_time(end_a))
        extract_b_times.append(start_b.elapsed_time(end_b))
        match_times.append(start_m.elapsed_time(end_m))

    def stats(times: list[float]) -> dict:
        return {"median_ms": round(statistics.median(times), 3), "min_ms": round(min(times), 3)}

    print(json.dumps({
        "preset": pipeline.matcher.preset, "rdd_preset": pipeline.extractor.preset, "engine": args.engine, "attention": args.attention,
        "resize": args.resize, "top_k": args.top_k, "matches": len(result.confidence),
        "full_pipeline": stats(total_times),
        "extract_a": stats(extract_a_times), "extract_b": stats(extract_b_times), "match": stats(match_times),
        "sum_parts_median_ms": round(
            statistics.median(extract_a_times) + statistics.median(extract_b_times) + statistics.median(match_times), 3
        ),
        "gpu": torch.cuda.get_device_name(),
    }))


if __name__ == "__main__":
    main()
