"""LightGlue matching latency on the features of one fixed MegaDepth pair (RDD, 4096 keypoints per image).

Features are extracted once; ``warmups`` untimed ``match_features`` calls (including compilation), then
``iterations`` calls timed with CUDA events. Run each configuration in its own process.

    python benchmarks/speed/lightglue.py --preset fp8 --engine compile
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

import torch

import flashim

ROMAV2_ROOT = Path.home() / "work" / "RoMaV2"
IMAGE_A = ROMAV2_ROOT / "data/megadepth/Undistorted_SfM/0015/images/3383162411_67cc9af85e_o.jpg"
IMAGE_B = ROMAV2_ROOT / "data/megadepth/Undistorted_SfM/0015/images/299056923_ecd980298c_o.jpg"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--preset", default="fp8")
    parser.add_argument("--engine", default="eager")
    parser.add_argument("--attention", default="sdpa")
    parser.add_argument("--top-k", type=int, default=4096)
    parser.add_argument("--warmups", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--extra", default="", help="JSON dict of extra flashim.load kwargs")
    args = parser.parse_args()

    rdd = flashim.load("rdd", preset="fp8")
    a, b = (rdd.extract(str(p), resize=1600, top_k=args.top_k) for p in (IMAGE_A, IMAGE_B))
    del rdd
    model = flashim.load(
        "lightglue", preset=args.preset, engine=args.engine, attention=args.attention,
        **(json.loads(args.extra) if args.extra else {}),
    )
    for _ in range(args.warmups):
        model.match_features(a, b)
    times = []
    for _ in range(args.iterations):
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        result = model.match_features(a, b)
        end.record()
        end.synchronize()
        times.append(start.elapsed_time(end))
    print(json.dumps({
        "preset": model.preset, "engine": args.engine, "attention": args.attention, "top_k": args.top_k,
        "matches": len(result.confidence),
        "median_ms": round(statistics.median(times), 3), "min_ms": round(min(times), 3),
        "gpu": torch.cuda.get_device_name(),
    }))


if __name__ == "__main__":
    main()
