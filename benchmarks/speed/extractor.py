"""Sparse-extractor latency: ``extract`` of one fixed MegaDepth image (1600 px long side, 4096 keypoints by default).

``warmups`` untimed calls (including compilation), then ``iterations`` calls timed with CUDA events. Run each
configuration in its own process. Also reports the mutual-NN ``match_features`` time and the keypoint count.

    python benchmarks/speed/extractor.py --model superpoint --preset int8 [--engine compile] [--resize 1600]
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

import torch

import flashim

IMAGE = Path.home() / "work/RoMaV2/data/megadepth/Undistorted_SfM/0015/images/3383162411_67cc9af85e_o.jpg"
IMAGE_B = Path.home() / "work/RoMaV2/data/megadepth/Undistorted_SfM/0015/images/299056923_ecd980298c_o.jpg"


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
    parser.add_argument("--model", required=True)
    parser.add_argument("--preset", default=None)
    parser.add_argument("--engine", default="eager")
    parser.add_argument("--resize", type=int, default=1600)
    parser.add_argument("--top-k", type=int, default=4096)
    parser.add_argument("--warmups", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--overrides", nargs="*", default=[], help="dotted config overrides")
    parser.add_argument("--tag", default="", help="copied to the JSON line (sparse_matrix.sh uses it to resume)")
    args = parser.parse_args()

    model = flashim.load(args.model, args.preset, overrides=[f"engine.kind={args.engine}", *args.overrides])
    from flashim.models.romav2.model import load_image

    image = load_image(str(IMAGE), model.device)
    if not hasattr(model, "extract"):  # pair-level matchers (LoFTR): time ``match`` on the fixed pair
        image_b = load_image(str(IMAGE_B), model.device)
        top_k = args.top_k if args.top_k > 0 else None
        times, result = timed(lambda: model.match(image, image_b, resize=args.resize, top_k=top_k), args.warmups, args.iterations)
        print(json.dumps({
            "tag": args.tag, "model": args.model, "preset": model.preset, "engine": args.engine, "resize": args.resize, "pair_level": True,
            "match_median_ms": round(statistics.median(times), 3), "match_min_ms": round(min(times), 3),
            "matches": len(result.confidence), "peak_mem_mb": round(torch.cuda.max_memory_allocated() / 2**20, 1),
            "gpu": torch.cuda.get_device_name(),
        }))
        return
    times, feats = timed(lambda: model.extract(image, resize=args.resize, top_k=args.top_k), args.warmups, args.iterations)
    match_ms = match_count = None
    if hasattr(model, "match_features"):  # RaCo-ALIKED only feeds LightGlue: no matcher of its own
        other = model.extract(str(IMAGE_B), resize=args.resize, top_k=args.top_k)
        match_times, matches = timed(lambda: model.match_features(feats, other), 3, 20)
        match_ms, match_count = round(statistics.median(match_times), 3), len(matches.confidence)
    print(json.dumps({
        "tag": args.tag, "model": args.model, "preset": model.preset, "engine": args.engine, "resize": args.resize, "top_k": args.top_k,
        "keypoints": int(feats.keypoints.shape[1]),
        "extract_median_ms": round(statistics.median(times), 3), "extract_min_ms": round(min(times), 3),
        "match_median_ms": match_ms, "matches": match_count,
        "peak_mem_mb": round(torch.cuda.max_memory_allocated() / 2**20, 1),
        "gpu": torch.cuda.get_device_name(),
    }))


if __name__ == "__main__":
    main()
