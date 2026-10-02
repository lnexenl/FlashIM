"""Peak GPU memory of a sparse matcher on MegaDepth-1500 pairs (unified protocol: 1200 px, 4096 keypoints).

One config per process: ``weights_mb`` is ``memory_allocated`` after loading (weights, packed quantized weights,
calibration buffers); ``peak_mb`` / ``reserved_mb`` are the CUDA caching allocator's ``max_memory_allocated`` /
``max_memory_reserved`` over ``--pairs`` pairs (the first pair of every scene, so image sizes vary). The CUDA context
(~0.3-0.5 GB) is not counted.

    python benchmarks/accuracy/vram.py --model xfeat-lightglue --preset fp8 --top-k 4096 --out results/accuracy/unified/vram.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
from run import MEGA_SCENES, ROMAV2_DATA, _read_rgb

import flashim
import flashim.registry


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True)
    parser.add_argument("--preset", default="fp32")
    parser.add_argument("--resize", type=int, default=1200)
    parser.add_argument("--top-k", type=int, default=4096)
    parser.add_argument("--pairs", type=int, default=5)
    parser.add_argument("--overrides", nargs="*", default=[])
    parser.add_argument("--label", default="")
    parser.add_argument("--out", default="")
    args = parser.parse_args()

    root = ROMAV2_DATA / "megadepth"
    jobs = []
    for scene_name in MEGA_SCENES[: args.pairs]:
        scene = np.load(root / scene_name, allow_pickle=True)
        idx1, idx2 = scene["pair_infos"][0][0]
        jobs.append((root / scene["image_paths"][idx1], root / scene["image_paths"][idx2]))

    torch.cuda.empty_cache()
    model = flashim.load(args.model, args.preset, overrides=["engine.kind=eager", *args.overrides])
    torch.cuda.synchronize()
    weights = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    for a, b in jobs:
        model.match(_read_rgb(a), _read_rgb(b), resize=args.resize, top_k=args.top_k)
    torch.cuda.synchronize()
    mb = 1 / 2**20
    result = {
        "model": args.model,
        "preset": args.preset,
        "label": args.label,
        "resize": args.resize,
        "top_k": args.top_k,
        "pairs": len(jobs),
        "weights_mb": round(weights * mb, 1),
        "peak_mb": round(torch.cuda.max_memory_allocated() * mb, 1),
        "reserved_mb": round(torch.cuda.max_memory_reserved() * mb, 1),
        "gpu": torch.cuda.get_device_name(),
    }
    print(json.dumps(result))
    if args.out:
        with open(args.out, "a") as f:
            f.write(json.dumps(result) + "\n")


if __name__ == "__main__":
    main()
