"""Measure cheap, kernel-free wins on top of a preset: extra compile regions, VGG BN folding.

Each variant runs in its own process (compile caches and autotune state do not leak):

    python benchmarks/profile/quick_wins.py --variant base
    python benchmarks/profile/quick_wins.py --variant vgg_fold
    python benchmarks/profile/quick_wins.py --variant compile_all
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

import torch

from flashim.engines.compile import compile_regions
from flashim.models.romav2.build import load_romav2

ROMAV2_DATA = Path.home() / "work" / "RoMaV2" / "data"
IMAGE_A = ROMAV2_DATA / "megadepth/Undistorted_SfM/0015/images/3383162411_67cc9af85e_o.jpg"
IMAGE_B = ROMAV2_DATA / "megadepth/Undistorted_SfM/0015/images/299056923_ecd980298c_o.jpg"


VARIANTS = {
    "base": ([], False),
    "vgg_fold": ([], True),
    "compile_vgg": (["refiner_features"], True),
    "compile_dpt": (["matcher.head"], False),
    "compile_refiner_blocks": (
        [f"refiners.{s}.{p}" for s in (4, 2, 1) for p in ("block1", "hidden_blocks")],
        False,
    ),
    "compile_all": (
        ["refiner_features", "matcher.head"]
        + [f"refiners.{s}.{p}" for s in (4, 2, 1) for p in ("block1", "hidden_blocks")],
        True,
    ),
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", choices=sorted(VARIANTS), required=True)
    parser.add_argument("--preset", default="fp8")
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--out", default="")
    args = parser.parse_args()
    torch.set_float32_matmul_precision("highest")
    regions, fold = VARIANTS[args.variant]
    model = load_romav2(args.preset, setting="mega1500", engine="compile")
    if fold:
        model.refiner_features.fold_batchnorm(channels_last=True)
    if regions:
        compile_regions(model, regions)
    inputs = model.prepare(IMAGE_A, IMAGE_B)
    with torch.inference_mode():
        for _ in range(10):
            model(*inputs)
        torch.cuda.synchronize()
        times = []
        for _ in range(args.iterations):
            s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            s.record()
            out = model(*inputs)
            e.record()
            e.synchronize()
            times.append(s.elapsed_time(e))
        warp = out["warp_AB"].float().cpu()
        overlap = out["confidence_AB"][..., 0].sigmoid().float().cpu()
    result = {"variant": args.variant, "preset": args.preset, "median_ms": statistics.median(times),
              "min_ms": min(times)}
    print(json.dumps(result))
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        torch.save({"result": result, "warp_AB": warp, "overlap_AB": overlap}, args.out)


if __name__ == "__main__":
    main()
