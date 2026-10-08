"""Where the time of one sparse / semi-dense configuration goes (unified setting: 1200 px, 4096 keypoints).

Per stage (``extract`` of one image, ``match_features`` of the fixed pair; ``match`` for pair-level models): wall time
(median of the timed calls, CUDA-synchronized, without the profiler), then ``torch.profiler`` over a few calls: GPU
kernel time and launches per call, the share of the wall time the GPU is busy, kernels grouped by what they compute,
the top kernels and the ``flashim::<op>[impl]`` op ranges. One configuration per process.

    python benchmarks/profile/sparse_sweep.py --model superpoint-lightglue --preset bf16 --out results/profile/unified_sweep
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from collections import defaultdict
from pathlib import Path

import torch
from torch.profiler import ProfilerActivity, profile

import flashim
from flashim.models.romav2.model import load_image
from flashim.paths import MEGADEPTH_IMAGES

IMAGES = MEGADEPTH_IMAGES
IMAGE_A, IMAGE_B = IMAGES / "3383162411_67cc9af85e_o.jpg", IMAGES / "299056923_ecd980298c_o.jpg"

# first match wins
GROUPS = (
    ("layout nchw<->nhwc", ("nchwtonhwc", "nhwctonchw")),
    ("conv (cudnn)", ("cudnn", "implicit_convolve", "fprop", "conv")),
    ("attention (flash)", ("flash_fwd", "flash_bwd")),
    ("gemm fp32 simt", ("simt", "sgemm", "magma")),
    ("gemm tensor core", ("tensorop", "xmma", "nvjet", "gemm", "cutlass")),
    ("softmax", ("softmax",)),
    ("norm", ("layer_norm", "batch_norm", "group_norm")),
    ("pool / nms", ("pool",)),
    ("resample", ("upsample", "grid_sampler", "interpolate")),
    ("sort / topk / index", ("sort", "topk", "radix", "index", "gather", "scatter", "nonzero", "unique", "merge", "copy_if", "partition", "reduce")),
    ("memcpy / fill", ("memcpy", "memset", "copy", "fill")),
    ("elementwise", ("elementwise", "launch_cla", "unary", "binary")),
)


def group_of(name: str) -> str:
    lowered = name.lower()
    if not lowered.startswith(("void ", "mem")):  # Triton / TileLang / JIT CUDA kernels carry their plain function name
        return "flashim kernel"
    for group, needles in GROUPS:
        if any(n in lowered for n in needles):
            return group
    return "other"


def wall(fn, budget_s: float = 4.0) -> tuple[float, int]:
    for _ in range(5):
        fn()
    torch.cuda.synchronize()
    times, start = [], time.perf_counter()
    while len(times) < 10 or (len(times) < 50 and time.perf_counter() - start < budget_s):
        t = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        times.append((time.perf_counter() - t) * 1e3)
    return statistics.median(times), len(times)


def stage(name: str, fn, calls: int = 4) -> dict:
    wall_ms, timed = wall(fn)
    with profile(activities=[ProfilerActivity.CUDA, ProfilerActivity.CPU]) as prof:
        for _ in range(calls):
            fn()
        torch.cuda.synchronize()
    events = [e for e in prof.key_averages() if e.device_time_total > 0 and e.device_type == torch.autograd.DeviceType.CUDA]
    ranges = [e for e in events if e.key.startswith("flashim::")]  # op ranges: their kernels are listed separately
    kernels = [e for e in events if not e.key.startswith("flashim::")]
    gpu_ms = sum(e.device_time_total for e in kernels) / calls / 1e3
    groups: dict[str, float] = defaultdict(float)
    for e in kernels:
        groups[group_of(e.key)] += e.device_time_total / calls / 1e3
    row = lambda e: {"name": e.key[:110], "ms": round(e.device_time_total / calls / 1e3, 3), "calls": e.count // calls}
    return {
        "stage": name,
        "wall_ms": round(wall_ms, 3),
        "timed_calls": timed,
        "gpu_ms": round(gpu_ms, 3),
        "gpu_busy": round(gpu_ms / wall_ms, 3),
        "launches": sum(e.count for e in kernels) // calls,
        "groups": {k: round(v, 3) for k, v in sorted(groups.items(), key=lambda kv: -kv[1])},
        "kernels": [row(e) for e in sorted(kernels, key=lambda e: -e.device_time_total)[:12]],
        "ops": [row(e) for e in sorted(ranges, key=lambda e: -e.device_time_total)[:8]],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True)
    parser.add_argument("--preset", default="bf16")
    parser.add_argument("--resize", type=int, default=1200)
    parser.add_argument("--top-k", type=int, default=4096)
    parser.add_argument("--overrides", nargs="*", default=[])
    parser.add_argument("--tag", default="")
    parser.add_argument("--out", default="")
    args = parser.parse_args()

    model = flashim.load(args.model, args.preset, overrides=["engine.kind=eager", *args.overrides])
    a, b = load_image(str(IMAGE_A), model.device), load_image(str(IMAGE_B), model.device)
    stages = []
    with torch.inference_mode():
        if hasattr(model, "extract"):
            stages.append(stage("extract", lambda: model.extract(a, resize=args.resize, top_k=args.top_k)))
            fa = model.extract(a, resize=args.resize, top_k=args.top_k)
            fb = model.extract(b, resize=args.resize, top_k=args.top_k)
            stages.append(stage("match", lambda: model.match_features(fa, fb)))
        else:
            top_k = args.top_k if args.top_k > 0 else None
            stages.append(stage("match (pair)", lambda: model.match(a, b, resize=args.resize, top_k=top_k)))
    result = {
        "tag": args.tag or f"{args.model}-{args.preset}", "model": args.model, "preset": args.preset,
        "resize": args.resize, "top_k": args.top_k, "gpu": torch.cuda.get_device_name(), "stages": stages,
    }
    for s in stages:
        groups = ", ".join(f"{k} {v:.1f}" for k, v in list(s["groups"].items())[:5])
        print(f"{result['tag']:28s} {s['stage']:13s} wall {s['wall_ms']:8.2f} ms  gpu {s['gpu_ms']:8.2f} ms "
              f"({100 * s['gpu_busy']:3.0f}% busy)  {s['launches']:5d} launches | {groups}")
    if args.out:
        out = Path(args.out)
        out.mkdir(parents=True, exist_ok=True)
        (out / f"{result['tag']}.json").write_text(json.dumps(result, indent=1) + "\n")


if __name__ == "__main__":
    main()
