"""Top CUDA kernels of one sparse-extractor ``extract`` call (torch.profiler), grouped by what they compute.

    python benchmarks/profile/extractor_profile.py --model superpoint --preset int8 [--resize 1600 --top-k 4096]

Used to check where an extractor's time goes: the kernel table separates the GEMM/conv compute from
quantize / im2col byte movement / output layout copies. INT8 convs (``quant/methods/conv2d.py``) dispatch to
``ops.int8_conv2d``'s fused implicit-GEMM kernel (no im2col buffer); MXFP8 convs still go through the older
im2col-then-GEMM pipeline, which this script was originally written to diagnose as slower than BF16.
"""

from __future__ import annotations

import argparse
from collections import defaultdict

import torch
from torch.profiler import ProfilerActivity, profile

import flashim
from flashim.models.romav2.model import load_image
from flashim.paths import IMAGE_A

IMAGE = IMAGE_A

GROUPS = (
    ("gemm", ("gemm", "cutlass", "cublas", "nvjet", "int8_linear", "scaled_mm", "_mm", "sm80_xmma", "sm90_xmma")),
    ("conv", ("conv", "cudnn", "implicit_convolve", "xmma_fprop")),
    ("quantize", ("quantize", "_q8", "to_int8", "round", "clamp")),
    ("data movement", ("copy", "cat", "pad", "permute", "contiguous", "clone", "index", "gather", "slice")),
)


def group_of(name: str) -> str:
    lowered = name.lower()
    for group, needles in GROUPS:
        if any(n in lowered for n in needles):
            return group
    return "other"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True)
    parser.add_argument("--preset", default=None)
    parser.add_argument("--resize", type=int, default=1600)
    parser.add_argument("--top-k", type=int, default=4096)
    parser.add_argument("--rows", type=int, default=12)
    args = parser.parse_args()

    model = flashim.load(args.model, args.preset)
    image = load_image(str(IMAGE), model.device)
    for _ in range(5):
        model.extract(image, resize=args.resize, top_k=args.top_k)
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA, ProfilerActivity.CPU]) as prof:
        model.extract(image, resize=args.resize, top_k=args.top_k)
        torch.cuda.synchronize()
    events = [e for e in prof.key_averages() if e.device_time_total > 0 and e.device_type == torch.autograd.DeviceType.CUDA]
    total = sum(e.device_time_total for e in events)
    grouped: dict[str, float] = defaultdict(float)
    for e in events:
        grouped[group_of(e.key)] += e.device_time_total
    print(f"{args.model} {args.preset}: {total / 1000:.2f} ms of CUDA kernel time")
    for group, t in sorted(grouped.items(), key=lambda kv: -kv[1]):
        print(f"  {group:14s} {t / 1000:7.2f} ms  {100 * t / total:5.1f}%")
    print("top kernels:")
    for e in sorted(events, key=lambda e: -e.device_time_total)[: args.rows]:
        print(f"  {e.device_time_total / 1000:7.2f} ms  x{e.count:<3d} {e.key[:100]}")


if __name__ == "__main__":
    main()
