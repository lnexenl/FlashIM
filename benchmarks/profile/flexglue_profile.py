"""Top CUDA kernels of one FlexGlue ``match_keypoints`` call (torch.profiler), eager or compiled.

    python benchmarks/profile/flexglue_profile.py --preset int8 --attention sage [--engine eager] [--top 15]

Used to see where the time goes when ``optimize.attention=sage*`` is slower than the default SDPA (H20: SageAttention's
sm_89 kernel runs on Hopper; the quantize-Q/K/V kernels are extra byte movement). Kernels are grouped by what they
compute: attention (flash / cudnn SDPA / Sage's qk-int8 kernels), the Sage quantizers, GEMM, convolution, everything else.
"""

from __future__ import annotations

import argparse
from collections import defaultdict

import torch
from torch.profiler import ProfilerActivity, profile

import flashim
from flashim.models.romav2.model import load_image
from flashim.paths import IMAGE_A, IMAGE_B

GROUPS = (
    ("attention", ("attn", "flash", "sdpa", "fmha", "cudnn_sdpa", "qk_int8", "sageattn")),
    ("sage quantize", ("quant_per", "per_block", "per_warp", "per_thread", "transpose_pad", "mean_", "sub_mean",
                       "_quant", "quantize")),
    ("gemm", ("gemm", "cutlass", "cublas", "nvjet", "scaled_mm", "sm80_xmma", "sm90_xmma", "int8_linear")),
    ("conv", ("conv", "cudnn", "implicit_convolve", "xmma_fprop")),
)


def group_of(name: str) -> str:
    lowered = name.lower()
    for group, needles in GROUPS:
        if any(n in lowered for n in needles):
            return group
    return "other"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--preset", default="int8")
    parser.add_argument("--engine", default="eager")
    parser.add_argument("--attention", default=None)
    parser.add_argument("--top-k", type=int, default=2048)
    parser.add_argument("--top", type=int, default=15)
    parser.add_argument("--iterations", type=int, default=10)
    args = parser.parse_args()

    rdd = flashim.load("rdd", "fp8")
    kpts = [rdd.extract(str(p), top_k=args.top_k).keypoints.reshape(-1, 2) for p in (IMAGE_A, IMAGE_B)]
    del rdd
    overrides = {"optimize.attention": args.attention} if args.attention else {}
    model = flashim.load("flexglue", args.preset, engine=args.engine, overrides=overrides)

    a, b = (load_image(str(p), model.device) for p in (IMAGE_A, IMAGE_B))  # decoded once: only the model's kernels

    def run():
        return model.match_keypoints(a, kpts[0], b, kpts[1])

    for _ in range(10):
        run()
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA, ProfilerActivity.CPU]) as prof:
        for _ in range(args.iterations):
            run()
        torch.cuda.synchronize()

    per_kernel = defaultdict(lambda: [0.0, 0])
    for event in prof.events():
        # ``flashim::<op>[<impl>]`` ranges (op annotations) span the kernels they launch: counting them would double
        # every kernel inside them, so only the kernels themselves are summed.
        if event.device_type == torch.autograd.DeviceType.CUDA and not event.name.startswith("flashim::"):
            per_kernel[event.name][0] += event.device_time
            per_kernel[event.name][1] += 1
    total = sum(t for t, _ in per_kernel.values())
    groups = defaultdict(float)
    for name, (t, _) in per_kernel.items():
        groups[group_of(name)] += t
    n = args.iterations
    print(f"GPU {torch.cuda.get_device_name()}  preset={model.preset} engine={args.engine} attention={args.attention}")
    print(f"CUDA kernel time per call: {total / n / 1000:.2f} ms")
    for group, t in sorted(groups.items(), key=lambda kv: -kv[1]):
        print(f"  {group:14s} {t / n / 1000:7.2f} ms  {100 * t / total:5.1f}%")
    print(f"top {args.top} kernels (per call):")
    for name, (t, count) in sorted(per_kernel.items(), key=lambda kv: -kv[1][0])[: args.top]:
        print(f"  {t / n / 1000:7.3f} ms  x{count // n:<3d} {name[:110]}")


if __name__ == "__main__":
    main()
