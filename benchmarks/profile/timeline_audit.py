"""GPU timeline of one model stage (eager): idle gaps split into "after a host sync" vs "launch-bound", and runs of
consecutive memory-bound kernels between compute kernels (fusion candidates), each labelled with the aten / flashim ops
that launched it and their input shapes.

    python benchmarks/profile/timeline_audit.py --model dedode --preset int8 [--stage extract|match|pair]
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import torch
from torch.profiler import ProfilerActivity, profile

import flashim
from flashim.models.romav2.model import load_image

IMAGES = Path.home() / "work/RoMaV2/data/megadepth/Undistorted_SfM/0015/images"
IMAGE_A, IMAGE_B = IMAGES / "3383162411_67cc9af85e_o.jpg", IMAGES / "299056923_ecd980298c_o.jpg"
SYNCS = ("cudaStreamSynchronize", "cudaDeviceSynchronize", "cudaEventSynchronize", "cudaMemcpy")
# compute kernels end a fusion run; everything else (elementwise, copies, cat, resample, norm, pool, quantize) is
# memory-bound glue
COMPUTE = ("gemm", "conv", "fprop", "xmma", "cutlass", "flash_fwd", "attn", "nvjet", "_int8_gemm", "_fp8_gemm",
           "_int8_conv2d", "_mxfp8_conv2d", "main_kernel", "depthwise5x5", "dw5x5", "sgemm", "_mm", "dual_softmax",
           "_col_lse", "_row_", "resblock", "multiscale", "refiner")


def is_compute(name: str) -> bool:
    lowered = name.lower()
    return any(k in lowered for k in COMPUTE) and "quantize" not in lowered


def owner(ev, by_id) -> str:
    """Innermost aten / flashim op above the runtime call that launched a device event, with shapes."""
    runtime = by_id.get(ev.id)
    op = runtime.cpu_parent if runtime is not None else None
    chain = []
    while op is not None:
        if op.name.startswith(("aten::", "flashim::")):
            chain.append(op)
        op = op.cpu_parent
    if not chain:
        return "?" + ev.name.split("(")[0].replace("void ", "")[-60:]
    top = chain[-1]  # outermost aten op (e.g. aten::conv2d rather than aten::cudnn_convolution)
    shapes = [list(s) for s in (top.input_shapes or []) if s][:2]
    return f"{top.name}{shapes}"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--preset", default="int8")
    p.add_argument("--stage", default="extract")
    p.add_argument("--resize", type=int, default=1200)
    p.add_argument("--top-k", type=int, default=4096)
    p.add_argument("--min-run-us", type=float, default=150.0)
    p.add_argument("--overrides", nargs="*", default=[])
    a = p.parse_args()
    model = flashim.load(a.model, a.preset, overrides=["engine.kind=eager", *a.overrides])
    ia, ib = load_image(str(IMAGE_A), model.device), load_image(str(IMAGE_B), model.device)
    with torch.inference_mode():
        if a.stage == "extract":
            fn = lambda: model.extract(ia, resize=a.resize, top_k=a.top_k)
        elif a.stage == "match":
            fa, fb = (model.extract(i, resize=a.resize, top_k=a.top_k) for i in (ia, ib))
            fn = lambda: model.match_features(fa, fb)
        else:
            kw = {"resize": a.resize} | ({"top_k": a.top_k} if a.top_k > 0 else {})
            fn = lambda: model.match(ia, ib, **kw)
        for _ in range(4):
            fn()
        torch.cuda.synchronize()
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True) as prof:
            fn()
            torch.cuda.synchronize()
    events = prof.events()
    by_id = {e.id: e for e in events if e.device_type == torch.autograd.DeviceType.CPU and e.name.startswith("cu")}
    dev = sorted((e for e in events if e.device_type == torch.autograd.DeviceType.CUDA
                  and not e.name.startswith(("flashim::", "aten::"))), key=lambda e: e.time_range.start)  # no op ranges
    syncs = [e for e in events if e.device_type == torch.autograd.DeviceType.CPU and e.name in SYNCS]
    t0 = min(e.time_range.start for e in events if e.device_type == torch.autograd.DeviceType.CPU)
    t1 = max(e.time_range.end for e in events)
    busy = 0.0
    end = dev[0].time_range.start
    sync_gap, launch_gap = 0.0, 0.0
    sync_sites: dict[str, float] = defaultdict(float)
    for e in dev:
        s, f = e.time_range.start, e.time_range.end
        if s > end:
            gap = s - end
            hit = [y for y in syncs if end <= y.time_range.end <= s]
            if hit:
                sync_gap += gap
                parent = hit[-1].cpu_parent
                while parent is not None and parent.cpu_parent is not None and not parent.cpu_parent.name.startswith("<"):
                    parent = parent.cpu_parent
                sync_sites[parent.name if parent else "?"] += gap
            else:
                launch_gap += gap
        busy += max(0.0, f - max(s, end))
        end = max(end, f)
    wall = (t1 - t0) / 1e3
    kernels = [e for e in dev if not e.name.startswith("Mem")]
    print(f"## {a.model}-{a.preset} {a.stage}: wall(profiled) {wall:.2f} ms, GPU busy {busy / 1e3:.2f} ms, "
          f"{len(kernels)} kernels, {len(syncs)} sync calls; idle: after-sync {sync_gap / 1e3:.2f} ms, "
          f"launch-bound {launch_gap / 1e3:.2f} ms, before first kernel {(dev[0].time_range.start - t0) / 1e3:.2f} ms")
    for site, g in sorted(sync_sites.items(), key=lambda kv: -kv[1])[:6]:
        print(f"     sync gap {g / 1e3:6.3f} ms after {site}")
    # runs of memory-bound kernels between compute kernels
    runs, cur = [], []
    for e in kernels:
        if is_compute(e.name):
            if cur:
                runs.append(cur)
            cur = []
        else:
            cur.append(e)
    if cur:
        runs.append(cur)
    glue = sum(sum(k.time_range.elapsed_us() for k in r) for r in runs)
    print(f"   memory-bound glue: {glue / 1e3:.2f} ms in {sum(len(r) for r in runs)} kernels / {len(runs)} runs; "
          f"runs >= {a.min_run_us:.0f} us:")
    agg: dict[tuple, list] = defaultdict(lambda: [0.0, 0])
    for r in runs:
        us = sum(k.time_range.elapsed_us() for k in r)
        key = tuple(owner(k, by_id) for k in r)
        agg[key][0] += us
        agg[key][1] += 1
    for key, (us, n) in sorted(agg.items(), key=lambda kv: -kv[1][0]):
        if us < a.min_run_us:
            break
        print(f"   {us / 1e3:6.2f} ms x{n:<3d} " + " -> ".join(dict.fromkeys(key))[:400])


if __name__ == "__main__":
    main()
