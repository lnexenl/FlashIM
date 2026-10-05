"""Host <-> device synchronizations per model stage: every synchronizing CUDA call (``.item()``, ``nonzero``, blocking
H2D / D2H ``copy_`` such as ``new_tensor([...])`` on a CUDA tensor) caught with ``torch.cuda.set_sync_debug_mode``, with
the innermost flashim frames of its Python stack, plus the memcpy kinds/bytes the profiler sees and GPU-busy share.

    python benchmarks/profile/sync_audit.py --model dedode --preset int8
"""

from __future__ import annotations

import argparse
import time
import traceback
import warnings
from collections import Counter
from pathlib import Path

import torch
from torch.profiler import ProfilerActivity, profile

import flashim
from flashim.models.romav2.model import load_image

IMAGES = Path.home() / "work/RoMaV2/data/megadepth/Undistorted_SfM/0015/images"
IMAGE_A, IMAGE_B = IMAGES / "3383162411_67cc9af85e_o.jpg", IMAGES / "299056923_ecd980298c_o.jpg"
ROOT = str(Path(flashim.__file__).parent)


def site(stack: list[traceback.FrameSummary]) -> str:
    frames = [f for f in stack if f.filename.startswith(ROOT)]
    return " <- ".join(f"{f.filename[len(ROOT) + 1:]}:{f.lineno} {f.name}" for f in reversed(frames[-3:])) or "?"


def audit(name: str, fn, calls: int = 2) -> None:
    fn()
    torch.cuda.synchronize()
    hits: Counter[str] = Counter()

    def show(message, category, filename, lineno, file=None, line=None):
        if "synchronizing" in str(message):
            hits[site(traceback.extract_stack()[:-1])] += 1

    old = warnings.showwarning
    warnings.showwarning = show
    warnings.simplefilter("always")
    torch.cuda.set_sync_debug_mode("warn")
    try:
        for _ in range(calls):
            fn()
    finally:
        torch.cuda.set_sync_debug_mode("default")
        warnings.showwarning = old
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA, ProfilerActivity.CPU]) as prof:
        t = time.perf_counter()
        for _ in range(calls):
            fn()
        torch.cuda.synchronize()
        wall = (time.perf_counter() - t) * 1e3 / calls
    copies: Counter[str] = Counter()
    gpu = 0.0
    for e in prof.events():
        if e.device_type == torch.autograd.DeviceType.CUDA:
            gpu += e.device_time_total if hasattr(e, "device_time_total") else 0
            if e.name.startswith("Memcpy"):
                copies[e.name.split(" (")[0]] += 1
    syncs = sum(hits.values()) / calls
    print(f"\n## {name}: {syncs:.0f} syncs/call, memcpys/call { {k: v // calls for k, v in copies.items()} }, "
          f"wall {wall:.2f} ms")
    for where, n in hits.most_common():
        print(f"   {n / calls:5.1f}x  {where}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--preset", default="int8")
    p.add_argument("--resize", type=int, default=1200)
    p.add_argument("--top-k", type=int, default=4096)
    p.add_argument("--overrides", nargs="*", default=[])
    a = p.parse_args()
    model = flashim.load(a.model, a.preset, overrides=["engine.kind=eager", *a.overrides])
    ia, ib = load_image(str(IMAGE_A), model.device), load_image(str(IMAGE_B), model.device)
    tag = f"{a.model}-{a.preset}"
    with torch.inference_mode():
        if hasattr(model, "extract"):
            audit(f"{tag} extract", lambda: model.extract(ia, resize=a.resize, top_k=a.top_k))
            fa, fb = (model.extract(i, resize=a.resize, top_k=a.top_k) for i in (ia, ib))
            if hasattr(model, "match_features"):
                audit(f"{tag} match_features", lambda: model.match_features(fa, fb))
            if hasattr(model, "matcher"):  # pipeline: the whole pair
                audit(f"{tag} match (pipeline)", lambda: model.match(ia, ib, resize=a.resize, top_k=a.top_k))
        else:
            kw = {"resize": a.resize} | ({"top_k": a.top_k} if a.top_k > 0 else {})
            audit(f"{tag} match", lambda: model.match(ia, ib, **kw))


if __name__ == "__main__":
    main()
