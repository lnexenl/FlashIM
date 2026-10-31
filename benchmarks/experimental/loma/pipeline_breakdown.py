"""Where a LoMa pair's time goes: CUDA-event time per module (forward hooks, summed over calls) of ``extract`` and
``match_features``, the rest of each call as "other" (resizes, keypoint sampling, matching / filtering glue), and the
GPU's busy fraction (kernel time / wall, torch.profiler) to separate kernel time from host gaps. Eager, one MegaDepth
pair, 1024 px, 4096 keypoints; median over 20 calls.

    PYTHONPATH=$PWD python benchmarks/experimental/loma/pipeline_breakdown.py --variant b --config int8+dynvit
"""
import argparse
import statistics
import sys
from collections import defaultdict
from pathlib import Path

import torch
from torch.profiler import ProfilerActivity, profile

sys.path.insert(0, str(Path(__file__).resolve().parent))
from vit_dynamic_int8 import quantize_vit

import flashim

IMAGES = ("assets/megadepth_0015/101776896_21fc24b031_o.jpg", "assets/megadepth_0015/101795964_72e8f63ab1_o.jpg")


class Timer:
    """Per-module CUDA-event intervals of the current call (outermost entry only per module)."""

    def __init__(self, modules: dict[str, torch.nn.Module]) -> None:
        self.marks: dict[str, list] = defaultdict(list)
        for name, module in modules.items():
            module.register_forward_pre_hook(lambda m, a, n=name: self._mark(n, "s"))
            module.register_forward_hook(lambda m, a, r, n=name: self._mark(n, "e"))

    def _mark(self, name: str, kind: str) -> None:
        event = torch.cuda.Event(enable_timing=True)
        event.record()
        self.marks[name].append((kind, event))

    def collect(self) -> dict[str, float]:
        out = {}
        for name, marks in self.marks.items():
            starts = [e for k, e in marks if k == "s"]
            ends = [e for k, e in marks if k == "e"]
            out[name] = sum(s.elapsed_time(e) for s, e in zip(starts, ends))
        self.marks.clear()
        return out


def timed_calls(fn, timer: Timer, iterations: int = 20) -> dict[str, float]:
    for _ in range(5):
        fn()
    torch.cuda.synchronize()
    timer.collect()
    rows = defaultdict(list)
    for _ in range(iterations):
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        parts = timer.collect()
        rows["total"].append(start.elapsed_time(end))
        for name, ms in parts.items():
            rows[name].append(ms)
    return {name: statistics.median(v) for name, v in rows.items()}


def busy(fn) -> tuple[float, float]:
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(5):
            fn()
        end.record()
        torch.cuda.synchronize()
    kernels = sum(e.device_time_total for e in prof.events() if e.device_type.name == "CUDA") / 5000
    return kernels, start.elapsed_time(end) / 5


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", default="b")
    parser.add_argument("--config", default="int8+dynvit", help="bf16 | int8 | int8+dynvit | fp8")
    args = parser.parse_args()
    preset = "int8" if args.config.startswith("int8") else args.config
    with torch.inference_mode():
        pipeline = flashim.load(f"loma-{args.variant}", preset)
        if args.config == "int8+dynvit":
            quantize_vit(pipeline.extractor)
        ext, matcher = pipeline.extractor, pipeline.matcher
        parts = {"detector.encoder": ext.detector.encoder, "detector.decoder": ext.detector.decoder,
                 "descriptor.vgg": ext.descriptor.encoder.vgg, "descriptor.vit": ext.descriptor.encoder.dinov2,
                 "descriptor.decoder": ext.descriptor.decoder}
        timer = Timer(parts)
        extract = timed_calls(lambda: ext.extract(IMAGES[0], resize=1024, top_k=4096), timer)
        a, b = (ext.extract(image, resize=1024, top_k=4096) for image in IMAGES)
        modules = {"input_proj": matcher.input_proj, "posenc": matcher.posenc,
                   "log_assignment": matcher.log_assignment, "token_confidence": matcher.token_confidence}
        for i, layer in enumerate(matcher.transformers):
            modules[f"self_attn.{i}"] = layer.self_attn
            modules[f"cross_attn.{i}"] = layer.cross_attn
        mtimer = Timer(modules)
        match = timed_calls(lambda: matcher.match_features(a, b), mtimer)
        ek, ew = busy(lambda: ext.extract(IMAGES[0], resize=1024, top_k=4096))
        mk, mw = busy(lambda: matcher.match_features(a, b))

    def show(title: str, rows: dict[str, float], kernels: float, wall: float) -> None:
        total = rows.pop("total")
        merged = defaultdict(float)
        for name, ms in rows.items():
            merged[name.split(".")[0] if name.startswith(("self_attn", "cross_attn")) else name] += ms
        print(f"{title}: {total:.2f} ms  (GPU kernels {kernels:.2f} / wall {wall:.2f} under the profiler: "
              f"{kernels / wall:.0%} busy)")
        for name, ms in sorted(merged.items(), key=lambda kv: -kv[1]):
            print(f"  {name:22s} {ms:6.2f}  {ms / total:5.1%}")
        other = total - sum(merged.values())
        print(f"  {'other':22s} {other:6.2f}  {other / total:5.1%}")

    print(f"## loma-{args.variant} {args.config}")
    show("extract", extract, ek, ew)
    show("match_features", match, mk, mw)


if __name__ == "__main__":
    main()
