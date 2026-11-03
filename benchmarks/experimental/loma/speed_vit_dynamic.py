"""Latency (``--engine eager | compile``) of the LoMa extractor (one MegaDepth image, 1024 px, 4096 keypoints) and the matcher, for bf16 / int8 /
int8 + per-token dynamic INT8 ViT (``vit_dynamic_int8.py``) / fp8; interleaved rounds, best of the per-round medians.

    PYTHONPATH=$PWD python benchmarks/experimental/loma/speed_vit_dynamic.py --variant b [--engine compile]
"""
import argparse
import json
import statistics
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from vit_dynamic_int8 import quantize_vit

import flashim

IMAGES = ("assets/megadepth_0015/101776896_21fc24b031_o.jpg", "assets/megadepth_0015/101795964_72e8f63ab1_o.jpg")


def timed(fn, warmups: int = 5, iterations: int = 20) -> float:
    for _ in range(warmups):
        fn()
    times = []
    for _ in range(iterations):
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))
    return statistics.median(times)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", default="b")
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--engine", default="eager")
    parser.add_argument("--config", action="append", default=None,
                        help="bf16 | int8 | int8+dynvit | int8+dynvit+sage-fp8 | int8+dynvit+msage (sage-fp8 in the matcher) | int8+dynvit+sage-fp8+msage (both) | fp8 (repeatable; default all). Compiled: one per process, "
                        "two compiled instances in one process share Dynamo graphs")
    parser.add_argument("--eager-conv", action="store_true",
                        help="run backends/triton/conv.py's impls outside the compiled graphs (a graph break per conv): "
                        "their autotuned kernels (prune_configs_by) fail under Dynamo when one graph launches one twice")
    args = parser.parse_args()
    configs = {}
    with torch.inference_mode():
        for name, preset in (("bf16", "bf16"), ("int8", "int8"), ("int8+dynvit", "int8"),
                             ("int8+dynvit+sage-fp8", "int8"), ("int8+dynvit+msage", "int8"),
                             ("int8+dynvit+sage-fp8+msage", "int8"), ("fp8", "fp8")):
            if args.config and name not in args.config:
                continue
            overrides = ["components.matcher.optimize.attention=sage-fp8"] if name.endswith("+msage") else []
            pipeline = flashim.load(f"loma-{args.variant}", preset, engine=args.engine, overrides=overrides)
            if name.startswith("int8+dynvit"):
                quantize_vit(pipeline.extractor)
            if "+sage-fp8" in name:
                for block in pipeline.extractor.descriptor.encoder.dinov2.blocks:
                    block.attn.sdpa_impl = "sage-fp8"
            configs[name] = pipeline
        if args.eager_conv:  # after loading: the backends register their impls on first use
            from flashim.ops._registry import OPS
            wrapped = [(op.name, impl.name) for op in OPS.values() for impl in op.impls
                       if impl.fn.__module__ == "flashim.backends.triton.conv"]
            for op in OPS.values():
                for impl in op.impls:
                    if impl.fn.__module__ == "flashim.backends.triton.conv":
                        impl.fn = torch.compiler.disable(impl.fn)
            print("eager:", wrapped, file=sys.stderr)
        result = {name: {"extract": [], "match": []} for name in configs}
        for _ in range(args.rounds):
            for name, pipeline in configs.items():
                extract = pipeline.extractor.extract
                a, b = (extract(image, resize=1024, top_k=4096) for image in IMAGES)
                result[name]["extract"].append(timed(lambda e=extract: e(IMAGES[0], resize=1024, top_k=4096)))
                result[name]["match"].append(timed(lambda m=pipeline.matcher, a=a, b=b: m.match_features(a, b)))
    base = None
    for name, times in result.items():
        extract, match = min(times["extract"]), min(times["match"])
        pair = 2 * extract + match
        base = base or pair
        print(json.dumps({  # vs_bf16: vs the first config of this run
"variant": args.variant, "engine": args.engine + ("/eager-conv" if args.eager_conv else ""), "config": name, "extract_ms": round(extract, 2),
                          "match_ms": round(match, 2), "pair_ms": round(pair, 1), "vs_bf16": round(base / pair, 2)}))


if __name__ == "__main__":
    main()
