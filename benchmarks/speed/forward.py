"""Preloaded-forward latency of RoMaV2 (FlashIM, or the RoMaV2 reference for comparison).

Method (RoMaV2 P8 benchmark): one fixed MegaDepth pair, preprocessed once; B=1;
``warmups`` untimed forwards (including compilation), then ``iterations`` forwards
timed with CUDA events. Run each configuration in its own process.

    python benchmarks/speed/forward.py --impl flashim --preset fp8 --engine compile
    python benchmarks/speed/forward.py --impl romav2 --preset fp8 --engine compile
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import torch

from flashim.paths import IMAGE_A, IMAGE_B, ORACLES, ROMAV2_CALIBRATION

_ROMAV2_VARIANTS = {"fp8-p8": "none", "fp8-s4": "s4_fused", "fp8-s42": "s4_fused_s2_fused"}


def _build_flashim(args):
    from flashim.models.romav2.build import load_romav2

    model = load_romav2(args.preset, setting=args.setting, engine=args.engine, attention=args.attention)
    inputs = model.prepare(args.image_a, args.image_b)
    return model, inputs, (model.quant_report.summary() if model.quant_report else None)


def _build_romav2(args):
    """RoMaV2's own FastFP8 runtime (reference numbers, same process conditions)."""
    sys.path.insert(0, str(ORACLES / "romav2"))  # the vendored reference package
    from romav2 import RoMaV2
    from romav2.fast_fp8 import FastFP8Model
    from romav2.full_fp8_runtime import FullFP8Calibration
    from romav2.graph_compile import compile_graph_regions_for_inference
    from romav2.staged_fp8 import apply_refiner_variant

    model = RoMaV2(RoMaV2.Cfg(compile=False)).eval()
    model.apply_setting(args.setting)
    runner, target = model, model
    if args.preset != "bf16":
        cal = FullFP8Calibration.load(ROMAV2_CALIBRATION)
        runner = FastFP8Model(model, cal, keep_bf16_modules=("refiners",))
        apply_refiner_variant(runner.model, cal, _ROMAV2_VARIANTS[args.preset])
        target = runner.model
        del model
    if args.engine == "compile":
        import torch._inductor.config as inductor_config

        inductor_config.triton.cudagraphs = False
        compile_graph_regions_for_inference(target, dynamic=False, mode="max-autotune-no-cudagraphs")
    from flashim.models.romav2.config import SETTINGS
    from flashim.models.romav2.model import prepare_pair

    # FlashIM preprocessing is identical to RoMaV2.match's.
    a_lr, b_lr, a_hr, b_hr = prepare_pair(
        args.image_a, args.image_b, SETTINGS[args.setting], torch.device("cuda")
    )

    def call(*_):
        return runner(a_lr, b_lr, img_A_hr=a_hr, img_B_hr=b_hr)

    return call, (a_lr, b_lr, a_hr, b_hr), None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--impl", choices=("flashim", "romav2"), default="flashim")
    parser.add_argument("--preset", default="fp8", choices=("bf16", "int8", "fp8", "fp8-p8", "fp8-s4", "fp8-s42"))
    parser.add_argument("--engine", default="compile", choices=("eager", "compile", "compile-full"))
    parser.add_argument("--setting", default="mega1500")
    parser.add_argument("--image-a", default=str(IMAGE_A))
    parser.add_argument("--image-b", default=str(IMAGE_B))
    parser.add_argument("--warmups", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--attention", default="sdpa", choices=("sdpa", "sage", "sage-fp8"))
    parser.add_argument("--out", default="")
    args = parser.parse_args()

    torch.manual_seed(0)
    torch.set_float32_matmul_precision("highest")
    build = _build_flashim if args.impl == "flashim" else _build_romav2
    model, inputs, packing = build(args)
    static_bytes = torch.cuda.memory_allocated()

    with torch.inference_mode():
        started = time.perf_counter()
        for _ in range(args.warmups):
            model(*inputs)
        torch.cuda.synchronize()
        warmup_s = time.perf_counter() - started
        torch.cuda.reset_peak_memory_stats()
        times = []
        for _ in range(args.iterations):
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            start.record()
            model(*inputs)
            end.record()
            end.synchronize()
            times.append(start.elapsed_time(end))

    times.sort()
    result = {
        "impl": args.impl,
        "preset": getattr(model, "preset", args.preset),  # FP8 presets are replaced on a GPU without FP8
        "engine": args.engine,
        "setting": args.setting,
        "gpu": torch.cuda.get_device_name(),
        "torch": torch.__version__,
        "warmups": args.warmups,
        "iterations": args.iterations,
        "median_ms": statistics.median(times),
        "p90_ms": times[int(0.9 * (len(times) - 1))],
        "min_ms": times[0],
        "warmup_s": warmup_s,
        "static_memory_bytes": static_bytes,
        "peak_memory_bytes": torch.cuda.max_memory_allocated(),
        "packing": packing,
    }
    print(json.dumps(result, indent=2))
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
