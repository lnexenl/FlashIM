"""Per-module audit of one model across precisions: what is quantized, with which kernel, and whether it pays off.

For every preset (``fp32`` / ``bf16`` / ``int8`` / ``fp8``) the model's stage (``extract`` of the fixed MegaDepth image at
the unified setting; ``match_features`` for a LightGlue pairing's matcher; ``match`` for pair-level models) runs with a
CUDA-synchronized timer around every *unit*: a quantized module of :mod:`flashim.quant` (its children are not timed
separately) or a leaf module. The report gives, per unit, its class, kernel, input shape and time in each preset; for
every Conv2d / Linear of the BF16 model whether the quantized presets replaced it and, if not, why (protected, ignored,
not targeted, grouped); the time outside any module ("glue": functional ops between modules); and the op impls that ran
(reference fallbacks on the GPU are kernels we do not have).

Synchronizing around every module removes launch overlap, so the totals are a little above the untimed latency: compare
units within one report, not with the speed benchmarks.

    python benchmarks/profile/quant_audit.py --model dedode --out results/profile/quant_audit
    python benchmarks/profile/quant_audit.py --model superpoint-lightglue --part matcher --out results/profile/quant_audit
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from collections import defaultdict
from fnmatch import fnmatchcase
from pathlib import Path

import torch
from torch import nn

import flashim
from flashim import ops
from flashim.build import get_spec
from flashim.models.romav2.model import load_image

IMAGES = Path.home() / "work/RoMaV2/data/megadepth/Undistorted_SfM/0015/images"
IMAGE_A, IMAGE_B = IMAGES / "3383162411_67cc9af85e_o.jpg", IMAGES / "299056923_ecd980298c_o.jpg"
PRESETS = ("fp32", "bf16", "int8", "fp8")


def is_quant(module: nn.Module) -> bool:
    return type(module).__module__.startswith("flashim.quant")


def units(model: nn.Module) -> dict[str, nn.Module]:
    """Quantized modules (without their descendants) and leaf modules, by name."""
    found: dict[str, nn.Module] = {}
    skip: tuple[str, ...] = ()
    for name, module in model.named_modules():
        if not name or name.startswith(skip):
            continue
        if is_quant(module):
            found[name] = module
            skip = (*skip, name + ".")
        elif not any(True for _ in module.children()):
            found[name] = module
    return found


def shape_of(args: tuple) -> str:
    for arg in args:
        shape = getattr(arg, "logical_shape", None) or getattr(arg, "shape", None)
        if shape is not None:
            return "x".join(str(int(s)) for s in shape)
    return "?"


def kernel_of(module: nn.Module) -> str:
    name = getattr(module, "kernel_name", None)
    if name:
        return str(name)
    if is_quant(module):
        inner = sorted({str(k) for m in module.modules() if (k := getattr(m, "kernel_name", None))})
        return "+".join(inner) if inner else type(module).__name__
    weight = getattr(module, "weight", None)
    return str(weight.dtype).replace("torch.", "") if isinstance(weight, torch.Tensor) else ""


def measure(model: nn.Module, target: nn.Module, run, repeats: int, parents: tuple[str, ...] = ()) -> dict:
    """Median per-unit time of ``run()`` over ``repeats`` calls, with the ops that ran. ``parents``: names of non-leaf
    modules to time as a whole as well (what a fused quantized block replaces); reported apart from the units."""
    found = units(target)
    named = dict(target.named_modules())
    extra = {name: named[name] for name in parents if name in named and name not in found}
    current: dict[str, float] = defaultdict(float)
    calls: dict[str, int] = defaultdict(int)
    shapes: dict[str, str] = {}
    started: dict[str, float] = {}
    handles = []

    def pre(name):
        def hook(module, args):
            torch.cuda.synchronize()
            shapes.setdefault(name, shape_of(args))
            started[name] = time.perf_counter()
        return hook

    def post(name):
        def hook(module, args, output):
            torch.cuda.synchronize()
            current[name] += (time.perf_counter() - started[name]) * 1e3
            calls[name] += 1
        return hook

    for name, module in (found | extra).items():
        handles += [module.register_forward_pre_hook(pre(name)), module.register_forward_hook(post(name))]
    with torch.inference_mode():
        for _ in range(3):
            run()
        runs, totals = [], []
        ops.reset_call_counts()
        for _ in range(repeats):
            current.clear()
            calls.clear()
            torch.cuda.synchronize()
            start = time.perf_counter()
            run()
            torch.cuda.synchronize()
            totals.append((time.perf_counter() - start) * 1e3)
            runs.append((dict(current), dict(calls)))
        counts = {op: dict(impls) for op, impls in ops.call_counts().items() if impls}
    for handle in handles:
        handle.remove()
    rows = {}
    for name, module in found.items():
        times = [r[0].get(name, 0.0) for r in runs]
        if not any(times):
            continue
        rows[name] = {
            "cls": type(module).__name__, "quant": is_quant(module), "kernel": kernel_of(module),
            "shape": shapes.get(name, "?"), "calls": runs[0][1].get(name, 0), "ms": round(statistics.median(times), 4),
            "groups": getattr(module, "groups", None),
        }
    total = statistics.median(totals)
    return {
        "total_ms": round(total, 3), "module_ms": round(sum(r["ms"] for r in rows.values()), 3),
        "parents": {name: round(statistics.median([r[0].get(name, 0.0) for r in runs]), 4) for name in extra},
        "ops": {op: {impl: n // repeats for impl, n in impls.items()} for op, impls in counts.items()}, "units": rows,
    }


def status(name: str, module: nn.Module, quant_units: dict, spec, preset: str) -> str:
    """Why a BF16 Conv2d / Linear is (not) quantized in ``preset``."""
    if any(name == u or name.startswith(u + ".") for u, row in quant_units.items() if row["quant"]):
        return "quantized"
    config = spec.quant_configs.get(preset)
    if any(fnmatchcase(name, p) for p in spec.protected):
        return "protected"
    if config is not None and any(fnmatchcase(name, p) for p in config.ignore):
        return "ignored"
    if isinstance(module, nn.Conv2d) and module.groups != 1:
        return "grouped conv (no quantized kernel)"
    if config is not None and config.scheme_for(name) is not None:
        return "targeted but not converted"
    return "not targeted"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True)
    parser.add_argument("--part", choices=("model", "matcher"), default="model", help="matcher: a pipeline's LightGlue")
    parser.add_argument("--resize", type=int, default=1200)
    parser.add_argument("--top-k", type=int, default=4096)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--out", default="")
    args = parser.parse_args()

    spec = get_spec("lightglue" if args.part == "matcher" else args.model)
    report: dict = {"model": args.model, "part": args.part, "presets": {}, "layers": {}}
    reference_modules: dict[str, nn.Module] = {}
    fused: set[str] = set()  # quantized units that replace a whole (non-leaf) module of the float model
    for preset in ("int8", "fp8", "fp32", "bf16"):  # quantized first: they say which parents to time in the float presets
        overrides = ["engine.kind=eager"]
        if args.part == "matcher":
            overrides.append(f"components.extractor.preset={preset}")
        try:
            model = flashim.load(args.model, preset, overrides=overrides)
        except Exception as exc:  # noqa: BLE001 - a preset a model does not ship is reported, not fatal
            report["presets"][preset] = {"error": f"{type(exc).__name__}: {exc}"[:200]}
            continue
        a, b = load_image(str(IMAGE_A), model.device), load_image(str(IMAGE_B), model.device)
        if args.part == "matcher":
            with torch.inference_mode():
                fa, fb = (model.extract(im, resize=args.resize, top_k=args.top_k) for im in (a, b))
            target, stage = model.matcher, "match_features"
            run = lambda m=model, fa=fa, fb=fb: m.match_features(fa, fb)
        elif hasattr(model, "extract"):
            target, stage = model, "extract"
            run = lambda m=model, a=a: m.extract(a, resize=args.resize, top_k=args.top_k)
        else:
            target, stage = model, "match"
            top_k = args.top_k if args.top_k > 0 else None
            run = lambda m=model, a=a, b=b, k=top_k: m.match(a, b, resize=args.resize, top_k=k)
        result = measure(model, target, run, args.repeats, parents=tuple(sorted(fused)) if preset in ("fp32", "bf16") else ())
        fused |= {name for name, row in result["units"].items() if row["quant"]}
        result["stage"] = stage
        result["passes"] = {k: (v if isinstance(v, (str, int)) else len(v)) for k, v in model.build_report.passes.items()} \
            if getattr(model, "build_report", None) and args.part == "model" else {}
        report["presets"][preset] = result
        if preset == "bf16":
            reference_modules = {n: m for n, m in target.named_modules() if isinstance(m, (nn.Conv2d, nn.Linear))}
        del model
        torch.cuda.empty_cache()

    bf16 = report["presets"].get("bf16", {}).get("units", {})
    for name, module in reference_modules.items():
        row = {
            "cls": type(module).__name__, "dtype": str(module.weight.dtype).replace("torch.", ""),
            "groups": getattr(module, "groups", None), "bf16_ms": bf16.get(name, {}).get("ms", 0.0),
            "shape": bf16.get(name, {}).get("shape", "?"),
        }
        for preset in ("int8", "fp8"):
            if "units" in report["presets"].get(preset, {}):
                row[preset] = status(name, module, report["presets"][preset]["units"], spec, preset)
        report["layers"][name] = row
    tag = f"{args.model}{'-matcher' if args.part == 'matcher' else ''}"
    report["presets"] = {p: report["presets"][p] for p in PRESETS if p in report["presets"]}
    totals = "  ".join(f"{p} {r['total_ms']:.1f}" for p, r in report["presets"].items() if "total_ms" in r)
    print(f"{tag}: {totals} ms")
    if args.out:
        out = Path(args.out)
        out.mkdir(parents=True, exist_ok=True)
        (out / f"{tag}.json").write_text(json.dumps(report, indent=1) + "\n")


if __name__ == "__main__":
    main()
