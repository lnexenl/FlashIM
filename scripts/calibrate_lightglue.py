"""Calibrate static activation scales for LightGlue's transformer linears (MXFP8 or INT8).

Runs RDD (``--rdd-preset``, default ``fp8``: the features LightGlue sees in deployment) on the images of manifest
pairs, then LightGlue in BF16 with observers on every linear its FP8/INT8 presets target.

    python scripts/calibrate_lightglue.py --format fp8 \\
        --manifest data/quantization/calibration_local.jsonl \\
        --root . --max-pairs 32 --out flashim/models/lightglue/calibration/lightglue_fp8_mega.safetensors

``--pipeline`` calibrates the matcher of a LightGlue-style pipeline on its own extractor instead (``--rdd-preset``:
the extractor's preset, default ``bf16``), e.g. LoMa's (``--out`` defaults to the matcher spec's table):

    python scripts/calibrate_lightglue.py --format int8 --pipeline loma-g \\
        --manifest data/quantization/calibration_local.jsonl --root . --max-pairs 32 --resize 1024 --top-k 4096

A matcher whose ``int8`` preset runs ``rotate_residual_stream`` is calibrated with the rotation folded in, and its
``ROTATED_INT8_SMOOTHING`` inputs get SmoothQuant factors (one more pass over the pairs, first).
"""

from __future__ import annotations

import argparse
import hashlib
import json
from fnmatch import fnmatchcase
from pathlib import Path

import torch

import flashim
from flashim.build import get_spec
from flashim.models.lightglue.quant import PRESETS, ROTATED_INT8_SMOOTHING
from flashim.quant.calibration import Calibrator, Int8Calibrator, SmoothingCalibrator


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--format", choices=("fp8", "int8"), required=True)
    parser.add_argument("--manifest", required=True, help="JSONL (or JSON list) with image_a / image_b")
    parser.add_argument("--root", default=".", help="base directory for relative image paths")
    parser.add_argument("--max-pairs", type=int, default=32)
    parser.add_argument("--resize", type=int, default=1600)
    parser.add_argument("--top-k", type=int, default=4096)
    parser.add_argument("--rdd-preset", default=None, help="extractor preset (default: fp8 for rdd, fp32 otherwise)")
    parser.add_argument("--extractor", default="rdd", help="rdd | aliked | raco-aliked | superpoint | disk | xfeat (LightGlue variant = extractor)")
    parser.add_argument("--percentile", type=float, default=0.9999, help="INT8 observer quantile of |x|")
    parser.add_argument("--pipeline", default=None, help="calibrate this pipeline's matcher (e.g. loma-b) instead")
    parser.add_argument("--checkpoint")
    parser.add_argument("--out", default=None, help="required without --pipeline")
    args = parser.parse_args()

    text = Path(args.manifest).read_text()
    try:
        raw = json.loads(text)
        pairs = raw["pairs"] if isinstance(raw, dict) else raw
    except json.JSONDecodeError:
        pairs = [json.loads(line) for line in text.splitlines() if line.strip()]
    pairs = pairs[: args.max_pairs] if args.max_pairs else pairs
    if not pairs:
        raise ValueError("empty calibration manifest")

    rotated = False
    if args.pipeline:
        args.rdd_preset = args.rdd_preset or "bf16"
        overrides = [f"components.extractor.preset={args.rdd_preset}"]
        if args.checkpoint:
            overrides.append(f"components.matcher.checkpoint={args.checkpoint}")
        spec = get_spec(flashim.resolve(args.pipeline, "bf16").components["matcher"].model)
        passes = spec.presets.get(args.format, {}).get("optimize", {}).get("passes", ())
        rotated = "rotate_residual_stream" in passes
        if rotated:  # observe the inputs the quantized layers will see
            overrides.append(f"components.matcher.optimize.passes={json.dumps(list(passes))}")
        pipeline = flashim.load(args.pipeline, "bf16", overrides=overrides)
        rdd, model = pipeline.extractor, pipeline.matcher
        matcher = pipeline.run_config.components["matcher"]
        config = spec.quant_configs[args.format]
        args.out = args.out or str(spec.calibrations[f"{matcher.variant}/{args.format}"])
        args.extractor = pipeline.run_config.components["extractor"].model
    else:
        if not args.out:
            parser.error("--out is required without --pipeline")
        rdd = flashim.load(args.extractor, preset=args.rdd_preset or ("fp8" if args.extractor == "rdd" else "fp32"))
        model = flashim.load(
            "lightglue", "bf16", config={"variant": args.extractor, "checkpoint": args.checkpoint}
        )
        config = PRESETS[args.format].config
    targets = [
        name for name, module in model.named_modules() if isinstance(module, torch.nn.Linear) and config.scheme_for(name)
    ]
    root = Path(args.root).expanduser()

    def run(observer, label: str) -> None:
        with torch.inference_mode(), observer:
            for index, item in enumerate(pairs, start=1):
                a, b = (rdd.extract(root / item[key], resize=args.resize, top_k=args.top_k) for key in ("image_a", "image_b"))
                model.match_features(a, b)
                print(f"{label} {index}/{len(pairs)}", flush=True)

    smooth = None
    if rotated and args.format == "int8":
        alphas = {name: alpha for name in targets for glob, alpha in ROTATED_INT8_SMOOTHING.items() if fnmatchcase(name, glob)}
        smoothing = SmoothingCalibrator(model, alphas)
        run(smoothing, "smoothing")
        smooth = smoothing.factors()
    observer = (
        Int8Calibrator(model, targets, percentile=args.percentile, smooth=smooth) if args.format == "int8"
        else Calibrator(model, targets)
    )
    run(observer, "calibrated")
    table = observer.table(
        {
            "model": args.pipeline or "lightglue",
            "extractor": args.extractor,
            "rdd_preset": str(args.rdd_preset),
            "resize": str(args.resize),
            "pairs": str(len(pairs)),
            "manifest": Path(args.manifest).name,
            "manifest_sha256": hashlib.sha256(Path(args.manifest).read_bytes()).hexdigest(),
            "top_k": str(args.top_k),
            **({"passes": "rotate_residual_stream", "smoothing": json.dumps(ROTATED_INT8_SMOOTHING)} if smooth else {}),
        }
    )
    tmp = Path(args.out).with_suffix(".tmp.safetensors")
    table.save(tmp)
    tmp.replace(args.out)
    print(json.dumps({"out": args.out, "pairs": len(pairs), "layers": len(targets)}))


if __name__ == "__main__":
    main()
