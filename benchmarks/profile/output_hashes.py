"""Output fingerprints of every model / preset, for "is this change bit-identical?" checks across two source trees.

Each configuration runs on the fixed MegaDepth pair (eager): sparse extractors ``extract`` image A and ``match_features``
A/B, pipelines and pair-level models ``match`` A/B, RoMaV2 ``match`` with 5000 seeded samples. For each, the sha256 of
every output tensor, the tensors themselves (``<out>/<config>.pt``, for tolerance diffs when they are not identical)
and ``ops.call_counts()`` (which impls ran) are written. Run it once per tree and compare:

    PYTHONPATH=/path/to/base-worktree python benchmarks/profile/output_hashes.py --out results/hashes/base
    python benchmarks/profile/output_hashes.py --out results/hashes/new
    python benchmarks/profile/output_hashes.py --compare results/hashes/base results/hashes/new

Triton autotuning can pick other tile configs in another process, which changes the accumulation order of
floating-point kernels: run the base tree twice to see which configurations are not reproducible to begin with.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from fnmatch import fnmatchcase
from pathlib import Path

import torch

from flashim.paths import MEGADEPTH_IMAGES

IMAGES = MEGADEPTH_IMAGES
IMAGE_A, IMAGE_B = IMAGES / "3383162411_67cc9af85e_o.jpg", IMAGES / "299056923_ecd980298c_o.jpg"
EXTRACTORS = ("superpoint", "alike", "disk", "xfeat", "ripe", "dedode", "dedode-g", "dad")


def configs() -> list[tuple[str, str, str, list[str], str]]:
    """``(name, model, preset, overrides, kind)``; kind is ``extract`` / ``pipeline`` / ``pair`` / ``dense``."""
    out = [(f"{m}-{p}", m, p, [], "extract") for m in EXTRACTORS for p in ("bf16", "int8", "fp8")]
    out += [(f"{m}-lightglue-{p}", f"{m}-lightglue", p, [f"components.extractor.preset={p}"], "pipeline")
            for m in ("xfeat", "superpoint", "disk") for p in ("int8", "fp8")]
    out += [(f"loftr-{p}", "loftr", p, [], "pair") for p in ("bf16", "int8", "fp8")]
    out += [(f"rdd-lightglue-{p}", "rdd-lightglue", p, [], "pipeline") for p in ("fp8", "int8")]
    out += [("aliked-lightglue-int8", "aliked-lightglue", "int8", [], "pipeline")]
    out += [("raco-aliked-lightglue-bf16x", "raco-aliked-lightglue", "int8", ["components.extractor.preset=bf16"],
             "pipeline")]
    out += [(f"romav2-{p}", "romav2", p, [], "dense") for p in ("bf16", "fp8", "int8")]
    return out


def tensors_of(obj, prefix: str) -> dict[str, torch.Tensor]:
    if isinstance(obj, torch.Tensor):
        return {prefix: obj.detach().cpu()}
    fields = obj.as_dict() if hasattr(obj, "as_dict") else getattr(obj, "__dict__", {})
    return {f"{prefix}.{k}": v.detach().cpu() for k, v in fields.items() if isinstance(v, torch.Tensor)}


def run(name: str, model_name: str, preset: str, overrides: list[str], kind: str) -> dict[str, torch.Tensor]:
    import flashim
    from flashim.models.romav2.model import load_image

    model = flashim.load(model_name, preset, overrides=["engine.kind=eager", *overrides])
    a, b = load_image(str(IMAGE_A), model.device), load_image(str(IMAGE_B), model.device)
    out: dict[str, torch.Tensor] = {}
    with torch.inference_mode():
        if kind == "extract":
            fa = model.extract(a, resize=1200, top_k=4096)
            fb = model.extract(b, resize=1200, top_k=4096)
            out |= tensors_of(fa, "A") | tensors_of(fb, "B") | tensors_of(model.match_features(fa, fb), "M")
        elif kind == "pipeline":
            out |= tensors_of(model.match(a, b, resize=1200, top_k=4096), "M")
        elif kind == "pair":
            out |= tensors_of(model.match(a, b, resize=1200), "M")
        else:
            generator = torch.Generator(device=model.device).manual_seed(0)
            out |= tensors_of(model.match(a, b, num_samples=5000, generator=generator), "M")
    return out


def digest(t: torch.Tensor) -> str:
    t = t.contiguous()
    raw = t.view(torch.uint8) if t.dtype != torch.bool else t.to(torch.uint8)
    return hashlib.sha256(raw.numpy().tobytes()).hexdigest()[:16] + f":{tuple(t.shape)}:{str(t.dtype)[6:]}"


def record(out_dir: Path, wanted: list[str]) -> None:
    from flashim import ops

    out_dir.mkdir(parents=True, exist_ok=True)
    for name, model_name, preset, overrides, kind in configs():
        if wanted and not any(fnmatchcase(name, w) for w in wanted):
            continue
        ops.reset_call_counts()
        try:
            tensors = run(name, model_name, preset, overrides, kind)
        except Exception as exc:  # noqa: BLE001 -- keep going: a broken config is itself a finding
            print(f"{name}: FAILED {type(exc).__name__}: {exc}", flush=True)
            (out_dir / f"{name}.json").write_text(json.dumps({"error": f"{type(exc).__name__}: {exc}"}) + "\n")
            continue
        torch.save(tensors, out_dir / f"{name}.pt")
        report = {"hashes": {k: digest(v) for k, v in tensors.items()}, "calls": ops.call_counts()}
        (out_dir / f"{name}.json").write_text(json.dumps(report, indent=1, sort_keys=True) + "\n")
        print(f"{name}: {len(tensors)} tensors", flush=True)
        torch.cuda.empty_cache()


def compare(base: Path, new: Path) -> None:
    for path in sorted(base.glob("*.json")):
        other = new / path.name
        if not other.exists():
            print(f"{path.stem:32s} missing in {new}")
            continue
        a, b = json.loads(path.read_text()), json.loads(other.read_text())
        if "error" in a or "error" in b:
            print(f"{path.stem:32s} error: {a.get('error') or b.get('error')}")
            continue
        differ = sorted(k for k in a["hashes"].keys() | b["hashes"].keys() if a["hashes"].get(k) != b["hashes"].get(k))
        calls = "" if a["calls"] == b["calls"] else f"  calls {a['calls']} -> {b['calls']}"
        if not differ:
            print(f"{path.stem:32s} identical{calls}")
            continue
        ta, tb = torch.load(path.with_suffix(".pt")), torch.load(other.with_suffix(".pt"))
        notes = []
        for k in differ:
            x, y = ta.get(k), tb.get(k)
            if x is None or y is None or x.shape != y.shape:
                notes.append(f"{k} shape {None if x is None else tuple(x.shape)} -> {None if y is None else tuple(y.shape)}")
            elif x.is_floating_point():
                diff = (x.float() - y.float()).abs()
                notes.append(f"{k} max {diff.max().item():.3g} frac {(diff > 0).float().mean().item():.3g}")
            else:
                notes.append(f"{k} {(x != y).float().mean().item():.3g} differ")
        print(f"{path.stem:32s} DIFFERENT: " + "; ".join(notes) + calls)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--configs", nargs="*", default=[], help="fnmatch patterns over config names")
    parser.add_argument("--compare", nargs=2, type=Path, metavar=("BASE", "NEW"))
    parser.add_argument("--list", action="store_true")
    args = parser.parse_args()
    if args.list:
        print("\n".join(c[0] for c in configs()))
    elif args.compare:
        compare(*args.compare)
    else:
        record(args.out, args.configs)


if __name__ == "__main__":
    main()
