"""Markdown latency tables from the results of benchmarks/speed/speed_table.sh and speed_table_models.sh.

Median ms per call, the fastest number of a row in bold, grouped by model family (dense and semi-dense, sparse
extractors, sparse matchers, pair time of sparse pipelines, FlexGlue).

    python benchmarks/speed/speed_table.py [results/speed/table]     # writes <dir>/table.md and prints it
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

PRESETS = ("fp32", "bf16", "int8", "fp8")
EXTRACTORS = ("superpoint", "disk", "alike", "xfeat", "dedode", "dedode-g", "dad", "ripe", "ripepp", "aliked",
              "raco-aliked", "rdd", "traqpoint", "sift", "orb")
# a model whose result is one `match` of the pair (no `extract`): dense and semi-dense matchers
PAIR_LEVEL = ("loftr", "roma", "minima-roma", "tiny-roma", "mast3r", "dust3r", "ufm", "eloftr", "xoftr",
              "minima-loftr")
# feature-level pipelines: pair time = 2 x extract + match_features
LIGHTGLUE = ("superpoint-lightglue", "disk-lightglue", "xfeat-lightglue", "aliked-lightglue",
             "raco-aliked-lightglue", "rdd-lightglue", "ripepp-lightglue")
OTHER_PIPELINES = ("loma-b128", "loma-b", "loma-r", "loma-l", "loma-g", "superpoint-superglue", "omniglue")
PIPELINES = LIGHTGLUE + OTHER_PIPELINES
MATCHERS = PIPELINES + ("rdd", "traqpoint", "orb", "sift")
ROMAV2_PRESETS = ("bf16", "int8", "fp8")
ROMAV2_ENGINES = ("eager", "compile", "compile-full")


def _fmt(value, best):
    if value is None:
        return "-"
    if isinstance(value, str):  # "fail" or a fallback marker such as "→ int8"
        return value
    text = f"{value:.1f}" if value >= 10 else f"{value:.2f}"
    return f"**{text}**" if value == best else text


def _cells(times: dict, keys) -> list[str]:
    """``eager / compile`` per key, the fastest number of the row in bold."""
    values = [v for v in times.values() if isinstance(v, float | int)]
    best = min(values) if values else None
    return [" / ".join(_fmt(times.get((key, engine)), best) for engine in ("eager", "compile"))
            if any(times.get((key, e)) is not None for e in ("eager", "compile")) else "" for key in keys]


def _fallback(requested: str, resolved: str) -> str | None:
    """``→ int8`` when no component built the preset asked for (``resolved`` is ``extractor+matcher`` for a pipeline)."""
    parts = resolved.split("+")
    return None if requested in parts else f"→ {parts[-1]}"


def _table(title: str, intro: str, header: str, models, data) -> list[str]:
    rows = [f"| {m} | " + " | ".join(_cells(data[m], PRESETS)) + " |" for m in models if m in data]
    if not rows:
        return []
    return ["", f"## {title}", "", intro, "", f"| {header} | " + " | ".join(PRESETS) + " |",
            "|---|" + "---|" * len(PRESETS), *rows]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("directory", nargs="?", default="results/speed/table", type=Path)
    args = parser.parse_args()
    d = args.directory

    extract, pair, matcher, pair_level, flex = (defaultdict(dict) for _ in range(5))
    mixed = {}  # pipelines whose components build different presets: (model, requested) -> "extractor+matcher"
    gpu = ""
    for name in ("run.log", "run_models.log"):  # crashed runs (results below override)
        log = d / name
        for line in log.read_text().splitlines() if log.exists() else ():
            if line.startswith("FAILED ") and not line.startswith("FAILED romav2"):
                model, preset, engine = line.removeprefix("FAILED ").split("|")
                if model in PAIR_LEVEL:
                    pair_level[model][(preset, engine)] = "fail"
                elif model == "flexglue":
                    flex[(preset, engine)] = "fail"
                else:
                    extract[model][(preset, engine)] = "fail"
                    if model in PIPELINES:
                        pair[model][(preset, engine)] = "fail"

    lines = []
    for name in ("sparse.jsonl", "models.jsonl"):
        path = d / name
        lines += path.read_text().splitlines() if path.exists() else []
    for line in lines:
        r = json.loads(line)
        model, preset, engine = r["tag"].split("|")
        gpu = r["gpu"]
        key = (preset, engine)
        if model == "flexglue":
            flex[key] = r
            continue
        mark = _fallback(preset, r["preset"])
        if model in PIPELINES and not mark and r["preset"] != f"{preset}+{preset}":
            mixed[(model, preset)] = r["preset"]
        if r.get("pair_level"):
            pair_level[model][key] = mark or r["match_median_ms"]
            continue
        extract[model][key] = mark or r["extract_median_ms"]
        if r.get("match_median_ms") is not None:
            matcher[model][key] = mark or r["match_median_ms"]
            if model in PIPELINES:
                pair[model][key] = mark or 2 * r["extract_median_ms"] + r["match_median_ms"]

    romav2 = defaultdict(dict)
    for path in d.glob("romav2_*.json"):
        r = json.loads(path.read_text())
        gpu, key = r["gpu"], r["engine"]
        best = romav2[r["preset"]].get(key)
        romav2[r["preset"]][key] = r["median_ms"] if best is None else min(best, r["median_ms"])

    out = [f"# Inference latency ({gpu})", "",
           ("Median ms per call, one configuration per process, 10 warmups + 50 timed calls (CUDA events). "
            "Generated by `benchmarks/speed/speed_table.py` from `benchmarks/speed/speed_table.sh` and "
            "`benchmarks/speed/speed_table_models.sh`. `→ int8` / `→ bf16`: the model has no such preset and "
            "builds the one shown; `fail`: the run crashed; `-`: not measured.")]

    out += ["", "## Dense and semi-dense models", "",
            ("RoMaV2: preloaded bidirectional forward, mega1500 setting (800² LR + 1024² HR), B=1. Compiled "
             "engines: best of two processes. Speed-up: over `bf16` in the same engine."), "",
            "| preset | eager | compile | compile-full | speed-up (compile-full) |", "|---|---|---|---|---|"]
    base = romav2.get("bf16", {}).get("compile-full")
    for preset in ROMAV2_PRESETS:
        times = romav2.get(preset, {})
        values = [times.get(e) for e in ROMAV2_ENGINES]
        best = min((v for v in values if v is not None), default=None)
        cells = ["-" if v is None else (f"**{v:.1f}**" if v == best else f"{v:.1f}") for v in values]
        full = times.get("compile-full")
        ratio = f"{base / full:.2f}x" if base and full else "-"
        out.append(f"| `{preset}` | " + " | ".join(cells) + f" | {ratio} |")
    out += _table("", ("`match` of one MegaDepth pair at each model's own default resolution; dense models keep "
                       "5000 sampled matches (`max_matches=5000`), semi-dense ones all their matches. Each cell is "
                       "`eager / compile`."), "model", PAIR_LEVEL, pair_level)[1:]
    # the sub-table above carries an empty title: turn it into a plain paragraph
    out = [line for line in out if line != "## "]

    out += _table("Sparse extractors",
                  ("`extract` of one MegaDepth image, 1200 px long side, 4096 keypoints (1024 px / 1024 "
                   "keypoints for `omniglue` and `superpoint-superglue`, 1024 px / 4096 for `loma-*`); each cell is "
                   "`eager / compile`. SIFT and ORB are OpenCV (fp32, eager)."), "model",
                  EXTRACTORS + OTHER_PIPELINES, extract)
    out += _table("Sparse matchers",
                  ("`match_features` of two sets of keypoints with descriptors (the model's own matcher: LightGlue, "
                   "LoMa, SuperGlue, OmniGlue; mutual nearest neighbours for `rdd`, `orb`, `sift`); median of 20 "
                   "calls after 3 warmups. Each cell is `eager / compile`."), "model", MATCHERS, matcher)
    out += _table("Pair time of sparse pipelines",
                  ("One image pair: 2 x `extract` + the matcher's `match_features`, the column being the "
                   "pipeline's preset, built as `flashim.load` builds it (each component gets the preset its spec "
                   "gives it; the table below the grid lists the pipelines whose components differ). Same sizes as "
                   "above. Each cell is `eager / compile`."),
                  "pipeline", PIPELINES, pair)
    if mixed:
        out += ["", ("Pipelines whose two components build different presets (`extractor+matcher`; a component without the "
                    "requested preset builds the nearest one it has):"), "",
                "| pipeline | requested | built |", "|---|---|---|"]
        out += [f"| {m} | {p} | {built} |" for (m, p), built in sorted(mixed.items())]

    if flex:
        out += ["", "## FlexGlue (sparse-to-dense)", "",
                ("One MegaDepth pair, 512x512 descriptor: `match_keypoints` with 2048 RDD keypoints per image and "
                 "dense `match()` with 5000 sampled matches. Each cell is `eager / compile`."), "",
                "| call | " + " | ".join(PRESETS) + " |", "|---|" + "---|" * len(PRESETS)]
        for label, field in (("`match_keypoints`", "keypoints_median_ms"), ("dense `match()`", "dense_median_ms")):
            cells = {}
            for (preset, engine), r in flex.items():
                cells[(preset, engine)] = r if isinstance(r, str) else r[field]
            out.append(f"| {label} | " + " | ".join(_cells(cells, PRESETS)) + " |")

    failed = sorted(
        f"{model}|{preset}|{engine}" for table in (extract, pair, matcher, pair_level) for model, times in table.items()
        for (preset, engine), value in times.items() if value == "fail"
    )
    failed += sorted(f"flexglue|{p}|{e}" for (p, e), v in flex.items() if v == "fail")
    failed = sorted(set(failed))
    if failed:
        out += ["", "## Failures", ""]
        for tag in failed:  # the first sentence of the exception line in the run's stderr
            err = d / f"{tag.replace('|', '_')}.err"
            lines = [line for line in err.read_text().splitlines() if "Error" in line] if err.exists() else []
            message = lines[-1].strip().split(". ")[0] if lines else "see the .err file"
            out.append(f"- `{tag}`: {message}")

    text = "\n".join(out) + "\n"
    (d / "table.md").write_text(text)
    print(text)


if __name__ == "__main__":
    main()
