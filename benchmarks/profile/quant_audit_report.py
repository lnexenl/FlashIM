"""Summarize ``quant_audit.py`` reports: coverage, unquantized layers and why, quantized units slower than BF16,
module classes and glue per preset, op impls (reference fallbacks).

    python benchmarks/profile/quant_audit_report.py results/profile/quant_audit [model ...]
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path


def under(name: str, unit: str) -> bool:
    return name == unit or name.startswith(unit + ".")


def summarize(path: Path) -> None:
    d = json.loads(path.read_text())
    presets = {p: r for p, r in d["presets"].items() if "units" in r}
    print(f"\n=== {path.stem} ({next(iter(presets.values()))['stage']})")
    for p, r in presets.items():
        by_cls: dict[str, float] = defaultdict(float)
        for u in r["units"].values():
            by_cls[u["cls"]] += u["ms"]
        top = ", ".join(f"{c} {ms:.1f}" for c, ms in sorted(by_cls.items(), key=lambda kv: -kv[1])[:6])
        print(f"  {p:5s} total {r['total_ms']:7.1f} ms  glue {r['total_ms'] - r['module_ms']:6.1f} | {top}")
    bf16 = presets.get("bf16", {}).get("units", {})
    parents = presets.get("bf16", {}).get("parents", {})
    layers = d["layers"]
    for p in ("int8", "fp8"):
        if p not in presets:
            continue
        r = presets[p]
        status: dict[str, list] = defaultdict(list)
        for name, row in layers.items():
            status[row.get(p, "?")].append((row["bf16_ms"], name, row))
        total = sum(row["bf16_ms"] for row in layers.values()) or 1.0
        cov = "; ".join(f"{s}: {len(v)} layers, {100 * sum(x[0] for x in v) / total:.0f}% of bf16 conv/linear time" for s, v in sorted(status.items(), key=lambda kv: -sum(x[0] for x in kv[1])))
        print(f"  [{p}] {cov}")
        left = sorted((x for s, v in status.items() if s != "quantized" for x in v), reverse=True)[:5]
        if left:
            print("     largest unquantized: " + "; ".join(f"{n} ({row['cls']}{' g=' + str(row['groups']) if row['groups'] not in (None, 1) else ''} {row['dtype']} {row['shape']}) {ms:.2f} ms [{row[p]}]" for ms, n, row in left if ms > 0))
        qunits = {n: u for n, u in r["units"].items() if u["quant"]}
        rows = []
        for n, u in qunits.items():
            base = parents[n] if n in parents else sum(b["ms"] for bn, b in bf16.items() if under(bn, n))
            rows.append((u["ms"] - base, n, u, base))
        qb, qq = sum(x[3] for x in rows), sum(x[2]["ms"] for x in rows)
        slower = [x for x in sorted(rows, reverse=True) if x[2]["ms"] > 1.05 * x[3] and x[0] > 0.02]
        print(f"     quantized units: {len(rows)}, bf16 {qb:.1f} ms -> {p} {qq:.1f} ms; slower than bf16: {len(slower)} units, +{sum(x[0] for x in slower):.1f} ms")
        for diff, n, u, base in slower[:4]:
            print(f"       {n} ({u['cls']} {u['shape']} x{u['calls']}): {base:.2f} -> {u['ms']:.2f} ms")
        faster = sorted(rows)[:2]
        print("       best: " + "; ".join(f"{n} {base:.2f} -> {u['ms']:.2f}" for diff, n, u, base in faster if diff < 0))
        refs = {op: impls for op, impls in r["ops"].items() if "reference" in impls}
        print(f"     ops: {r['ops']}" + (f"  REFERENCE: {refs}" if refs else ""))


def main() -> None:
    root = Path(sys.argv[1])
    wanted = sys.argv[2:]
    for path in sorted(root.glob("*.json")):
        if not wanted or path.stem in wanted:
            summarize(path)


if __name__ == "__main__":
    main()
