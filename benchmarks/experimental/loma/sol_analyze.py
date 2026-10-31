"""Aggregate an ncu ``--csv --page raw`` dump of ``sol_target.py``: time per (NVTX region, kernel kind) with
time-weighted SM / memory / DRAM / tensor-pipe throughput (% of peak), and time per limiter (compute: SM% >= memory%,
memory: memory% > SM%, latency: both < 40%).

    ncu --profile-from-start off --nvtx --csv --page raw --metrics gpu__time_duration.sum,\
sm__throughput.avg.pct_of_peak_sustained_elapsed,gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed,\
dram__throughput.avg.pct_of_peak_sustained_elapsed,sm__inst_executed_pipe_tensor.avg.pct_of_peak_sustained_elapsed \
python benchmarks/experimental/loma/sol_target.py opt > sol.csv
    python benchmarks/experimental/loma/sol_analyze.py sol.csv
"""
import csv
import re
import sys
from collections import defaultdict

KINDS = (("attention", r"attn|flash|fmha|sage|qk_int|sdpa"),
         ("gemm", r"gemm|matmul|cutlass|sm\d+_xmma|nvjet|_mm_|linear"),
         ("conv", r"conv|implicit|cudnn|winograd|igemm"),
         ("quantize", r"quant|row_quant|amax"),
         ("norm/softmax", r"norm|softmax|layer_norm"))
METRICS = (("sm", "sm__throughput"), ("mem", "gpu__compute_memory_throughput"), ("dram", "dram__throughput"),
           ("tc", "sm__inst_executed_pipe_tensor"))


def value(row: dict, key: str) -> float:
    try:
        return float(row[key].replace(",", ""))
    except (KeyError, ValueError):
        return 0.0


def pct(row: dict, metric: str) -> float:
    return value(row, metric + ".avg.pct_of_peak_sustained_elapsed")


def region(row: dict, nvtx: str) -> str:
    names = re.findall(r"(extract|match|det_enc|det_dec|vgg|vit|desc_dec)", row[nvtx])
    if not names:
        return "?"
    return "extract:other" if names[-1] == "extract" else names[-1]


def kind(row: dict) -> str:
    name = row["Kernel Name"].lower()
    return next((k for k, pattern in KINDS if re.search(pattern, name)), "elementwise/other")


def limiter(row: dict) -> str:
    sm, mem = pct(row, "sm__throughput"), pct(row, "gpu__compute_memory_throughput")
    if max(sm, mem) < 40:
        return "latency"
    return "compute" if sm >= mem else "memory"


def main() -> None:
    with open(sys.argv[1]) as f:
        lines = [line for line in f if line.startswith('"')]  # the target's own stdout is interleaved
    rows = list(csv.DictReader(lines))[1:]  # second row = units
    nvtx = next(k for k in rows[0] if k.startswith("thread Domain"))
    agg = defaultdict(lambda: defaultdict(float))
    by_limiter = defaultdict(lambda: defaultdict(float))
    total = defaultdict(float)
    for row in rows:
        ms = value(row, "gpu__time_duration.sum") / 1e6
        g = region(row, nvtx)
        a = agg[(g, kind(row))]
        a["t"] += ms
        a["n"] += 1
        for short, metric in METRICS:
            a[short] += ms * pct(row, metric)
        by_limiter[g][limiter(row)] += ms
        total[g] += ms
    print(f"{'region':14s}{'kind':20s}{'n':>5s}{'ms':>8s}{'SM%':>6s}{'mem%':>6s}{'DRAM%':>6s}{'TC%':>6s}")
    for (g, k), a in sorted(agg.items(), key=lambda kv: (kv[0][0], -kv[1]["t"])):
        t = a["t"]
        print(f"{g:14s}{k:20s}{int(a['n']):5d}{t:8.2f}" + "".join(f"{a[s] / t:6.0f}" for s, _ in METRICS))
    print("\ntime by limiter")
    for g in sorted(total):
        print(f"{g:14s} total {total[g]:7.2f} ms  " + "  ".join(
            f"{b} {by_limiter[g][b]:6.2f} ({by_limiter[g][b] / total[g]:.0%})" for b in ("compute", "memory", "latency")))


if __name__ == "__main__":
    main()
