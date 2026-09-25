"""HardMatch (LoMa, ECCV 2026): 1000 hand-annotated difficult Wikimedia pairs (900 test / 100 val).

Uses the official ``hardmatch`` package protocol unchanged: the matcher returns pixel
keypoints of the original files; a MAGSAC fundamental matrix (0.25 px, 100k iterations) is
fit and scored by the symmetric epipolar error of the annotated correspondences; the metric
is mAA over 0..10 px. Data (660 MB) is downloaded on first use.

    python benchmarks/accuracy/run_hardmatch.py --preset fp8-fast --engine compile-full --seed 17
"""

from __future__ import annotations

import argparse
import json
import random
import subprocess
import time
from pathlib import Path

import numpy as np
import torch

import flashim

DEFAULT_ROOT = Path.home() / "work" / "FlashIM" / "data" / "hardmatch"


class FlashIMMatcher:
    """``hardmatch.types.BenchmarkMatcher`` adapter: dense match + balanced sampling."""

    def __init__(self, model, num_samples: int, offset: float = 0.0) -> None:
        self.model = model
        self.num_samples = num_samples
        # FlashIM keypoints use the pixel-center convention of RoMa/HardMatch (x_pix = (x + 1) / 2 * W).
        self.offset = offset
        self.seconds = 0.0
        self.pairs = 0

    def match(self, img_A_path, img_B_path) -> tuple[np.ndarray, np.ndarray]:
        torch.cuda.synchronize()
        start = time.perf_counter()
        result = self.model.match(img_A_path, img_B_path, num_samples=self.num_samples)
        kpts_a = result.keypoints_A.double().cpu().numpy()
        kpts_b = result.keypoints_B.double().cpu().numpy()
        self.seconds += time.perf_counter() - start
        self.pairs += 1
        return kpts_a, kpts_b


def _git(*args: str) -> str:
    try:
        return subprocess.run(["git", *args], cwd=Path(__file__).parent, capture_output=True, text=True,
                              check=True).stdout.strip()
    except Exception:
        return ""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default="romav2")
    parser.add_argument("--preset", default="fp8-fast")
    parser.add_argument("--engine", default="compile-full", choices=("eager", "compile", "compile-full"))
    parser.add_argument("--setting", default="wxbs", help="RoMaV2 resolution preset (wxbs: 800 LR + 1024 HR)")
    parser.add_argument("--num-samples", type=int, default=5000)
    parser.add_argument("--split", default="test", choices=("test", "val"))
    parser.add_argument("--data-root", default=str(DEFAULT_ROOT))
    parser.add_argument("--offset", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--out", default="")
    args = parser.parse_args()

    from hardmatch import HardMatchBenchmark

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    try:
        import cv2

        cv2.setRNGSeed(args.seed)
    except ImportError:
        pass

    model = flashim.load(args.model, preset=args.preset, setting=args.setting, engine=args.engine)
    matcher = FlashIMMatcher(model, args.num_samples, args.offset)
    benchmark = HardMatchBenchmark(HardMatchBenchmark.Cfg(data_root=args.data_root, split=args.split))
    started = time.time()
    metrics = benchmark.benchmark(matcher)
    pck = np.asarray(metrics["avg_pck"])
    result = {
        "benchmark": "hardmatch",
        "split": args.split,
        "model": args.model,
        "preset": args.preset,
        "engine": args.engine,
        "setting": args.setting,
        "num_samples": args.num_samples,
        "offset": args.offset,
        "seed": args.seed,
        "pairs_scored": matcher.pairs,
        "mAA_10px": metrics["mAA_10px"],
        "mAA_5px": float(pck[:6].mean()),
        "avg_pck": metrics["avg_pck"],
        "match_ms_per_pair": 1000 * matcher.seconds / max(matcher.pairs, 1),
        "seconds": time.time() - started,
        "fast_sampling": getattr(model, "fast_sampling", None),
        "torch": torch.__version__,
        "gpu": torch.cuda.get_device_name() if torch.cuda.is_available() else "cpu",
        "flashim_git": _git("rev-parse", "HEAD"),
    }
    print(json.dumps({k: v for k, v in result.items() if k != "avg_pck"}, indent=2))
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
