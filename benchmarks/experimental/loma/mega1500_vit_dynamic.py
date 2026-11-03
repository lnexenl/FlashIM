"""MegaDepth-1500 matches of ``loma-<variant>`` int8 with the extractor's DINOv2 ViT linears (BF16 in the preset) replaced
by the per-token dynamic INT8 prototype (``vit_dynamic_int8.py``), saved in ``run_sparse.py --save-matches`` format.

    PYTHONPATH=$PWD python benchmarks/experimental/loma/mega1500_vit_dynamic.py --variant l \
        --out results/loma/matches/loma-l_int8dynvit.npz [--sdpa sage-fp8] [--matcher-attention sage-fp8] \
        [--decoder-fusion]
"""
import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path("benchmarks/accuracy").resolve()))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from decoder_fusion import fuse_producers, fuse_refiners
from run_sparse import MatchRecorder, mega1500_sparse
from vit_dynamic_int8 import quantize_vit

import flashim
from flashim.paths import DATA


class _NoPool:
    def submit(self, *job) -> None:
        pass


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", default="l")
    parser.add_argument("--max-pairs", type=int, default=0)
    parser.add_argument("--sdpa", default=None, help="the ViT's ops.sdpa impl (e.g. sage-fp8); default: as built")
    parser.add_argument("--matcher-attention", default=None, help="the matcher's optimize.attention (e.g. sage-fp8)")
    parser.add_argument("--decoder-fusion", action="store_true",
                        help="decoder_fusion.py's A+B+D (depthwise INT8 producer, INT8 upsample-concat, folded residual)")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    with torch.inference_mode():
        overrides = ([f"components.matcher.optimize.attention={args.matcher_attention}"]
                     if args.matcher_attention else None)
        pipeline = flashim.load(f"loma-{args.variant}", "int8", overrides=overrides)
        print("dynamic ViT linears:", quantize_vit(pipeline.extractor), flush=True)
        if args.sdpa:
            for block in pipeline.extractor.descriptor.encoder.dinov2.blocks:
                block.attn.sdpa_impl = args.sdpa
        if args.decoder_fusion:
            blocks = fuse_producers(pipeline.extractor)
            print("decoder fusion:", len(blocks), "producer blocks,", fuse_refiners(pipeline.extractor, True, True),
                  "refiners", flush=True)
        recorder = MatchRecorder(_NoPool())
        count = mega1500_sparse(pipeline, DATA / "megadepth", recorder, args.max_pairs, {"workers": 4, "depth": 16},
                                {"resize": 1024, "max_matches": 4096}, False)
    recorder.save(args.out)
    print(f"{count} pairs -> {args.out}")


if __name__ == "__main__":
    main()
