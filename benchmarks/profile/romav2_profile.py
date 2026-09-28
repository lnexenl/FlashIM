"""Where does RoMaV2 inference time go? Region timings, kernel categories, measured roofline.

Three passes over one preloaded MegaDepth pair (mega1500 setting, B=1):

1. ``regions``: CUDA events on module boundaries. With ``--engine compile`` only
   uncompiled boundaries are hooked (hooks inside compiled regions would break
   their graphs); ``--engine eager`` also times sub-regions (attention/MLP blocks,
   refiner internals).
2. ``kernels``: torch.profiler CUDA kernel time, grouped into categories.
3. ``roofline``: measured peak of BF16 GEMM, MXFP8 block-scaled GEMM and DRAM copy
   bandwidth on this GPU, so region FLOPs/bytes can be put in context.

    python benchmarks/profile/romav2_profile.py --preset fp8 --engine compile --out results/profile/fp8_compile.json
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
from collections import defaultdict
from pathlib import Path

import torch

from flashim import ops
from flashim.models.romav2.build import load_romav2

ROMAV2_DATA = Path.home() / "work" / "RoMaV2" / "data"
IMAGE_A = ROMAV2_DATA / "megadepth/Undistorted_SfM/0015/images/3383162411_67cc9af85e_o.jpg"
IMAGE_B = ROMAV2_DATA / "megadepth/Undistorted_SfM/0015/images/299056923_ecd980298c_o.jpg"

TOP_REGIONS = ("f", "matcher", "matcher.mv_vit", "matcher.head", "refiner_features", "refiners.4", "refiners.2", "refiners.1")
EAGER_SUBREGIONS = (
    r"refiners\.\d\.(proj|block1|hidden_blocks|warp_head|confidence_head|disp_emb)",
    r"f\.blocks\.\d+",
    r"matcher\.mv_vit\.blocks\.\d+",
    r"matcher\.mv_vit\.(projector|output_projector)",
)

# (category, regex over CUDA kernel names), first match wins.
KERNEL_CATEGORIES = (
    ("gemm_mxfp8 (cuBLAS scaled_mm)", r"(?i)(block_?scal|mxf8|e4m3.*e4m3|f8f6f4|nvjet.*e4m3|sm1[02]\d.*f8)"),
    ("attention (SDPA/flash)", r"(?i)(flash|fmha|attention|efficient_attention)"),
    ("triton: layernorm/gelu->quant", r"_(layernorm|gelu)_quantize_kernel"),
    ("triton: depthwise+relu->mxfp8", r"_depthwise5x5_relu_mxfp8_kernel"),
    ("triton: rope", r"_rope_qk_kernel"),
    ("triton: fp8 gemm fallback", r"_fp8_gemm_kernel"),
    ("local_corr (fused_local_corr)", r"(?i)corr"),
    ("grid_sample / interpolate", r"(?i)(grid_sampl|upsample|interpolat|bilinear|bicubic)"),
    ("conv (cuDNN/implicit gemm)", r"(?i)(conv|implicit|cudnn|winograd|fprop|dgrad|xmma_fprop|depthwise)"),
    ("gemm bf16/fp32 (cuBLAS/cutlass)", r"(?i)(gemm|nvjet|cutlass|sm\d+_xmma|ampere_|sgemm|matmul)"),
    ("inductor fused (triton_*)", r"^triton_"),
    ("softmax / reduction", r"(?i)(softmax|reduce|norm|cdist)"),
    ("elementwise / copy / cast", r"(?i)(elementwise|vectorized|copy|cat|fill|index|unrolled|where|cast)"),
)


def measure_roofline(device: str = "cuda") -> dict[str, float]:
    def best_ms(fn, iters=20):
        for _ in range(3):
            fn()
        torch.cuda.synchronize()
        times = []
        for _ in range(iters):
            s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            s.record()
            fn()
            e.record()
            e.synchronize()
            times.append(s.elapsed_time(e))
        return min(times)

    n = 8192
    a = torch.randn(n, n, device=device, dtype=torch.bfloat16)
    b = torch.randn(n, n, device=device, dtype=torch.bfloat16)
    bf16_tflops = 2 * n**3 / best_ms(lambda: a @ b) / 1e9

    qa = a.clamp(-448, 448).to(torch.float8_e4m3fn)
    qb = b.clamp(-448, 448).to(torch.float8_e4m3fn)
    groups = n // 32
    scale = torch.full((n, groups), 127, device=device, dtype=torch.uint8).view(torch.float8_e8m0fnu).reshape(-1)
    mx_tflops = 2 * n**3 / best_ms(lambda: ops.mxfp8_scaled_mm.get("cublas")(qa, qb.t(), scale, scale, None)) / 1e9

    x = torch.empty(512 * 2**20, device=device, dtype=torch.uint8)
    y = torch.empty_like(x)
    bandwidth = 2 * x.numel() / best_ms(lambda: y.copy_(x)) / 1e6  # GB/s (read + write)
    return {"bf16_gemm_tflops": bf16_tflops, "mxfp8_gemm_tflops": mx_tflops, "dram_copy_gbps": bandwidth}


class RegionTimer:
    def __init__(self, model: torch.nn.Module, patterns: list[str]) -> None:
        self.events: dict[str, list[tuple]] = defaultdict(list)
        self.handles = []
        self.enabled = False
        for name, module in model.named_modules():
            if any(re.fullmatch(p, name) for p in patterns):
                self.handles.append(module.register_forward_pre_hook(self._pre(name)))
                self.handles.append(module.register_forward_hook(self._post(name)))
        self._open: dict[str, list] = defaultdict(list)

    def _pre(self, name):
        def hook(module, args):
            if self.enabled:
                event = torch.cuda.Event(enable_timing=True)
                event.record()
                self._open[name].append(event)

        return hook

    def _post(self, name):
        def hook(module, args, output):
            if self.enabled and self._open[name]:
                event = torch.cuda.Event(enable_timing=True)
                event.record()
                self.events[name].append((self._open[name].pop(), event))

        return hook

    def collect(self, iterations: int) -> dict[str, dict[str, float]]:
        torch.cuda.synchronize()
        out = {}
        for name, pairs in self.events.items():
            total = sum(s.elapsed_time(e) for s, e in pairs)
            out[name] = {"ms_per_forward": total / iterations, "calls_per_forward": len(pairs) / iterations}
        return out


def kernel_breakdown(model, inputs, iterations: int) -> dict:
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]) as prof:
        with torch.inference_mode():
            for _ in range(iterations):
                model(*inputs)
        torch.cuda.synchronize()
    per_kernel: dict[str, list[float]] = defaultdict(lambda: [0.0, 0])
    for event in prof.events():
        if event.device_type == torch.autograd.DeviceType.CUDA:
            per_kernel[event.name][0] += event.device_time_total / 1000.0
            per_kernel[event.name][1] += 1
    categories: dict[str, float] = defaultdict(float)
    kernels = []
    for name, (ms, count) in per_kernel.items():
        category = next((c for c, pattern in KERNEL_CATEGORIES if re.search(pattern, name)), "other")
        categories[category] += ms / iterations
        kernels.append({"name": name[:160], "category": category, "ms": ms / iterations, "calls": count / iterations})
    kernels.sort(key=lambda k: -k["ms"])
    total = sum(categories.values())
    return {
        "gpu_kernel_ms_per_forward": total,
        "categories": dict(sorted(categories.items(), key=lambda kv: -kv[1])),
        "top_kernels": kernels[:40],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--preset", default="fp8")
    parser.add_argument("--engine", default="compile", choices=("eager", "compile", "compile-full"))
    parser.add_argument("--setting", default="mega1500")
    parser.add_argument("--warmups", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--skip-roofline", action="store_true")
    parser.add_argument("--out", default="")
    args = parser.parse_args()
    torch.set_float32_matmul_precision("highest")

    model = load_romav2(args.preset, setting=args.setting, engine=args.engine)
    inputs = model.prepare(IMAGE_A, IMAGE_B)
    patterns = [re.escape(r) for r in TOP_REGIONS]
    if args.engine == "eager":
        patterns += list(EAGER_SUBREGIONS)
    else:  # sub-modules of compiled regions (f, matcher.mv_vit) must not be hooked
        patterns += [p for p in EAGER_SUBREGIONS if p.startswith("refiners")]
    timer = RegionTimer(model, patterns)

    with torch.inference_mode():
        for _ in range(args.warmups):
            model(*inputs)
        torch.cuda.synchronize()
        # End-to-end time without hook overhead.
        totals = []
        for _ in range(args.iterations):
            s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            s.record()
            model(*inputs)
            e.record()
            e.synchronize()
            totals.append(s.elapsed_time(e))
        timer.enabled = True
        for _ in range(args.iterations):
            model(*inputs)
        regions = timer.collect(args.iterations)
        timer.enabled = False

    ops.reset_call_counts()
    with torch.inference_mode():
        model(*inputs)
    op_impls = ops.call_counts()
    kernels = kernel_breakdown(model, inputs, max(3, args.iterations // 4))

    result = {
        "preset": args.preset,
        "engine": args.engine,
        "setting": args.setting,
        "gpu": torch.cuda.get_device_name(),
        "forward_median_ms": statistics.median(totals),
        "regions": dict(sorted(regions.items(), key=lambda kv: -kv[1]["ms_per_forward"])),
        "op_impls_per_forward": op_impls,
        "kernels": kernels,
        "roofline": None if args.skip_roofline else measure_roofline(),
    }
    print(json.dumps({k: result[k] for k in ("preset", "engine", "forward_median_ms", "roofline")}, indent=2))
    for name, row in result["regions"].items():
        if not re.search(r"\.(blocks\.\d+)$", name):
            print(f"  {name:36s} {row['ms_per_forward']:8.2f} ms  x{row['calls_per_forward']:.0f}")
    print("kernel categories (ms/forward):")
    for name, ms in kernels["categories"].items():
        print(f"  {name:36s} {ms:8.2f}")
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
