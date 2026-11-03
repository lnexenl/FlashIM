"""Can torch.compile fuse the dynamic INT8 ViT's glue? loma-l's DINOv2 ViT alone on a real MegaDepth input (784 px,
3137 tokens), CUDA-event median and min of 20 forwards, output cosine vs the BF16 eager ViT. One config per process (two
compiled instances in one process share Dynamo graphs).

Configs: ``bf16`` / ``bf16-compile``; ``triton`` (the prototype's hand-written row-quantize + INT8 GEMM kernels, eager
glue) / ``triton-compile`` (same kernels, Inductor fuses the glue around them); ``naive-compile`` /
``naive-autotune`` (the dynamic INT8 linear written in plain torch ops: row amax, round, ``torch._int_mm``, scales and
bias in torch, compiled with the default / ``max-autotune-no-cudagraphs`` mode, so Inductor can fuse the quantization into
LayerNorm / GELU and the scales into the GEMM); ``hybrid-compile`` (row quantize in torch ops, fused by Inductor, and the
prototype's Triton GEMM with the scales in its epilogue). INT8 configs use sage-fp8 attention unless ``--sdpa`` says otherwise.

    PYTHONPATH=$PWD python benchmarks/experimental/loma/vit_fusion_probe.py --config naive-compile
"""
import argparse
import json
import statistics
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import Tensor, nn

sys.path.insert(0, str(Path(__file__).resolve().parent))
from vit_dynamic_int8 import EPS, DynamicInt8Linear, gemm, quantize_vit, row_quant_reference

import flashim

IMAGE = "assets/megadepth_0015/101776896_21fc24b031_o.jpg"
CACHE = Path("results/loma/vit_fusion_input.pt")


class NaiveDynamicInt8Linear(nn.Module):
    """:class:`DynamicInt8Linear` in plain torch ops (same weights, same rounding: half to even)."""

    def __init__(self, linear: DynamicInt8Linear) -> None:
        super().__init__()
        self.register_buffer("weight_t", linear.weight.t())  # [K, N] column-major, cuBLASLt's int8 layout
        self.register_buffer("w_scale", linear.w_scale)
        self.register_buffer("bias", linear.bias)

    def forward(self, x: Tensor) -> Tensor:
        lead, k = x.shape[:-1], x.shape[-1]
        x = x.reshape(-1, k).float()
        s = (x.abs().amax(-1, keepdim=True) / 127.0).clamp_min(EPS)
        q = (x / s).round().clamp(-127, 127).to(torch.int8)
        out = torch._int_mm(q, self.weight_t).float() * (s * self.w_scale)
        if self.bias is not None:
            out = out + self.bias
        return out.to(torch.bfloat16).reshape(*lead, -1)


class HybridDynamicInt8Linear(DynamicInt8Linear):
    """Row quantize in plain torch ops (Inductor fuses it into the producer: LayerNorm, GELU, residual), the
    prototype's INT8 Triton GEMM with the scales and bias in its epilogue (BF16 out, no INT32 round trip)."""

    def __init__(self, linear: DynamicInt8Linear) -> None:
        nn.Module.__init__(self)
        self.register_buffer("weight", linear.weight)
        self.register_buffer("w_scale", linear.w_scale)
        self.register_buffer("bias", linear.bias)

    def forward(self, x: Tensor) -> Tensor:
        lead, k = x.shape[:-1], x.shape[-1]
        q, s = row_quant_reference(x.reshape(-1, k))
        return gemm(q, s, self.weight, self.w_scale, self.bias).reshape(*lead, -1)


def vit_input(extractor: nn.Module) -> Tensor:
    if CACHE.exists():
        return torch.load(CACHE).cuda()
    captured = []
    handle = extractor.descriptor.encoder.dinov2.register_forward_pre_hook(lambda m, a: captured.append(a[0].clone()))
    extractor.extract(IMAGE, resize=1024, top_k=4096)
    handle.remove()
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    torch.save(captured[0].cpu(), CACHE)
    return captured[0]


def build(config: str, sdpa: str | None) -> tuple[nn.Module, nn.Module, Tensor]:
    extractor = flashim.load("loma-l", "bf16" if config.startswith("bf16") else "int8").extractor
    reference = flashim.load("loma-l", "bf16").extractor.descriptor.encoder.dinov2
    x = vit_input(extractor)
    vit = extractor.descriptor.encoder.dinov2
    if not config.startswith("bf16"):
        quantize_vit(extractor)
        if config.startswith(("naive", "hybrid")):
            cls = NaiveDynamicInt8Linear if config.startswith("naive") else HybridDynamicInt8Linear
            for block in vit.blocks:
                for parent in (block.attn, block.mlp):
                    for name, child in list(parent.named_children()):
                        if isinstance(child, DynamicInt8Linear):
                            setattr(parent, name, cls(child))
        for block in vit.blocks:
            block.attn.sdpa_impl = sdpa
    if config.endswith("-compile"):
        vit = torch.compile(vit, dynamic=False)
    elif config.endswith("-autotune"):
        vit = torch.compile(vit, dynamic=False, mode="max-autotune-no-cudagraphs")
    return vit, reference, x


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--sdpa", default="sage-fp8", help="INT8 configs' ViT attention impl (bf16 configs: dispatch)")
    args = parser.parse_args()
    with torch.inference_mode():
        vit, reference, x = build(args.config, None if args.config.startswith("bf16") else args.sdpa)
        expected = reference(x).float()
        del reference
        for _ in range(5):
            out = vit(x)
        torch.cuda.synchronize()
        times = []
        for _ in range(20):
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            start.record()
            out = vit(x)
            end.record()
            torch.cuda.synchronize()
            times.append(start.elapsed_time(end))
        cos = F.cosine_similarity(out.float().flatten(1), expected.flatten(1)).item()
    print(json.dumps({"config": args.config, "sdpa": args.sdpa if not args.config.startswith("bf16") else "dispatch",
                      "vit_ms": round(statistics.median(times), 2), "vit_min_ms": round(min(times), 2), "cos_vs_bf16": round(cos, 5)}))


if __name__ == "__main__":
    main()
