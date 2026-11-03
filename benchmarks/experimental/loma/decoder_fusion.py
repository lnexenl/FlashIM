"""Prototype (experiment, not library code): decoder fusions for the int8 preset's ConvRefiners. Config tokens
(``+``-joined, ``-compile`` suffix: also compile the two decoders):

- ``fused``: the hidden blocks at 256 / 512 / 768 channels (the ones ``fused_bf16_blocks`` leaves: cuDNN BF16 depthwise
  5x5, a ReLU pass, a layout / cast pass, a static INT8 quantize, the INT8 pointwise GEMM) as
  ``ops.depthwise5x5_relu_int8`` (one producer kernel: depthwise + bias + ReLU + static INT8 with the pointwise conv's
  calibrated scale, channels-last) feeding the same pointwise GEMM.
- ``concat``: the refiner input (``ops.upsample_concat``: features + upsampled context, BF16) written as ``block1.0``'s
  INT8 input directly, without the BF16 map and the separate quantize pass.
- ``res``: the residual ``(x + x0) / 1.4`` as ``x + x0`` with ``1 / 1.4`` folded into ``out_conv``'s weights.

Speed: loma-l extract, eager, base vs fused interleaved in one process, CUDA-event median of 20 per round; ``--compile``
compiles the two decoders (``detector.decoder`` / ``descriptor.decoder``) of one config per process instead.
Accuracy: keypoints and descriptors vs the unfused int8 extractor on the same images.

    PYTHONPATH=$PWD python benchmarks/experimental/loma/decoder_fusion.py
"""
import argparse
import json
import statistics
from dataclasses import dataclass

import torch
import torch.nn.functional as F
import triton
import triton.language as tl
from torch import Tensor, nn
from triton.language.extra import libdevice

import flashim
from flashim import ops
from flashim.backends.triton.resample import _bilinear_source, _bilinear_tile
from flashim.layers.conv import IndexedSeparableConvBlock, sequential_forward
from flashim.models.dedode.blocks import ConvRefiner, Decoder
from flashim.quant.methods.int8_linear import StaticInt8Conv1x1
from flashim.quant.passes.separable_conv_block import _packed_depthwise_weight
from flashim.quant.tensor import Int8Tensor

IMAGES = ("assets/megadepth_0015/101776896_21fc24b031_o.jpg", "assets/megadepth_0015/101795964_72e8f63ab1_o.jpg")


class ProducerInt8Block(nn.Module):
    """``pointwise(relu(depthwise(x)))`` with the depthwise conv, bias, ReLU and the pointwise conv's static INT8
    quantization in one kernel (BF16 rounding before quantizing, as the unfused path)."""

    def __init__(self, block: IndexedSeparableConvBlock) -> None:
        super().__init__()
        depthwise = block.conv_depthwise
        self.channels = depthwise.in_channels
        self.register_buffer("depthwise_weight", _packed_depthwise_weight(depthwise))
        self.register_buffer("depthwise_bias", depthwise.bias.detach().float().contiguous())
        self.pointwise = block.conv_pointwise

    def forward(self, x: Tensor) -> Tensor:
        batch, channels, height, width = x.shape
        if not x.is_contiguous(memory_format=torch.channels_last):
            x = x.contiguous(memory_format=torch.channels_last)
        scale = self.pointwise.input_scale
        q = ops.depthwise5x5_relu_int8(x, self.depthwise_weight, self.depthwise_bias, scale)
        out = self.pointwise.forward_prequantized(Int8Tensor(q.view(batch, height, width, channels), scale))
        return out.permute(0, 3, 1, 2)


def fuse_producers(extractor: nn.Module) -> list[str]:
    replaced = []
    for name, module in list(extractor.named_modules()):
        if (isinstance(module, IndexedSeparableConvBlock) and isinstance(module.conv_pointwise, StaticInt8Conv1x1)
                and isinstance(module.norm, nn.Identity) and module.conv_depthwise.in_channels % 32 == 0):
            parent, leaf = name.rsplit(".", 1)
            setattr(extractor.get_submodule(parent), leaf, ProducerInt8Block(module))
            replaced.append(name)
    return replaced


@triton.jit
def _upsample_concat_int8_kernel(
    coarse_ptr, other_ptr, out_ptr, scale_ptr,
    C_OTHER, C_COARSE, H, W, h, w, scale_h, scale_w,
    cs_b, cs_c, cs_h, cs_w, os_b, os_c, os_h, os_w,
    BLOCK_P: tl.constexpr, BLOCK_C: tl.constexpr,
):
    """``_upsample_combine_kernel``'s concat branch with the BF16 store replaced by the static INT8 quantization of
    the BF16 value (``_store_quantized``'s rounding): bit-identical to upsample_concat + quantize_int8."""
    p = tl.program_id(0) * BLOCK_P + tl.arange(0, BLOCK_P)
    b = tl.program_id(2).to(tl.int64)
    y, x = p // W, p % W
    p_ok = p < H * W
    y0, y1, ly0, ly1 = _bilinear_source(y, h, scale_h)
    x0, x1, lx0, lx1 = _bilinear_source(x, w, scale_w)
    c = tl.program_id(1) * BLOCK_C + tl.arange(0, BLOCK_C)
    c_out = C_OTHER + C_COARSE
    out_off = ((b * H * W + p)[:, None] * c_out + c[None, :])
    other_off = b * os_b + y[:, None].to(tl.int64) * os_h + x[:, None].to(tl.int64) * os_w
    from_other = c < C_OTHER
    copy = tl.load(other_ptr + other_off + c[None, :] * os_c, mask=p_ok[:, None] & from_other[None, :], other=0.0)
    cc = tl.where(from_other, 0, c - C_OTHER)
    ok = p_ok[:, None] & (~from_other & (c < c_out))[None, :]
    sample = _bilinear_tile(coarse_ptr + b * cs_b, cs_c, cs_h, cs_w, cc, y0, y1, x0, x1, ly0, ly1, lx0, lx1, ok)
    value = tl.where(from_other[None, :], copy.to(tl.float32), sample).to(tl.bfloat16).to(tl.float32)
    scaled = libdevice.rint(value * (1.0 / tl.load(scale_ptr)))
    tl.store(out_ptr + out_off, tl.maximum(tl.minimum(scaled, 127.0), -127.0).to(tl.int8),
             mask=p_ok[:, None] & (c < c_out)[None, :])


def upsample_concat_int8(features: Tensor, coarse: Tensor, scale: Tensor) -> Int8Tensor:
    batch, c_features, height, width = features.shape
    c_coarse, h, w = coarse.shape[1:]
    channels = c_features + c_coarse
    out = torch.empty(batch, height, width, channels, device=features.device, dtype=torch.int8)
    block_p, block_c = 64, min(64, triton.next_power_of_2(channels))
    grid = (triton.cdiv(height * width, block_p), triton.cdiv(channels, block_c), batch)
    _upsample_concat_int8_kernel[grid](
        coarse, features, out, scale, c_features, c_coarse, height, width, h, w, h / height, w / width,
        *coarse.stride(), *features.stride(), BLOCK_P=block_p, BLOCK_C=block_c, num_warps=4,
    )
    return Int8Tensor(out, scale)


@dataclass
class PendingInput:
    """A refiner input not formed yet: ``FusedRefiner.hidden`` writes it straight as ``block1.0``'s INT8 input."""

    features: Tensor
    context: Tensor


class FusedDecoder(Decoder):
    def refiner_input(self, features: Tensor, context: Tensor | None):
        if context is not None and context.shape[-2:] != features.shape[-2:]:
            return PendingInput(features, context)
        return Decoder.refiner_input(features, context)

    def forward(self, features: Tensor, context: Tensor | None, scale: str) -> tuple[Tensor, Tensor]:
        out = self.layers[scale](self.refiner_input(features, context))
        return out[:, : self.num_prototypes], out[:, self.num_prototypes :]


class FusedRefiner(ConvRefiner):
    concat_int8 = False
    fold_residual = False

    def hidden(self, x) -> Tensor:
        if isinstance(x, PendingInput):
            first = self.block1[0]
            if self.concat_int8 and isinstance(first, StaticInt8Conv1x1):
                q = upsample_concat_int8(x.features, x.context, first.input_scale)
                x0 = sequential_forward(self.block1[3:], first.forward_prequantized(q, relu=True).permute(0, 3, 1, 2))
            else:
                x0 = sequential_forward(self.block1, ops.upsample_concat(x.features, x.context))
        else:
            x0 = sequential_forward(self.block1, x)
        x = self.hidden_blocks(x0)
        if not self.residual:
            return x
        return x + x0 if self.fold_residual else (x + x0) / 1.4


def fuse_refiners(extractor: nn.Module, concat: bool, residual: bool) -> int:
    count = 0
    for net in (extractor.detector, extractor.descriptor):
        net.decoder.__class__ = FusedDecoder
        for refiner in net.decoder.layers.values():
            refiner.__class__ = FusedRefiner
            refiner.concat_int8 = concat
            if residual and refiner.residual:
                refiner.fold_residual = True
                refiner.out_conv.weight.data /= 1.4
            count += 1
    return count


def build(config: str) -> nn.Module:
    extractor = flashim.load("loma-l", "int8").extractor
    tokens = set(config.removesuffix("-compile").split("+"))
    if "fused" in tokens:
        print(f"# {config}: {len(fuse_producers(extractor))} blocks fused")
    if tokens & {"concat", "res"}:
        fuse_refiners(extractor, "concat" in tokens, "res" in tokens)
    if config.endswith("-compile"):
        for net in (extractor.detector, extractor.descriptor):
            net.decoder = torch.compile(net.decoder, dynamic=False)
    return extractor


def time_extract(extractor: nn.Module) -> float:
    times = []
    for _ in range(20):
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        extractor.extract(IMAGES[0], resize=1024, top_k=4096)
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))
    return statistics.median(times)


def compare(reference: nn.Module, candidate: nn.Module) -> dict:
    out = {}
    for i, image in enumerate(IMAGES):
        a = reference.extract(image, resize=1024, top_k=4096)
        b = candidate.extract(image, resize=1024, top_k=4096)
        # keypoints as sets (a borderline point flipping shifts every later index): fraction within 0.01 / 0.1 / 1 px;
        # descriptors compared at the points shared within 0.01 px
        ka, kb, da, db = (t.reshape(-1, t.shape[-1]) for t in (a.keypoints, b.keypoints, a.descriptors, b.descriptors))
        dist = torch.cdist(ka.float(), kb.float(), compute_mode="donot_use_mm_for_euclid_dist")
        nearest, j = dist.min(1)
        for tol in (0.01, 0.1, 1.0):
            out[f"img{i}_kp_within_{tol}px"] = round((nearest < tol).float().mean().item(), 4)
        shared = nearest < 0.01
        cos = F.cosine_similarity(da[shared].float(), db[j[shared]].float(), dim=-1)
        out[f"img{i}_desc_cos_mean"] = round(cos.mean().item(), 6)
        out[f"img{i}_desc_cos_min"] = round(cos.min().item(), 6)
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--configs", default="base,fused", help="comma-separated; eager configs share one process")
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--vs-bf16", action="store_true", help="also compare the bf16 extractor against the int8 one")
    args = parser.parse_args()
    configs = args.configs.split(",")
    with torch.inference_mode():
        models = {c: build(c) for c in configs}
        reference = models.get("base") or build("base")
        if args.vs_bf16:  # scale of the change: the int8 preset itself vs the bf16 one
            models["bf16"] = flashim.load("loma-l", "bf16").extractor
            configs.append("bf16")
        for model in models.values():
            for _ in range(5):
                model.extract(IMAGES[0], resize=1024, top_k=4096)
        torch.cuda.synchronize()
        times = {c: [] for c in configs}
        for _ in range(args.rounds):
            for c, model in models.items():
                times[c].append(time_extract(model))
        for c, model in models.items():
            print(json.dumps({"config": c, "extract_ms": round(min(times[c]), 2),
                              "rounds": [round(t, 2) for t in times[c]], **compare(reference, model)}))


if __name__ == "__main__":
    main()
