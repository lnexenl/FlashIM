"""Fused multi-scale feature fusion for ALIKED's ``dense()``.

Replaces: 3 bilinear upsamples (``F.interpolate``, ``align_corners=True``) + a channel concat + an FP32
cast + the score head's first layer (1x1 conv ``4C -> 8`` + SELU) + an L2 normalize (the descriptor feature map)
with one kernel and one pass. Safe because ``C = dim // 4 <= 32`` for every released ALIKED / RaCo variant, so one
program already holds all ``4 * C`` channels of its pixel tile in registers: the L2 reduction and the 1x1 conv
(FP32-accurate ``tf32x3`` dot; the score head is a protected-precision module) are free once they are gathered, and
the 512 MB FP32 ``[4C, H, W]`` concat is never materialized (the head reads back only its 8-channel result).
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl
from torch import Tensor


@triton.jit
def _multiscale_fuse_kernel(
    f1_ptr,
    f2_ptr,
    f3_ptr,
    f4_ptr,
    head_w_ptr,
    head_out_ptr,
    feature_ptr,
    C,
    H,
    W,
    H2,
    W2,
    H3,
    W3,
    H4,
    W4,
    EPS,
    BLOCK_PIX: tl.constexpr,
    BLOCK_C: tl.constexpr,
    OUT_DTYPE: tl.constexpr,
    WRITE_HEAD: tl.constexpr,
    WRITE_FEATURE: tl.constexpr,
):
    """``f1`` ``[C, H, W]`` copied as-is; ``f2``/``f3``/``f4`` (coarser, same ``C``) bilinearly upsampled
    (``align_corners=True``) to ``[H, W]``. Writes the concatenated ``[4*C, H, W]`` FP32 (for ``score_head``)
    and its L2-normalized cast to ``OUT_DTYPE`` (the descriptor feature map). Batch size 1 (ALIKED's
    ``extract`` contract); requires ``4 * C <= BLOCK_C`` (true for every released variant, dim//4 <= 32)."""
    pid_pix = tl.program_id(0)
    pix = pid_pix * BLOCK_PIX + tl.arange(0, BLOCK_PIX)
    pix_mask = pix < H * W
    oh = (pix // W).to(tl.float32)
    ow = (pix % W).to(tl.float32)

    c = tl.arange(0, BLOCK_C)
    c_mask = c < C
    mask2d = c_mask[:, None] & pix_mask[None, :]

    v1 = tl.load(f1_ptr + c[:, None] * (H * W) + pix[None, :], mask=mask2d, other=0.0).to(tl.float32)

    iy2 = oh * (H2 - 1) / (H - 1)
    ix2 = ow * (W2 - 1) / (W - 1)
    y0_2 = tl.minimum(tl.maximum(iy2.to(tl.int32), 0), H2 - 1)
    x0_2 = tl.minimum(tl.maximum(ix2.to(tl.int32), 0), W2 - 1)
    y1_2 = tl.minimum(y0_2 + 1, H2 - 1)
    x1_2 = tl.minimum(x0_2 + 1, W2 - 1)
    ly2 = (iy2 - y0_2.to(tl.float32))
    lx2 = (ix2 - x0_2.to(tl.float32))
    base2 = f2_ptr  # channels-last [H2, W2, C]: a pixel's channels are contiguous
    v2 = (
        tl.load(base2 + (y0_2 * W2 + x0_2)[None, :] * C + c[:, None], mask=mask2d, other=0.0).to(tl.float32) * (1 - ly2)[None, :] * (1 - lx2)[None, :]
        + tl.load(base2 + (y0_2 * W2 + x1_2)[None, :] * C + c[:, None], mask=mask2d, other=0.0).to(tl.float32) * (1 - ly2)[None, :] * lx2[None, :]
        + tl.load(base2 + (y1_2 * W2 + x0_2)[None, :] * C + c[:, None], mask=mask2d, other=0.0).to(tl.float32) * ly2[None, :] * (1 - lx2)[None, :]
        + tl.load(base2 + (y1_2 * W2 + x1_2)[None, :] * C + c[:, None], mask=mask2d, other=0.0).to(tl.float32) * ly2[None, :] * lx2[None, :]
    )

    iy3 = oh * (H3 - 1) / (H - 1)
    ix3 = ow * (W3 - 1) / (W - 1)
    y0_3 = tl.minimum(tl.maximum(iy3.to(tl.int32), 0), H3 - 1)
    x0_3 = tl.minimum(tl.maximum(ix3.to(tl.int32), 0), W3 - 1)
    y1_3 = tl.minimum(y0_3 + 1, H3 - 1)
    x1_3 = tl.minimum(x0_3 + 1, W3 - 1)
    ly3 = (iy3 - y0_3.to(tl.float32))
    lx3 = (ix3 - x0_3.to(tl.float32))
    base3 = f3_ptr  # channels-last [H3, W3, C]: a pixel's channels are contiguous
    v3 = (
        tl.load(base3 + (y0_3 * W3 + x0_3)[None, :] * C + c[:, None], mask=mask2d, other=0.0).to(tl.float32) * (1 - ly3)[None, :] * (1 - lx3)[None, :]
        + tl.load(base3 + (y0_3 * W3 + x1_3)[None, :] * C + c[:, None], mask=mask2d, other=0.0).to(tl.float32) * (1 - ly3)[None, :] * lx3[None, :]
        + tl.load(base3 + (y1_3 * W3 + x0_3)[None, :] * C + c[:, None], mask=mask2d, other=0.0).to(tl.float32) * ly3[None, :] * (1 - lx3)[None, :]
        + tl.load(base3 + (y1_3 * W3 + x1_3)[None, :] * C + c[:, None], mask=mask2d, other=0.0).to(tl.float32) * ly3[None, :] * lx3[None, :]
    )

    iy4 = oh * (H4 - 1) / (H - 1)
    ix4 = ow * (W4 - 1) / (W - 1)
    y0_4 = tl.minimum(tl.maximum(iy4.to(tl.int32), 0), H4 - 1)
    x0_4 = tl.minimum(tl.maximum(ix4.to(tl.int32), 0), W4 - 1)
    y1_4 = tl.minimum(y0_4 + 1, H4 - 1)
    x1_4 = tl.minimum(x0_4 + 1, W4 - 1)
    ly4 = (iy4 - y0_4.to(tl.float32))
    lx4 = (ix4 - x0_4.to(tl.float32))
    base4 = f4_ptr  # channels-last [H4, W4, C]: a pixel's channels are contiguous
    v4 = (
        tl.load(base4 + (y0_4 * W4 + x0_4)[None, :] * C + c[:, None], mask=mask2d, other=0.0).to(tl.float32) * (1 - ly4)[None, :] * (1 - lx4)[None, :]
        + tl.load(base4 + (y0_4 * W4 + x1_4)[None, :] * C + c[:, None], mask=mask2d, other=0.0).to(tl.float32) * (1 - ly4)[None, :] * lx4[None, :]
        + tl.load(base4 + (y1_4 * W4 + x0_4)[None, :] * C + c[:, None], mask=mask2d, other=0.0).to(tl.float32) * ly4[None, :] * (1 - lx4)[None, :]
        + tl.load(base4 + (y1_4 * W4 + x1_4)[None, :] * C + c[:, None], mask=mask2d, other=0.0).to(tl.float32) * ly4[None, :] * lx4[None, :]
    )

    if WRITE_HEAD:
        o = tl.arange(0, 16)
        o_mask = o < 8
        wmask = c_mask[:, None] & o_mask[None, :]
        w1 = tl.load(head_w_ptr + o[None, :] * (4 * C) + (0 * C + c)[:, None], mask=wmask, other=0.0)
        w2 = tl.load(head_w_ptr + o[None, :] * (4 * C) + (1 * C + c)[:, None], mask=wmask, other=0.0)
        w3 = tl.load(head_w_ptr + o[None, :] * (4 * C) + (2 * C + c)[:, None], mask=wmask, other=0.0)
        w4 = tl.load(head_w_ptr + o[None, :] * (4 * C) + (3 * C + c)[:, None], mask=wmask, other=0.0)
        acc = tl.dot(tl.trans(v1), w1, input_precision="tf32x3")
        acc = tl.dot(tl.trans(v2), w2, acc, input_precision="tf32x3")
        acc = tl.dot(tl.trans(v3), w3, acc, input_precision="tf32x3")
        acc = tl.dot(tl.trans(v4), w4, acc, input_precision="tf32x3")
        acc = 1.0507009873554805 * tl.where(acc > 0, acc, 1.6732632423543772 * (tl.exp(acc) - 1.0))  # SELU
        tl.store(head_out_ptr + o[None, :] * (H * W) + pix[:, None], acc, mask=pix_mask[:, None] & o_mask[None, :])
    if WRITE_FEATURE:
        sumsq = tl.sum(tl.where(c_mask[:, None], v1 * v1 + v2 * v2 + v3 * v3 + v4 * v4, 0.0), axis=0)
        inv_norm = 1.0 / tl.maximum(tl.sqrt(sumsq), EPS)  # [BLOCK_PIX]; matches F.normalize(p=2, dim=1)

        tl.store(feature_ptr + (0 * C + c)[:, None] * (H * W) + pix[None, :], (v1 * inv_norm[None, :]).to(OUT_DTYPE), mask=mask2d)
        tl.store(feature_ptr + (1 * C + c)[:, None] * (H * W) + pix[None, :], (v2 * inv_norm[None, :]).to(OUT_DTYPE), mask=mask2d)
        tl.store(feature_ptr + (2 * C + c)[:, None] * (H * W) + pix[None, :], (v3 * inv_norm[None, :]).to(OUT_DTYPE), mask=mask2d)
        tl.store(feature_ptr + (3 * C + c)[:, None] * (H * W) + pix[None, :], (v4 * inv_norm[None, :]).to(OUT_DTYPE), mask=mask2d)


_TL_DTYPE = {torch.bfloat16: tl.bfloat16, torch.float16: tl.float16, torch.float32: tl.float32}


def multiscale_fuse(
    f1: Tensor, f2: Tensor, f3: Tensor, f4: Tensor, head_weight: Tensor | None = None, *, normalize: bool = True
) -> tuple[Tensor | None, Tensor | None]:
    """``f1`` ``[1, C, H, W]``, ``f2``/``f3``/``f4`` ``[1, C, h_i, w_i]`` (coarser, ``C <= 32``) ->
    ``(head, feature_map)``. With ``fused = cat([f1, *upsample(f2..f4)])`` (bilinear, ``align_corners=True``):
    ``head = selu(conv1x1(fused, head_weight))`` ``[1, 8, H, W]`` FP32 (``head_weight`` ``[8, 4C]`` FP32; ``None`` skips
    it) and ``feature_map = F.normalize(fused, dim=1)`` in ``f1``'s dtype (``normalize=False`` skips it)."""
    _, c, h, w = f1.shape
    if c > 32:
        raise ValueError(f"multiscale_fuse needs C = dim // 4 <= 32, got {c}")
    dummy = f1
    head = torch.empty(1, 8, h, w, device=f1.device, dtype=torch.float32) if head_weight is not None else None
    feature_map = torch.empty(1, 4 * c, h, w, device=f1.device, dtype=f1.dtype) if normalize else None
    # native dtype (the kernel upcasts on load); the coarse maps go channels-last so each gathered pixel is one
    # contiguous ``C``-wide load (NCHW made every lane of a warp hit a different cache line: L1-bound, 93%)
    f1c = f1.contiguous()
    f2c, f3c, f4c = (f.permute(0, 2, 3, 1).contiguous() for f in (f2, f3, f4))
    block_pix = 64
    grid = (triton.cdiv(h * w, block_pix),)
    _multiscale_fuse_kernel[grid](
        f1c,
        f2c,
        f3c,
        f4c,
        head_weight if head_weight is not None else dummy,
        head if head is not None else dummy,
        feature_map if feature_map is not None else dummy,
        c,
        h,
        w,
        f2.shape[-2],
        f2.shape[-1],
        f3.shape[-2],
        f3.shape[-1],
        f4.shape[-2],
        f4.shape[-1],
        1e-12,
        BLOCK_PIX=block_pix,
        BLOCK_C=32,
        OUT_DTYPE=_TL_DTYPE[f1.dtype],
        WRITE_HEAD=head_weight is not None,
        WRITE_FEATURE=normalize,
    )
    return head, feature_map
