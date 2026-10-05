"""ALIKED / RaCo Triton kernels.

``deform_patches_int8``: fused deformable-conv patch gather + static INT8 quantize (one kernel, no FP32
intermediate).

``multiscale_fuse``: fused multi-scale feature fusion for ALIKED's ``dense()``. Replaces: 3 bilinear upsamples
(``F.interpolate``, ``align_corners=True``) + a channel concat + an FP32 cast + the score head's first layer (1x1 conv ``4C -> 8`` + SELU) + an L2 normalize (the descriptor feature map)
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
from triton.language.extra import libdevice

from flashim import ops
from flashim.ops._registry import DeviceSpec
from flashim.platforms import Feature
from flashim.quant.tensor import Int8Tensor


@triton.jit
def _dcn_gather_quantize_kernel(
    x_ptr,
    offset_ptr,
    out_ptr,
    inv_scale_ptr,
    C,
    H,
    W,
    OH,
    OW,
    stride_xb,
    stride_xc,
    stride_xh,
    stride_xw,
    stride_ob,
    stride_oc,
    K: tl.constexpr,
    PADDING: tl.constexpr,
    BLOCK_PIX: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    pid_pix = tl.program_id(0)
    pid_c = tl.program_id(1)
    b = tl.program_id(2)

    pix = pid_pix * BLOCK_PIX + tl.arange(0, BLOCK_PIX)
    pix_mask = pix < OH * OW
    oh = (pix // OW).to(tl.float32)
    ow = (pix % OW).to(tl.float32)

    c = pid_c * BLOCK_C + tl.arange(0, BLOCK_C)
    c_mask = c < C

    inv_scale = tl.load(inv_scale_ptr)  # a device scalar: no host sync to read the scale
    x_base = x_ptr + b * stride_xb
    off_base = offset_ptr + b * stride_ob

    for tap in range(K * K):
        kh = tap // K
        kw = tap % K
        # offset is contiguous [B, 2*K*K, OH, OW]: its spatial plane is row-major, so the flat pixel
        # index `pix` (= oh * OW + ow) already is the offset within one channel.
        off_h_ptr = off_base + (2 * tap) * stride_oc + pix
        off_w_ptr = off_base + (2 * tap + 1) * stride_oc + pix
        dy = tl.load(off_h_ptr, mask=pix_mask, other=0.0)
        dx = tl.load(off_w_ptr, mask=pix_mask, other=0.0)

        sample_h = oh - PADDING + kh + dy
        sample_w = ow - PADDING + kw + dx
        h0 = tl.floor(sample_h)
        w0 = tl.floor(sample_w)
        lh = sample_h - h0
        lw = sample_w - w0
        h0i = h0.to(tl.int32)
        w0i = w0.to(tl.int32)

        acc = tl.zeros((BLOCK_C, BLOCK_PIX), dtype=tl.float32)
        for dh in range(2):
            for dw in range(2):
                hi = h0i + dh
                wi = w0i + dw
                valid = (hi >= 0) & (hi < H) & (wi >= 0) & (wi < W) & pix_mask
                hi_c = tl.minimum(tl.maximum(hi, 0), H - 1)
                wi_c = tl.minimum(tl.maximum(wi, 0), W - 1)
                weight = tl.where(dh == 0, 1.0 - lh, lh) * tl.where(dw == 0, 1.0 - lw, lw)
                ptrs = x_base + c[:, None] * stride_xc + hi_c[None, :] * stride_xh + wi_c[None, :] * stride_xw
                mask2d = c_mask[:, None] & valid[None, :]
                val = tl.load(ptrs, mask=mask2d, other=0.0).to(tl.float32)
                acc += val * weight[None, :]

        scaled = libdevice.rint(acc * inv_scale)  # reciprocal multiply: exact division is ~2x slower
        quantized = tl.minimum(tl.maximum(scaled, -127.0), 127.0).to(tl.int8)
        out_col = c * (K * K) + tap  # [BLOCK_C]
        row = b * (OH * OW) + pix  # [BLOCK_PIX]
        out_ptrs = out_ptr + row[None, :] * (C * K * K) + out_col[:, None]
        tl.store(out_ptrs, quantized, mask=c_mask[:, None] & pix_mask[None, :])


def deform_patches_int8(x: Tensor, offset: Tensor, kernel_size: int, padding: int, scale: Tensor) -> Int8Tensor:
    """``x``: ``[B, C, H, W]`` FP32 (any layout; converted to channels-last so the per-pixel channel gather
    is coalesced — see the module docstring); ``offset``: ``[B, 2 * K * K, OH, OW]`` contiguous FP32.
    Returns an :class:`Int8Tensor` of logical shape ``[B, OH, OW, C * K * K]``."""
    x = x.contiguous(memory_format=torch.channels_last)
    b, c, h, w = x.shape
    oh, ow = offset.shape[-2:]
    out = torch.empty(b * oh * ow, c * kernel_size * kernel_size, device=x.device, dtype=torch.int8)
    scale = scale.reshape(())
    inv_scale = (1.0 / scale.float().clamp_min(torch.finfo(torch.float32).eps)).reshape(1)
    block_pix, block_c = 32, min(32, triton.next_power_of_2(c))
    grid = (triton.cdiv(oh * ow, block_pix), triton.cdiv(c, block_c), b)
    _dcn_gather_quantize_kernel[grid](
        x,
        offset,
        out,
        inv_scale,
        c,
        h,
        w,
        oh,
        ow,
        x.stride(0),
        x.stride(1),
        x.stride(2),
        x.stride(3),
        offset.stride(0),
        offset.stride(1),
        K=kernel_size,
        PADDING=padding,
        BLOCK_PIX=block_pix,
        BLOCK_C=block_c,
    )
    return Int8Tensor(out.view(b, oh, ow, c * kernel_size * kernel_size), scale.float())


# -- multi-scale fusion ------------------------------------------------------------------------------


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

# -- registrations ---------------------------------------------------------------------------------------

def _dcn_eligible(x: Tensor, offset: Tensor, kernel_size: int, padding: int, scale: Tensor) -> bool:
    # x's layout doesn't matter: the kernel converts to channels-last itself (coalesced channel gather).
    return offset.is_contiguous() and x.dtype == offset.dtype == torch.float32


ops.deform_patches_int8.register(
    "triton", device=DeviceSpec("cuda"), requires=[Feature.TRITON], when=_dcn_eligible
)(deform_patches_int8)


def _multiscale_fuse_eligible(
    f1: Tensor, f2: Tensor, f3: Tensor, f4: Tensor, head_weight: Tensor | None = None, *, normalize: bool = True
) -> bool:
    ok = f1.shape[0] == 1 and f1.shape[1] <= 32  # batch size 1 (the sparse extract contract); dim // 4 <= 32
    return ok and (head_weight is None or (head_weight.dtype == torch.float32 and head_weight.is_contiguous()))


ops.multiscale_fuse.register(
    "triton", device=DeviceSpec("cuda"), requires=[Feature.TRITON], when=_multiscale_fuse_eligible
)(multiscale_fuse)
