"""Fused RaCo residual block (``selu(conv3x3(selu(conv3x3(x))) + conv1x1(x))``, BatchNorm folded, <= 16 channels).

The unfused block reads/writes its 12-channel activation ~5 times (conv, conv, shortcut, add+SELU) and is memory-bound
at ~5x the roofline. Here one persistent kernel reads the block input once and writes the output once: each program
takes a ``TILE x TILE`` output tile, computes the first conv over the tile plus a 1-pixel halo as 9 tap-wise
``tl.dot``s, parks that intermediate in a small per-program scratch (L1/L2 resident: one slot per program, reused
across tiles), and reads it back shifted for the second conv. Activations are NHWC bf16 with the channel dim padded.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl
from torch import Tensor

TH, TW = 14, 14  # output tile (rows x cols); its conv1 region (+1px halo) is 16x16 = 256 lanes (tl.dot M)
CP = 16  # channel padding (tl.dot's minimum K/N)
NUM_WARPS = 8
MAXNREG = None


@triton.jit
def _selu(x):
    return 1.0507009873554805 * tl.where(x > 0, x, 1.6732632423543772 * (tl.exp(x) - 1.0))


@triton.jit
def _resblock_kernel(
    x_ptr,
    out_ptr,
    scratch_ptr,
    w1_ptr,
    b1_ptr,
    w2_ptr,
    b2_ptr,
    wm_ptr,
    bm_ptr,
    H,
    W,
    tiles_x,
    num_tiles,
    IN_CH: tl.constexpr,  # channels per pixel in x (>= CIN, 4 or 16)
    CIN: tl.constexpr,
    OUT_CH: tl.constexpr,  # channels per pixel in out (16)
    TH: tl.constexpr,
    TW: tl.constexpr,
    CP: tl.constexpr,
    LANES_: tl.constexpr,
):
    pid = tl.program_id(0)
    nprog = tl.num_programs(0)
    RW: tl.constexpr = TW + 2
    RH: tl.constexpr = TH + 2
    LANES: tl.constexpr = LANES_
    q = tl.arange(0, LANES)
    ty = q // RW
    tx = q % RW
    ci = tl.arange(0, CP)
    co = tl.arange(0, CP)
    b1 = tl.load(b1_ptr + co).to(tl.float32)
    b2 = tl.load(b2_ptr + co).to(tl.float32)
    bm = tl.load(bm_ptr + co).to(tl.float32)
    slot = scratch_ptr + pid * LANES * CP

    for tile in range(pid, num_tiles, nprog):
        oy0 = (tile // tiles_x) * TH
        ox0 = (tile % tiles_x) * TW
        # -- conv1 + SELU over the (TILE+2)^2 region ---------------------------------------------------------------
        py = oy0 - 1 + ty
        px = ox0 - 1 + tx
        acc = tl.zeros((LANES, CP), dtype=tl.float32)
        for tap in tl.static_range(9):
            iy = py + (tap // 3 - 1)
            ix = px + (tap % 3 - 1)
            ok = (iy >= 0) & (iy < H) & (ix >= 0) & (ix < W) & (ty < RH) & (tx < RW)
            a = tl.load(x_ptr + (iy * W + ix)[:, None] * IN_CH + ci[None, :], mask=ok[:, None] & ((ci[None, :] < CIN) | (IN_CH == CP)), other=0.0)
            w = tl.load(w1_ptr + tap * CP * CP + ci[:, None] * CP + co[None, :])
            acc += tl.dot(a, w)
        mid = _selu(acc + b1[None, :])
        inside = (py >= 0) & (py < H) & (px >= 0) & (px < W)  # zero padding of conv2's input
        mid = tl.where(inside[:, None], mid, 0.0).to(tl.bfloat16)
        tl.store(slot + q[:, None] * CP + co[None, :], mid, mask=(ty < RH)[:, None])
        tl.debug_barrier()
        # -- conv2 + 1x1 shortcut + SELU on the TILE^2 outputs -------------------------------------------------------
        oy = oy0 + ty
        ox = ox0 + tx
        live = (ty < TH) & (tx < TW) & (oy < H) & (ox < W)
        acc2 = tl.zeros((LANES, CP), dtype=tl.float32)
        for tap in tl.static_range(9):
            r = (ty + 1 + (tap // 3 - 1)) * RW + (tx + 1 + (tap % 3 - 1))
            m = tl.load(slot + r[:, None] * CP + ci[None, :], mask=live[:, None], other=0.0)
            w = tl.load(w2_ptr + tap * CP * CP + ci[:, None] * CP + co[None, :])
            acc2 += tl.dot(m, w)
        xs = tl.load(x_ptr + (oy * W + ox)[:, None] * IN_CH + ci[None, :], mask=live[:, None] & ((ci[None, :] < CIN) | (IN_CH == CP)), other=0.0)
        acc2 += tl.dot(xs, tl.load(wm_ptr + ci[:, None] * CP + co[None, :]))
        out = _selu(acc2 + b2[None, :] + bm[None, :])
        tl.store(out_ptr + (oy * W + ox)[:, None] * OUT_CH + co[None, :], out.to(tl.bfloat16), mask=live[:, None])
        tl.debug_barrier()  # the scratch slot is reused by this program's next tile


def pack_block(conv1: Tensor, conv2: Tensor, match: Tensor) -> tuple[Tensor, ...]:
    """Pad and lay out a (BatchNorm-folded) residual block's weights as ``[tap, cin, cout]`` bf16 (``CP`` wide)."""
    def taps(conv: Tensor) -> Tensor:
        out = torch.zeros(9, CP, CP, device=conv.device, dtype=torch.bfloat16)
        out[:, : conv.shape[1], : conv.shape[0]] = conv.detach().permute(2, 3, 1, 0).reshape(9, conv.shape[1], conv.shape[0]).to(torch.bfloat16)
        return out.contiguous()

    shortcut = torch.zeros(CP, CP, device=match.device, dtype=torch.bfloat16)
    shortcut[: match.shape[1], : match.shape[0]] = match.detach()[:, :, 0, 0].t().to(torch.bfloat16)
    return taps(conv1), taps(conv2), shortcut


def pad_bias(bias: Tensor) -> Tensor:
    out = torch.zeros(CP, device=bias.device, dtype=torch.float32)
    out[: bias.numel()] = bias.detach().float()
    return out


def resblock(x: Tensor, cin: int, packed: tuple[Tensor, ...], scratch: Tensor | None = None) -> tuple[Tensor, Tensor]:
    """``x``: NHWC bf16 ``[H, W, C_IN]`` (``C_IN`` in {4, 16} channels per pixel, ``cin`` valid). ``packed``:
    ``(w1, b1, w2, b2, wm, bm)`` from :func:`pack_block` / :func:`pad_bias`. Returns ``(out [H, W, 16] bf16, scratch)``."""
    h, w, in_ch = x.shape
    w1, b1, w2, b2, wm, bm = packed
    LANES = triton.next_power_of_2((TH + 2) * (TW + 2))
    tiles_y, tiles_x = triton.cdiv(h, TH), triton.cdiv(w, TW)
    num_tiles = tiles_y * tiles_x
    nprog = min(num_tiles, 4 * torch.cuda.get_device_properties(x.device).multi_processor_count)
    if scratch is None or scratch.numel() < nprog * LANES * CP:
        scratch = torch.empty(nprog * LANES * CP, device=x.device, dtype=torch.bfloat16)
    out = torch.empty(h, w, CP, device=x.device, dtype=torch.bfloat16)
    _resblock_kernel[(nprog,)](
        x, out, scratch, w1, b1, w2, b2, wm, bm, h, w, tiles_x, num_tiles,
        IN_CH=in_ch, CIN=cin, OUT_CH=CP, TH=TH, TW=TW, CP=CP, LANES_=LANES, num_warps=NUM_WARPS, maxnreg=MAXNREG,
    )
    return out, scratch


@triton.jit
def _head5x5_kernel(x_ptr, w_ptr, out_ptr, bias, H, W, BLOCK: tl.constexpr, CP: tl.constexpr):
    """5x5 conv, ``CP`` (zero-padded) channels -> 1, reflect padding, on NHWC bf16; FP32 ``[H, W]`` out."""
    pid = tl.program_id(0)
    pix = pid * BLOCK + tl.arange(0, BLOCK)
    live = pix < H * W
    y = pix // W
    x = pix % W
    c = tl.arange(0, CP)
    acc = tl.zeros((BLOCK,), dtype=tl.float32)
    for tap in tl.static_range(25):
        iy = y + (tap // 5 - 2)
        ix = x + (tap % 5 - 2)
        iy = tl.where(iy < 0, -iy, tl.where(iy >= H, 2 * (H - 1) - iy, iy))
        ix = tl.where(ix < 0, -ix, tl.where(ix >= W, 2 * (W - 1) - ix, ix))
        a = tl.load(x_ptr + (iy * W + ix)[:, None] * CP + c[None, :], mask=live[:, None], other=0.0).to(tl.float32)
        acc += tl.sum(a * tl.load(w_ptr + tap * CP + c).to(tl.float32)[None, :], axis=1)
    tl.store(out_ptr + pix, acc + bias, mask=live)


def pack_head(conv: Tensor) -> tuple[Tensor, float]:
    """``conv``: ``Conv2d(C<=16, 1, 5)`` -> tap-major ``[25, CP]`` FP32 weights and the bias."""
    weight = torch.zeros(25, CP, device=conv.weight.device, dtype=torch.float32)
    weight[:, : conv.in_channels] = conv.weight.detach()[0].permute(1, 2, 0).reshape(25, conv.in_channels).float()
    return weight, float(conv.bias.detach().item())


def head5x5(x: Tensor, packed: tuple[Tensor, float]) -> Tensor:
    """``x``: NHWC bf16 ``[H, W, 16]`` -> FP32 ``[1, 1, H, W]``."""
    h, w, _ = x.shape
    weight, bias = packed
    out = torch.empty(h * w, device=x.device, dtype=torch.float32)
    _head5x5_kernel[(triton.cdiv(h * w, 256),)](x, weight, out, bias, h, w, BLOCK=256, CP=CP, num_warps=4)
    return out.view(1, 1, h, w)
