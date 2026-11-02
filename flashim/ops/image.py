"""Image decoding: libjpeg's "fancy" chroma upsampling of a decode whose subsampled chroma was replicated (nvJPEG).

``jpeg_fancy_upsample(rgb, v_factor)``: ``[3, H, W]`` uint8 RGB decoded from a JPEG with 2x horizontally subsampled
chroma (``v_factor`` 1 = 4:2:2, 2 = 4:2:0) by pixel replication -> the RGB libjpeg-turbo (PIL) decodes, which upsamples
chroma with a triangle filter (``h2v1_fancy_upsample`` / ``h2v2_fancy_upsample`` in jdsample.c). Every pixel of a
replicated chroma block carries the block's ``(Cb, Cr)`` in ``(R - G, B - G)``, an injective function of ``(Cb, Cr)``
under libjpeg's integer YCbCr -> RGB tables (jdcolor.c), so the subsampled planes come back exactly from any pixel
with no channel clipped to 0 / 255; luma is an unclipped channel minus its chroma term. Blocks (pixels) where every
pixel (channel) clips use libjpeg's forward transform (jccolor.c) of the block's (pixel's) colour instead: the only
inexact part. Integer arithmetic throughout, so impls match the reference bit for bit.
"""

from __future__ import annotations

import torch
from torch import Tensor

from flashim.ops._registry import Op


def _fix(x: float) -> int:
    return int(x * (1 << 16) + 0.5)


# libjpeg's fixed-point (16-bit) colour conversion constants
CR_R, CB_B, CR_G, CB_G = _fix(1.40200), _fix(1.77200), _fix(0.71414), _fix(0.34414)
Y_R, Y_G, Y_B = _fix(0.29900), _fix(0.58700), _fix(0.11400)
CB_R, CB_GF, CR_GF, CR_BF = _fix(0.16874), _fix(0.33126), _fix(0.41869), _fix(0.08131)  # forward; B / R: FIX(0.5)
# (R - G, B - G) over every (Cb, Cr): ranges of the inverse table's index
D1_MIN, D2_MIN, D1_SPAN, D2_SPAN = -314, -362, 627, 722
_INVERSE: dict[torch.device, Tensor] = {}


def _cr_r(c):
    return (CR_R * (c - 128) + (1 << 15)) >> 16


def _cb_b(c):
    return (CB_B * (c - 128) + (1 << 15)) >> 16


def _g_term(cb, cr):
    return (-CB_G * (cb - 128) + (1 << 15) - CR_G * (cr - 128)) >> 16


def inverse_table(device: torch.device) -> Tensor:
    """int32 ``[D1_SPAN * D2_SPAN]``: ``Cb << 8 | Cr`` at ``(R - G - D1_MIN) * D2_SPAN + (B - G - D2_MIN)``, else -1."""
    if device not in _INVERSE:
        cb, cr = torch.meshgrid(torch.arange(256), torch.arange(256), indexing="ij")
        g = _g_term(cb, cr)
        key = (_cr_r(cr) - g - D1_MIN) * D2_SPAN + (_cb_b(cb) - g - D2_MIN)
        table = torch.full((D1_SPAN * D2_SPAN,), -1, dtype=torch.int32)
        table[key.flatten()] = (cb * 256 + cr).flatten().int()
        _INVERSE[device] = table.to(device)
    return _INVERSE[device]


def _shifted(c: Tensor, dim: int) -> tuple[Tensor, Tensor]:
    """The previous / next sample along ``dim``, edges replicated."""
    n = c.shape[dim]
    return (torch.cat([c.narrow(dim, 0, 1), c.narrow(dim, 0, n - 1)], dim),
            torch.cat([c.narrow(dim, 1, n - 1), c.narrow(dim, n - 1, 1)], dim))


def _interleave(even: Tensor, odd: Tensor, dim: int) -> Tensor:
    """``even[0], odd[0], even[1], ...`` along (negative) ``dim``."""
    return torch.stack([even, odd], dim).flatten(dim - 1, dim)


def _fancy(c: Tensor, v_factor: int) -> Tensor:
    """libjpeg's ``h2v1_fancy_upsample`` (``[m, n]`` -> ``[m, 2n]``) or ``h2v2_fancy_upsample`` (-> ``[2m, 2n]``); its
    first / last column special cases equal the general formula with the edge sample replicated."""
    if v_factor == 2:
        above, below = _shifted(c, -2)
        c = _interleave(3 * c + above, 3 * c + below, -2)
    (even, odd), shift = ((8, 7), 4) if v_factor == 2 else ((1, 2), 2)
    left, right = _shifted(c, -1)
    return _interleave((3 * c + left + even) >> shift, (3 * c + right + odd) >> shift, -1)


def _pad_edge(t: Tensor, rows: int, cols: int) -> Tensor:
    """Replicate the last ``rows`` / ``cols`` of ``[..., H, W]`` (F.pad's replicate mode has no integer kernels)."""
    if rows:
        t = torch.cat([t, t[..., -1:, :].expand(*t.shape[:-2], rows, t.shape[-1])], -2)
    if cols:
        t = torch.cat([t, t[..., -1:].expand(*t.shape[:-1], cols)], -1)
    return t


def _jpeg_fancy_upsample_reference(rgb: Tensor, v_factor: int) -> Tensor:
    """``[3, H, W]`` uint8 with replicated chroma (``v_factor`` 1 = 4:2:2, 2 = 4:2:0) -> ``[3, H, W]`` uint8 with
    libjpeg's fancy chroma upsampling."""
    _, h, w = rgb.shape
    x = rgb.int()
    r, g, b = x.unbind(0)
    inside = (x > 0) & (x < 255)
    key = (r - g - D1_MIN) * D2_SPAN + (b - g - D2_MIN)
    packed = torch.where(inside.all(0), inverse_table(rgb.device)[key], -1)
    hb, wb = -(-h // v_factor), -(-w // 2)
    pad = (hb * v_factor - h, wb * 2 - w)
    block = _pad_edge(packed, *pad).view(hb, v_factor, wb, 2).amax((1, 3))
    sums = _pad_edge(x, *pad).view(3, hb, v_factor, wb, 2).sum((2, 4))
    rs, gs, bs = sums.unbind(0)
    n_shift = 16 + v_factor  # block of 2 * v_factor pixels: log2 = v_factor
    offset = ((128 << 16) + (1 << 15) - 1) << v_factor
    cb_est = ((-CB_R * rs - CB_GF * gs + (1 << 15) * bs + offset) >> n_shift).clamp(0, 255)
    cr_est = (((1 << 15) * rs - CR_GF * gs - CR_BF * bs + offset) >> n_shift).clamp(0, 255)
    exact = block >= 0
    cb, cr = torch.where(exact, block >> 8, cb_est), torch.where(exact, block & 0xFF, cr_est)
    cb_rep = cb.repeat_interleave(v_factor, 0).repeat_interleave(2, 1)[:h, :w]
    cr_rep = cr.repeat_interleave(v_factor, 0).repeat_interleave(2, 1)[:h, :w]
    y_est = (Y_R * r + Y_G * g + Y_B * b + (1 << 15)) >> 16
    y = torch.where(inside[1], g - _g_term(cb_rep, cr_rep),
                    torch.where(inside[0], r - _cr_r(cr_rep), torch.where(inside[2], b - _cb_b(cb_rep), y_est)))
    cb, cr = _fancy(cb, v_factor)[:h, :w], _fancy(cr, v_factor)[:h, :w]
    return torch.stack([y + _cr_r(cr), y + _g_term(cb, cr), y + _cb_b(cb)]).clamp_(0, 255).to(torch.uint8)


jpeg_fancy_upsample = Op("jpeg_fancy_upsample", _jpeg_fancy_upsample_reference)
