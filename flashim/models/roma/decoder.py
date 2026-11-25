# Portions adapted from RoMa (MIT, Copyright (c) 2023 Johan Edstedt); see THIRD_PARTY_NOTICES.md.
"""RoMa (v1) coarse Gaussian-Process + Transformer matcher and multi-scale convolutional warp refiners."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from flashim import ops
from flashim.layers import IndexedSeparableConvBlock, TransformerBlock
from flashim.models.romav2.geometry import normalized_grid
from flashim.utils import grid_sample


@dataclass(frozen=True)
class RefinerSpec:
    feat_dim: int
    proj_dim: int
    disp_emb_dim: int
    local_corr_radius: int | None = None
    hidden_blocks: int = 8

    @property
    def channels(self) -> int:
        corr = 0 if self.local_corr_radius is None else (2 * self.local_corr_radius + 1) ** 2
        return 2 * self.proj_dim + self.disp_emb_dim + corr


REFINER_SPECS: dict[str, RefinerSpec] = {
    "16": RefinerSpec(feat_dim=1024, proj_dim=512, disp_emb_dim=128, local_corr_radius=7),
    "8": RefinerSpec(feat_dim=512, proj_dim=512, disp_emb_dim=64, local_corr_radius=3),
    "4": RefinerSpec(feat_dim=256, proj_dim=256, disp_emb_dim=32, local_corr_radius=2),
    "2": RefinerSpec(feat_dim=128, proj_dim=64, disp_emb_dim=16, local_corr_radius=None),
    "1": RefinerSpec(feat_dim=64, proj_dim=9, disp_emb_dim=6, local_corr_radius=None),
}
SCALES = ("16", "8", "4", "2", "1")
UPSAMPLE_SCALES = ("8", "4", "2", "1")


def cls_to_flow_refine(cls: Tensor) -> Tensor:
    """``[B, R^2, H, W]`` coarse coordinate logits -> ``[B, 2, H, W]`` sub-bin refined normalized warp."""
    _, c, _, _ = cls.shape
    res = round(math.sqrt(c))
    grid = normalized_grid(1, res, res, cls.device).reshape(c, 2)
    probs = cls.float().softmax(dim=1)
    mode = probs.max(dim=1).indices
    index = torch.stack((mode - 1, mode, mode + 1, mode - res, mode + res), dim=1).clamp(0, c - 1).long()
    neighbours = torch.gather(probs, dim=1, index=index)[..., None]
    flow = (
        neighbours[:, 0] * grid[index[:, 0]]
        + neighbours[:, 1] * grid[index[:, 1]]
        + neighbours[:, 2] * grid[index[:, 2]]
        + neighbours[:, 3] * grid[index[:, 3]]
        + neighbours[:, 4] * grid[index[:, 4]]
    )
    return (flow / neighbours.sum(dim=1)).permute(0, 3, 1, 2)


def cls_to_flow_refine_hwc(cls: Tensor) -> Tensor:
    """``[B, H, W, R^2]`` contiguous logits -> ``[B, 2, H, W]`` warp using fast last-dim warp-shuffle softmax."""
    c = cls.shape[-1]
    res = round(math.sqrt(c))
    grid = normalized_grid(1, res, res, cls.device).reshape(c, 2)
    probs = cls.float().softmax(dim=-1)
    mode = probs.argmax(dim=-1)
    index = torch.stack((mode - 1, mode, mode + 1, mode - res, mode + res), dim=-1).clamp(0, c - 1)
    neighbours = torch.gather(probs, dim=-1, index=index)
    flow = (
        neighbours[..., 0:1] * grid[index[..., 0]]
        + neighbours[..., 1:2] * grid[index[..., 1]]
        + neighbours[..., 2:3] * grid[index[..., 2]]
        + neighbours[..., 3:4] * grid[index[..., 3]]
        + neighbours[..., 4:5] * grid[index[..., 4]]
    )
    return (flow / neighbours.sum(dim=-1, keepdim=True)).permute(0, 3, 1, 2).contiguous()


class CosKernel(nn.Module):
    """Cosine-similarity exponential kernel ``exp((cos(x, y) - 1) / T)``."""

    def __init__(self, temperature: float = 0.2) -> None:
        super().__init__()
        self.temperature = temperature

    def forward(self, x: Tensor, y: Tensor, eps: float = 1e-6) -> Tensor:
        c = torch.einsum("bnd,bmd->bnm", x, y) / (x.norm(dim=-1)[..., None] * y.norm(dim=-1)[:, None] + eps)
        # Tensor divisor (not Python float) keeps IEEE div(c - 1, T) instead of mul(c - 1, 1/T), bit-exact to upstream,
        # and torch.full allocates directly on-device (no H2D copy, CUDA-graph safe).
        t = torch.full((), self.temperature, device=c.device, dtype=c.dtype)
        return ((c - 1.0) / t).exp()


class GP(nn.Module):
    """Gaussian Process posterior mean over Fourier coordinate embeddings (scale 16, always FP32)."""

    def __init__(self, gp_dim: int = 512, temperature: float = 0.2, sigma_noise: float = 0.1) -> None:
        super().__init__()
        self.K = CosKernel(temperature=temperature)
        self.sigma_noise = sigma_noise
        self.pos_conv = nn.Conv2d(2, gp_dim, 1, 1)
        self.dim = gp_dim

    def get_pos_enc(self, y: Tensor) -> Tensor:
        b, _, h, w = y.shape
        coords = normalized_grid(b, h, w, y.device).permute(0, 3, 1, 2).contiguous()
        return torch.cos(8 * math.pi * self.pos_conv(coords.to(self.pos_conv.weight.dtype)).float())

    def forward(self, x: Tensor, y: Tensor) -> Tensor:
        b, _, h1, w1 = x.shape
        _, _, h2, w2 = y.shape
        f = self.get_pos_enc(y).flatten(2).transpose(1, 2).contiguous()
        x_f = x.float().flatten(2).transpose(1, 2).contiguous()
        y_f = y.float().flatten(2).transpose(1, 2).contiguous()
        k_yy = self.K(y_f, y_f)
        k_xy = self.K(x_f, y_f)
        noise = self.sigma_noise * torch.eye(h2 * w2, device=x.device, dtype=torch.float32)[None]
        chol = torch.linalg.cholesky(k_yy + noise)
        pos_emb = torch.cholesky_solve(f, chol, upper=False)
        mu_x = k_xy @ pos_emb
        return mu_x.transpose(1, 2).reshape(b, self.dim, h1, w1).contiguous()


class TransformerDecoder(nn.Module):
    """5-block pre-norm Transformer (``qkv_bias=False``, ``eps=1e-5``) + 64x64+1 coordinate/certainty classifier."""

    def __init__(self, hidden_dim: int = 1024, num_heads: int = 8, depth: int = 5, cls_res: int = 64) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.out_dim = cls_res * cls_res + 1
        self.blocks = nn.Sequential(
            *(TransformerBlock(hidden_dim, num_heads, eps=1e-5, qkv_bias=False) for _ in range(depth))
        )
        self.to_out = nn.Linear(hidden_dim, self.out_dim)
        self.compute_dtype = torch.float32

    def forward(self, gp_posterior: Tensor, features: Tensor, *, hwc_cls: bool = False) -> tuple[Tensor, Tensor]:
        b, _, h, w = gp_posterior.shape
        x = torch.cat((gp_posterior, features), dim=1)
        tokens = x.flatten(2).transpose(1, 2).to(self.compute_dtype)
        z = self.blocks(tokens)
        out_dtype = self.to_out.weight.dtype if isinstance(self.to_out, nn.Linear) else self.compute_dtype
        out = self.to_out(z.to(out_dtype)).float()
        if hwc_cls:
            out_hwc = out.view(b, h, w, self.out_dim)
            return out_hwc[..., :-1], out_hwc[..., -1:].permute(0, 3, 1, 2).contiguous()
        out = out.transpose(1, 2).reshape(b, self.out_dim, h, w)
        return out[:, :-1], out[:, -1:]


class ConvRefiner(nn.Module):
    """Depthwise-separable 5x5 conv stack refining ``(flow, certainty)`` from warped features, displacement embedding,
    and optional local correlation. Uses :class:`IndexedSeparableConvBlock` so checkpoint keys match upstream
    ``block1.{0,1,3}`` / ``hidden_blocks.i.{0,1,3}`` while exposing the block type to FlashIM's passes."""

    def __init__(self, spec: RefinerSpec) -> None:
        super().__init__()
        self.spec = spec
        channels = spec.channels
        self.block1 = IndexedSeparableConvBlock(channels, kernel_size=5, momentum=0.01)
        self.hidden_blocks = nn.Sequential(
            *(IndexedSeparableConvBlock(channels, kernel_size=5, momentum=0.01) for _ in range(spec.hidden_blocks))
        )
        self.out_conv = nn.Conv2d(channels, 3, 1, 1, 0)
        self.disp_emb = nn.Conv2d(2, spec.disp_emb_dim, 1, 1, 0)
        self.local_corr_radius = spec.local_corr_radius
        self.compute_dtype = torch.float32
        self.channels_last = False
        self.padded_channels = channels
        self.fused_io = False
        self.register_buffer("_head_weight", None, persistent=False)

    def enable_fused_io(self) -> None:
        """Enable fused Triton refiner input assembly (``ops.roma_refiner_input``) and FP32 head (``ops.roma_refiner_head``)."""
        orig = self.spec.channels
        self.register_buffer("_head_weight", self.out_conv.weight.detach()[:, :orig, 0, 0].float().contiguous(), persistent=False)
        self.fused_io = True

    def pad_channels(self, target: int) -> bool:
        """Zero-pad separable block channels from ``spec.channels`` to ``target`` (must run after BatchNorm folding).

        Mathematically exact: extra input channels are zero-padded and extra weight/bias rows and columns are zero,
        so extra feature channels stay identically zero through every block and contribute zero to ``out_conv``.
        Aligning RoMa's odd channel counts (1377/1137/569/144/24 -> 1408/1152/576/192/32) unlocks aligned BF16
        cuBLAS GEMMs, TileLang TMA depthwise producers (C % 64 == 0), cuBLAS MXFP8 / INT8 GEMMs, and the fused
        32-channel BF16 block at scale 1.
        """
        orig = self.padded_channels
        if target <= orig:
            return False

        def _pad_dw(conv: nn.Conv2d) -> nn.Conv2d:
            out = nn.Conv2d(
                target, target, conv.kernel_size, stride=conv.stride, padding=conv.padding,
                groups=target, bias=conv.bias is not None, device=conv.weight.device, dtype=conv.weight.dtype,
            )
            with torch.no_grad():
                out.weight.zero_()
                out.weight[:orig].copy_(conv.weight)
                if conv.bias is not None:
                    out.bias.zero_()
                    out.bias[:orig].copy_(conv.bias)
            return out

        def _pad_pw(conv: nn.Conv2d) -> nn.Conv2d:
            out = nn.Conv2d(
                target, target, conv.kernel_size, stride=conv.stride, padding=conv.padding,
                bias=conv.bias is not None, device=conv.weight.device, dtype=conv.weight.dtype,
            )
            with torch.no_grad():
                out.weight.zero_()
                out.weight[:orig, :orig].copy_(conv.weight)
                if conv.bias is not None:
                    out.bias.zero_()
                    out.bias[:orig].copy_(conv.bias)
            return out

        for block in (self.block1, *self.hidden_blocks):
            if not isinstance(block.norm, nn.Identity):
                raise TypeError("pad_channels requires folded BatchNorm (run fold_batchnorm first)")
            block.conv_depthwise = _pad_dw(block.conv_depthwise)
            block.conv_pointwise = _pad_pw(block.conv_pointwise)
        out_conv = nn.Conv2d(
            target, self.out_conv.out_channels, 1, 1, 0,
            bias=self.out_conv.bias is not None, device=self.out_conv.weight.device, dtype=self.out_conv.weight.dtype,
        )
        with torch.no_grad():
            out_conv.weight.zero_()
            out_conv.weight[:, :orig].copy_(self.out_conv.weight)
            if self.out_conv.bias is not None:
                out_conv.bias.copy_(self.out_conv.bias)
        self.out_conv = out_conv
        self.padded_channels = target
        return True

    def forward(
        self, x: Tensor, y: Tensor, warp: Tensor, *, scale_factor: float = 1.0, pair: int = 0
    ) -> tuple[Tensor, Tensor]:
        if self.fused_io and self.compute_dtype == torch.bfloat16:
            radius = -1 if self.local_corr_radius is None else self.local_corr_radius
            d = ops.roma_refiner_input(
                x, y, warp, self.disp_emb.weight, self.disp_emb.bias, scale_factor, radius, self.padded_channels, pair
            )
            d = self.block1(d)
            d = self.hidden_blocks(d)
            head_w = self._head_weight if self._head_weight is not None else self.out_conv.weight[:, : self.spec.channels, 0, 0]
            return ops.roma_refiner_head(d, head_w, self.out_conv.bias)
        b, _, hs, ws = x.shape
        warp_bhwc = warp.permute(0, 2, 3, 1)
        x_hat = grid_sample(y, warp_bhwc, align_corners=False, mode="bilinear")
        im_a_coords = normalized_grid(b, hs, ws, x.device).permute(0, 3, 1, 2)
        in_displacement = warp - im_a_coords
        emb = self.disp_emb(((40.0 / 32.0) * scale_factor * in_displacement).to(self.disp_emb.weight.dtype)).float()
        if self.local_corr_radius is not None:
            corr = ops.local_corr(x, y, warp_bhwc, self.local_corr_radius)
            d = torch.cat((x, x_hat, emb, corr), dim=1)
        else:
            d = torch.cat((x, x_hat, emb), dim=1)
        if self.padded_channels > d.shape[1]:
            d = F.pad(d, (0, 0, 0, 0, 0, self.padded_channels - d.shape[1]))
        d = d.to(self.compute_dtype)
        if self.channels_last:
            d = d.to(memory_format=torch.channels_last)
        d = self.block1(d)
        d = self.hidden_blocks(d)
        d = self.out_conv(d.float())
        return d[:, :-1], d[:, -1:]


class Decoder(nn.Module):
    """Coarse-to-fine decoder: scale-16 GP + TransformerDecoder, then ConvRefiner across scales 16 -> 8 -> 4 -> 2 -> 1."""

    def __init__(self) -> None:
        super().__init__()
        self.embedding_decoder = TransformerDecoder()
        self.gps = nn.ModuleDict({"16": GP()})
        self.proj = nn.ModuleDict(
            {
                scale: nn.Sequential(nn.Conv2d(spec.feat_dim, spec.proj_dim, 1, 1), nn.BatchNorm2d(spec.proj_dim))
                for scale, spec in REFINER_SPECS.items()
            }
        )
        self.conv_refiner = nn.ModuleDict({scale: ConvRefiner(spec) for scale, spec in REFINER_SPECS.items()})
        self.refine_init = 4.0
        self.fused_io = False
        self.fast_softmax = False

    def batchnorm_pairs(self, prefix: str = "") -> list[tuple[str, str]]:
        pairs = [(f"{prefix}proj.{s}.0", f"{prefix}proj.{s}.1") for s in SCALES]
        for scale in SCALES:
            base = f"{prefix}conv_refiner.{scale}."
            pairs.append((f"{base}block1.0", f"{base}block1.1"))
            for i in range(len(self.conv_refiner[scale].hidden_blocks)):
                pairs.append((f"{base}hidden_blocks.{i}.0", f"{base}hidden_blocks.{i}.1"))
        return pairs

    def _project(self, scale: str, feat: Tensor) -> Tensor:
        proj = self.proj[scale]
        first = proj[0]
        dtype = first.weight.dtype if hasattr(first, "weight") else torch.bfloat16
        if self.fused_io and isinstance(first, nn.Conv2d) and isinstance(proj[1], nn.Identity):
            # Cast + channels-last in one pass, then cuBLAS GEMM directly into channels-last FP32 [B, P, H, W].
            x_nhwc = feat.to(dtype=dtype, memory_format=torch.channels_last).permute(0, 2, 3, 1)
            return F.linear(x_nhwc, first.weight[:, :, 0, 0], first.bias).float().permute(0, 3, 1, 2)
        return proj(feat.to(dtype)).float()

    def forward(
        self,
        f1: dict[int, Tensor],
        f2: dict[int, Tensor],
        *,
        upsample: bool = False,
        flow: Tensor | None = None,
        certainty: Tensor | None = None,
        scale_factor: float = 1.0,
        symmetric: bool = False,
    ) -> dict[int, dict[str, Tensor]]:
        all_scales = UPSAMPLE_SCALES if upsample else SCALES
        sizes = {scale: feat.shape[-2:] for scale, feat in f1.items()}
        h, w = sizes[1]
        b = f1[1].shape[0]
        device = f1[1].device
        coarsest = int(all_scales[0])

        if not upsample:
            flow = normalized_grid(b, *sizes[coarsest], device).permute(0, 3, 1, 2)
            certainty_t: Tensor | float = 0.0
        else:
            assert flow is not None and certainty is not None
            flow = F.interpolate(flow, size=sizes[coarsest], align_corners=False, mode="bilinear")
            certainty_t = F.interpolate(certainty, size=sizes[coarsest], align_corners=False, mode="bilinear")

        corresps: dict[int, dict[str, Tensor]] = {}
        for new_scale in all_scales:
            ins = int(new_scale)
            refiner = self.conv_refiner[new_scale]
            use_pair = symmetric and refiner.fused_io and refiner.compute_dtype == torch.bfloat16 and b % 2 == 0
            if symmetric:
                f1_s = self._project(new_scale, f1[ins])
                if ins == 16 or not use_pair:
                    a_s, b_s = f1_s.chunk(2, dim=0)
                    f2_s = torch.cat((b_s, a_s), dim=0)
                else:
                    f2_s = f1_s
            else:
                f1_s = self._project(new_scale, f1[ins])
                f2_s = self._project(new_scale, f2[ins])

            if ins == 16:
                gp_posterior = self.gps["16"](f1_s, f2_s)
                if self.fast_softmax:
                    gm_cls, certainty_t = self.embedding_decoder(gp_posterior, f1_s, hwc_cls=True)
                    flow = cls_to_flow_refine_hwc(gm_cls)
                else:
                    gm_cls, certainty_t = self.embedding_decoder(gp_posterior, f1_s)
                    flow = cls_to_flow_refine(gm_cls)

            delta_flow, delta_certainty = refiner(
                f1_s, f2_s, flow, scale_factor=scale_factor, pair=(b // 2 if use_pair else 0)
            )
            displacement = ins * torch.stack(
                (
                    delta_flow[:, 0].float() / (self.refine_init * w),
                    delta_flow[:, 1].float() / (self.refine_init * h),
                ),
                dim=1,
            )
            flow = flow + displacement
            certainty_t = certainty_t + delta_certainty
            corresps[ins] = {"certainty": certainty_t, "flow": flow}
            if new_scale != "1":
                flow = F.interpolate(flow, size=sizes[ins // 2], mode="bilinear")
                certainty_t = F.interpolate(certainty_t, size=sizes[ins // 2], mode="bilinear")
        return corresps
