"""RoMaV2 convolutional warp refiners (strides 4, 2, 1)."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from flashim import ops
from flashim.models.romav2.geometry import bhwc_grid_sample, normalized_grid


class RefinerBlock(nn.Module):
    """Depthwise 5x5 -> BatchNorm -> ReLU -> pointwise 1x1, under BF16 autocast. No residual."""

    def __init__(self, channels: int, kernel_size: int = 5) -> None:
        super().__init__()
        self.conv_depthwise = nn.Conv2d(channels, channels, kernel_size, padding=kernel_size // 2, groups=channels)
        self.norm: nn.Module = nn.BatchNorm2d(channels, momentum=0.01)
        self.relu = nn.ReLU(inplace=True)
        self.conv_pointwise = nn.Conv2d(channels, channels, 1)

    def forward(self, x: Tensor) -> Tensor:
        with torch.autocast(x.device.type, torch.bfloat16):
            return self.conv_pointwise(self.relu(self.norm(self.conv_depthwise(x))))


@dataclass(frozen=True)
class RefinerConfig:
    feat_dim: int
    proj_dim: int
    disp_emb_dim: int
    local_corr_radius: int | None
    hidden_blocks: int = 8
    confidence_dim: int = 4
    refine_init: float = 4.0

    @property
    def channels(self) -> int:
        corr = 0 if self.local_corr_radius is None else (2 * self.local_corr_radius + 1) ** 2
        return 2 * self.proj_dim + self.disp_emb_dim + corr


REFINER_CONFIGS = {
    4: RefinerConfig(feat_dim=256, proj_dim=192, disp_emb_dim=79, local_corr_radius=3),
    2: RefinerConfig(feat_dim=128, proj_dim=48, disp_emb_dim=23, local_corr_radius=1),
    1: RefinerConfig(feat_dim=64, proj_dim=12, disp_emb_dim=8, local_corr_radius=None),
}


class ConvRefiner(nn.Module):
    def __init__(self, cfg: RefinerConfig) -> None:
        super().__init__()
        self.cfg = cfg
        channels = cfg.channels
        self.proj = nn.Linear(cfg.feat_dim, cfg.proj_dim)
        self.block1 = RefinerBlock(channels)
        self.hidden_blocks = nn.Sequential(*(RefinerBlock(channels) for _ in range(cfg.hidden_blocks)))
        self.disp_emb = nn.Conv2d(2, cfg.disp_emb_dim, 1)
        self.warp_head = nn.Conv2d(channels, 2, 1)
        self.confidence_head = nn.Conv2d(channels, cfg.confidence_dim, 1)
        # Set by the refiner optimization pass (BN fold + channels-last weights).
        self.channels_last = False

    def forward(
        self, f_a: Tensor, f_b: Tensor, prev_warp: Tensor, prev_confidence: Tensor, scale_factor: Tensor
    ) -> tuple[Tensor, Tensor]:
        cfg = self.cfg
        batch, h, w, _ = f_a.shape
        f_a = self.proj(f_a.reshape(batch, h * w, cfg.feat_dim).float()).reshape(batch, h, w, cfg.proj_dim)
        f_b = self.proj(f_b.reshape(batch, h * w, cfg.feat_dim).float()).reshape(batch, h, w, cfg.proj_dim)
        f_ba = bhwc_grid_sample(f_b, prev_warp, mode="bilinear", align_corners=False)
        displacement = (prev_warp - normalized_grid(batch, h, w, f_a.device)).permute(0, 3, 1, 2)
        emb = self.disp_emb(scale_factor[None, :, None, None] * displacement)
        parts = [f_a.permute(0, 3, 1, 2), f_ba.permute(0, 3, 1, 2), emb]
        if cfg.local_corr_radius is not None:
            parts.append(ops.local_corr(parts[0], f_b.permute(0, 3, 1, 2), prev_warp, cfg.local_corr_radius))
        d = torch.cat(parts, dim=1)
        if self.channels_last:
            d = d.to(memory_format=torch.channels_last)
        z = self.hidden_blocks(self.block1(d)).float()

        displacement_out = self.warp_head(z).permute(0, 2, 3, 1).reshape(batch, h, w, 2)
        delta = self.confidence_head(z).permute(0, 2, 3, 1).reshape(batch, h, w, cfg.confidence_dim)
        normalization = torch.empty(2, device=prev_warp.device, dtype=prev_warp.dtype)
        normalization[0].fill_(w)
        normalization[1].fill_(h)
        warp = prev_warp + displacement_out / (cfg.refine_init * normalization[None, None, None])

        l00 = F.softplus(delta[..., 1]) + 1e-6
        l10 = delta[..., 2]
        l11 = F.softplus(delta[..., 3]) + 1e-6
        delta = torch.stack([delta[..., 0], l00 * l00, l00 * l10, l10 * l10 + l11 * l11], dim=-1)
        if prev_confidence.shape[-1] != delta.shape[-1]:
            padded = torch.zeros_like(delta)
            padded[..., :1] = prev_confidence
            prev_confidence = padded
        return warp, prev_confidence + delta


class Refiners(nn.ModuleDict):
    def __init__(self) -> None:
        super().__init__({str(stride): ConvRefiner(REFINER_CONFIGS[stride]) for stride in (4, 2, 1)})
