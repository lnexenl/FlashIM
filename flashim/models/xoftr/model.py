"""XoFTR (Tuzcuoğlu et al., CVPR IMW 2024): cross-modal (visible-thermal) detector-free matcher with a ResNet_8_2
backbone (1/8, 1/4, 1/2 features), an 8-layer LoFTR linear-attention coarse transformer, one-way-union coarse matching,
hierarchical medium (3x3) + fine (5x5) window attention, dual-softmax fine matching and an MLP sub-pixel refiner."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from flashim import ops
from flashim.models import sparse
from flashim.models.base import ImageMatcher
from flashim.models.loftr.model import (
    BasicBlock,
    LocalFeatureTransformer,
    _inner_cells,
    conv1x1,
    position_encoding,
)
from flashim.models.romav2.model import ImageLike, load_image
from flashim.models.xoftr import weights
from flashim.models.xoftr.config import XoFTRConfig
from flashim.platforms import default_device
from flashim.types import Matches
from flashim.utils import device_constant


class ResNet_8_2(nn.Module):
    """Returns ``(coarse 1/8 [256 ch], medium 1/4 [196 ch], fine 1/2 [128 ch])`` without FPN top-down layers."""

    def __init__(self, initial_dim: int = 128, block_dims: tuple[int, int, int] = (128, 196, 256)) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(1, initial_dim, 7, 2, 3, bias=False)
        self.bn1 = nn.BatchNorm2d(initial_dim)
        self.relu = nn.ReLU(inplace=True)
        d1, d2, d3 = block_dims
        self.layer1 = nn.Sequential(BasicBlock(initial_dim, d1, 1), BasicBlock(d1, d1))
        self.layer2 = nn.Sequential(BasicBlock(d1, d2, 2), BasicBlock(d2, d2))
        self.layer3 = nn.Sequential(BasicBlock(d2, d3, 2), BasicBlock(d3, d3))
        self.layer3_outconv = conv1x1(d3, d3)

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        x0 = self.relu(self.bn1(self.conv1(x)))
        x1 = self.layer1(x0)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        return self.layer3_outconv(x3), x2, x1


class CoarseMatching(nn.Module):
    """Projects coarse tokens and selects the union of row-wise and column-wise softmax nearest neighbours."""

    def __init__(self, d_model: int = 256) -> None:
        super().__init__()
        self.final_proj = nn.Linear(d_model, d_model, bias=True)

    def forward(
        self,
        feat_c0: Tensor,
        feat_c1: Tensor,
        hc0: int,
        wc0: int,
        hc1: int,
        wc1: int,
        *,
        threshold: float,
        border_rm: int,
        temperature: float,
        fold_temperature: bool = False,
    ) -> tuple[Tensor, Tensor]:
        a = self.final_proj(feat_c0).float() / (feat_c0.shape[-1] ** 0.5)
        b = self.final_proj(feat_c1).float() / (feat_c1.shape[-1] ** 0.5)
        if fold_temperature:
            sim = torch.einsum("nlc,nsc->nls", a / temperature, b)[0]
        else:
            sim = torch.einsum("nlc,nsc->nls", a, b)[0] / temperature

        # Union of one-way softmax thresholded nearest neighbours without materializing [L, S] softmax / bool masks:
        # (softmax(sim, dim=1) > thr & row_max) OR (softmax(sim, dim=0) > thr & col_max).
        row_max, row_j = sim.max(dim=1)
        row_prob = (row_max - torch.logsumexp(sim, dim=1)).exp()
        col_max, col_i = sim.max(dim=0)
        col_prob = (col_max - torch.logsumexp(sim, dim=0)).exp()
        del sim

        keep_row = row_prob > threshold
        keep_col = col_prob > threshold
        if border_rm > 0:
            inner0 = _inner_cells(hc0, wc0, border_rm, row_prob.device)
            inner1 = _inner_cells(hc1, wc1, border_rm, row_prob.device)
            keep_row = keep_row & inner0 & inner1[row_j]
            keep_col = keep_col & inner1 & inner0[col_i]

        col_j_all = torch.arange(col_prob.shape[0], device=col_prob.device)
        already_in_row = keep_row[col_i] & (row_j[col_i] == col_j_all)
        keep_col_only = keep_col & ~already_in_row

        i_row = torch.nonzero(keep_row)[:, 0]
        j_row = row_j[i_row]
        j_col = torch.nonzero(keep_col_only)[:, 0]
        i_col = col_i[j_col]

        i_ids = torch.cat((i_row, i_col), dim=0)
        j_ids = torch.cat((j_row, j_col), dim=0)
        if i_ids.numel() > 1:
            order = torch.argsort(i_ids * (hc1 * wc1) + j_ids)
            i_ids, j_ids = i_ids[order], j_ids[order]
        return i_ids, j_ids


class WindowMlp(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int) -> None:
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, out_dim)

    def forward(self, x: Tensor) -> Tensor:
        return self.fc2(self.act(self.fc1(x)))


class WindowAttention(nn.Module):
    """Multi-head scaled dot-product attention over local feature windows (dispatches through ``ops.sdpa``)."""

    def __init__(self, dim: int, num_heads: int) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.kv_proj = nn.Linear(dim, dim * 2, bias=False)
        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.merge = nn.Linear(dim, dim, bias=True)

    def forward(self, x_q: Tensor, x_kv: Tensor | None = None) -> Tensor:
        if x_kv is None:
            x_kv = x_q
        bs, _, dim = x_q.shape
        kv = self.kv_proj(x_kv).reshape(bs, -1, 2, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q = self.q_proj(x_q).reshape(bs, -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        out = ops.sdpa(q, kv[0], kv[1]).transpose(1, 2).reshape(bs, -1, dim)
        return self.merge(out)


class CrossBidirectionalAttention(nn.Module):
    """Shared-QK bidirectional cross-attention between matched local windows (FP32 softmax)."""

    def __init__(self, dim: int, num_heads: int) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.25
        self.qk_proj = nn.Linear(dim, dim, bias=False)
        self.v_proj = nn.Linear(dim, dim, bias=False)
        self.merge = nn.Linear(dim, dim, bias=False)
        # Present in upstream checkpoints (unused in forward); kept for strict=True loading.
        self.temperature = nn.Parameter(torch.zeros(1))

    def forward(self, x0: Tensor, x1: Tensor) -> tuple[Tensor, Tensor]:
        bs = x0.shape[0]
        qk0, qk1 = self.qk_proj(x0), self.qk_proj(x1)
        v0, v1 = self.v_proj(x0), self.v_proj(x1)
        qk0, qk1, v0, v1 = (
            t.reshape(bs, -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3) for t in (qk0, qk1, v0, v1)
        )
        sim = ((qk0 * self.scale) @ (qk1 * self.scale).transpose(-2, -1)).float()
        attn01 = F.softmax(sim, dim=-1).to(v1.dtype)
        attn10 = F.softmax(sim.transpose(-2, -1), dim=-1).to(v0.dtype)
        out0 = (attn01 @ v1).transpose(1, 2).flatten(-2)
        out1 = (attn10 @ v0).transpose(1, 2).flatten(-2)
        return self.merge(out0), self.merge(out1)


class WindowPosEmbMLP(nn.Module):
    """Relative 2-D grid positional embedding MLP for a square window, cached per ``(seq_length, device, dtype)``."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.pos_mlp = nn.Sequential(nn.Linear(2, 512, bias=True), nn.ReLU(), nn.Linear(512, dim, bias=False))
        self._cache: dict[tuple[int, torch.device, torch.dtype], Tensor] = {}

    def forward(self, x: Tensor) -> Tensor:
        seq_length = x.shape[1]
        dtype = self.pos_mlp[0].weight.dtype
        key = (seq_length, x.device, dtype)
        pos = self._cache.get(key)
        if pos is None:
            side = int(seq_length**0.5)
            coords = torch.arange(0, side, device=x.device, dtype=dtype)
            grid = torch.stack(torch.meshgrid(coords, coords, indexing="ij")).unsqueeze(0)
            half = side // 2
            grid = (grid - half) / half
            pos = self.pos_mlp(grid.flatten(2).transpose(1, 2))
            self._cache[key] = pos
        return x + pos.to(x.dtype)


class WindowSelfAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int, mlp_hidden_coef: int, use_pre_pos_embed: bool = False) -> None:
        super().__init__()
        self.mlp = WindowMlp(dim * 2, dim * mlp_hidden_coef, dim)
        self.gamma = nn.Parameter(torch.ones(dim))
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.attn = WindowAttention(dim, num_heads)
        self.pos_embed = WindowPosEmbMLP(dim)
        self.pos_embed_pre: nn.Module = WindowPosEmbMLP(dim) if use_pre_pos_embed else nn.Identity()

    def forward(self, x: Tensor, x_pre: Tensor) -> tuple[Tensor, Tensor]:
        ww, ww_pre = x.shape[1], x_pre.shape[1]
        x = torch.cat((self.pos_embed(x), self.pos_embed_pre(x_pre)), dim=1)
        x = x + self.gamma * self.norm1(self.mlp(torch.cat((x, self.attn(self.norm2(x))), dim=-1)))
        return x.split((ww, ww_pre), dim=1)


class WindowCrossAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int, mlp_hidden_coef: int) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = WindowMlp(dim * 2, dim * mlp_hidden_coef, dim)
        self.cross_attn = CrossBidirectionalAttention(dim, num_heads)
        self.gamma = nn.Parameter(torch.ones(dim))

    def forward(self, x0: Tensor, x1: Tensor) -> tuple[Tensor, Tensor]:
        m0, m1 = self.cross_attn(self.norm1(x0), self.norm1(x1))
        x0 = x0 + self.gamma * self.norm2(self.mlp(torch.cat((x0, m0), dim=-1)))
        x1 = x1 + self.gamma * self.norm2(self.mlp(torch.cat((x1, m1), dim=-1)))
        return x0, x1


class FineProcess(nn.Module):
    """Coarse-to-medium (1/4, 3x3) and medium-to-fine (1/2, 5x5) window attention around each coarse match."""

    def __init__(self, cfg: XoFTRConfig) -> None:
        super().__init__()
        d1, d2, d3 = cfg.block_dims
        self.block_dims = cfg.block_dims
        self.W_m = cfg.medium_window_size
        self.W_f = cfg.fine_window_size
        self.conv_merge = nn.Sequential(
            conv1x1(d3 * 2, d2),
            nn.Conv2d(d2, d2, 3, 1, 1, groups=d2, bias=False),
            nn.BatchNorm2d(d2),
        )
        self.out_conv_m = conv1x1(d2, d2)
        self.out_conv_f = conv1x1(d1, d1)
        self.self_attn_m = WindowSelfAttention(d2, cfg.nhead_medium_level, cfg.mlp_hidden_dim_coef, False)
        self.cross_attn_m = WindowCrossAttention(d2, cfg.nhead_medium_level, cfg.mlp_hidden_dim_coef)
        self.self_attn_f = WindowSelfAttention(d1, cfg.nhead_fine_level, cfg.mlp_hidden_dim_coef, True)
        self.cross_attn_f = WindowCrossAttention(d1, cfg.nhead_fine_level, cfg.mlp_hidden_dim_coef)
        self.down_proj_m_f = nn.Linear(d2, d1, bias=False)

    def forward(
        self,
        feat_f0: Tensor,
        feat_f1: Tensor,
        feat_m0: Tensor,
        feat_m1: Tensor,
        feat_c0: Tensor,
        feat_c1: Tensor,
        feat_c0_pre: Tensor,
        feat_c1_pre: Tensor,
        i_ids: Tensor,
        j_ids: Tensor,
    ) -> tuple[Tensor, Tensor]:
        if i_ids.numel() == 0:
            empty = feat_f0.new_empty((0, self.W_f * self.W_f, self.block_dims[0]))
            return empty, empty

        hc0, wc0 = feat_c0_pre.shape[-2:]
        hc1, wc1 = feat_c1_pre.shape[-2:]
        d2 = self.block_dims[1]

        c0_2d = feat_c0.permute(0, 2, 1).reshape(1, -1, hc0, wc0)
        c1_2d = feat_c1.permute(0, 2, 1).reshape(1, -1, hc1, wc1)
        c0_m = self.conv_merge(torch.cat((c0_2d, feat_c0_pre), dim=1)).permute(0, 2, 3, 1).reshape(1, -1, 1, d2)
        c1_m = self.conv_merge(torch.cat((c1_2d, feat_c1_pre), dim=1)).permute(0, 2, 3, 1).reshape(1, -1, 1, d2)

        if feat_m0.shape[2:] == feat_m1.shape[2:]:
            feat_m0, feat_m1 = self.out_conv_m(torch.cat((feat_m0, feat_m1), dim=0)).chunk(2, dim=0)
            feat_f0, feat_f1 = self.out_conv_f(torch.cat((feat_f0, feat_f1), dim=0)).chunk(2, dim=0)
        else:
            feat_m0, feat_m1 = self.out_conv_m(feat_m0), self.out_conv_m(feat_m1)
            feat_f0, feat_f1 = self.out_conv_f(feat_f0), self.out_conv_f(feat_f1)

        stride_m = feat_m0.shape[2] // hc0
        stride_f = feat_f0.shape[2] // hc0

        def unfold_select(feat: Tensor, win: int, stride: int, ids: Tensor) -> Tensor:
            u = F.unfold(feat, kernel_size=(win, win), stride=stride, padding=win // 2)
            return u.reshape(1, feat.shape[1], win * win, -1).permute(0, 3, 2, 1)[0, ids]

        m0 = unfold_select(feat_m0, self.W_m, stride_m, i_ids)
        m1 = unfold_select(feat_m1, self.W_m, stride_m, j_ids)
        f0 = unfold_select(feat_f0, self.W_f, stride_f, i_ids)
        f1 = unfold_select(feat_f1, self.W_f, stride_f, j_ids)
        c0_sel, c1_sel = c0_m[0, i_ids], c1_m[0, j_ids]

        # Self + cross attention at medium level (1/4, 3x3 + coarse token)
        m_both, _ = self.self_attn_m(torch.cat((m0, m1), dim=0), torch.cat((c0_sel, c1_sel), dim=0))
        m0, m1 = m_both.chunk(2, dim=0)
        m0, m1 = self.cross_attn_m(m0, m1)

        # Down-project medium window to fine dim, then self + cross attention at fine level (1/2, 5x5 + 3x3)
        m0, m1 = self.down_proj_m_f(torch.cat((m0, m1), dim=0)).chunk(2, dim=0)
        f_both, _ = self.self_attn_f(torch.cat((f0, f1), dim=0), torch.cat((m0, m1), dim=0))
        f0, f1 = f_both.chunk(2, dim=0)
        return self.cross_attn_f(f0, f1)


class FineSubMatching(nn.Module):
    """Fine 5x5 dual-softmax matching (FP32) + MLP sub-pixel coordinate refinement."""

    def __init__(self, cfg: XoFTRConfig) -> None:
        super().__init__()
        d1 = cfg.block_dims[0]
        self.W_f = cfg.fine_window_size
        self.fine_proj = nn.Linear(d1, d1, bias=False)
        self.subpixel_mlp = nn.Sequential(
            nn.Linear(2 * d1, 2 * d1, bias=False),
            nn.ReLU(),
            nn.Linear(2 * d1, 4, bias=False),
        )

    def forward(
        self,
        feat_f0_unfold: Tensor,
        feat_f1_unfold: Tensor,
        i_ids_c: Tensor,
        j_ids_c: Tensor,
        hc0: int,
        wc0: int,
        wc1: int,
        hf0: int,
        h0: int,
        *,
        fine_threshold: float,
        temperature: float,
        denser: bool,
    ) -> tuple[Tensor, Tensor, Tensor]:
        m_count = feat_f0_unfold.shape[0]
        if m_count == 0:
            empty_pts = feat_f0_unfold.new_empty((0, 2), dtype=torch.float32)
            return empty_pts, empty_pts, feat_f0_unfold.new_empty((0,), dtype=torch.float32)

        f0 = self.fine_proj(feat_f0_unfold).float() / (feat_f0_unfold.shape[-1] ** 0.5)
        f1 = self.fine_proj(feat_f1_unfold).float() / (feat_f1_unfold.shape[-1] ** 0.5)
        sim = torch.einsum("nlc,nsc->nls", f0, f1) / temperature
        conf = F.softmax(sim, dim=1) * F.softmax(sim, dim=2)

        mask = conf > fine_threshold
        if not mask.any():
            mask[0, 0, 0] = True
            conf[0, 0, 0] = 1.0

        if not denser:
            mask = mask & (conf == conf.amax(dim=(1, 2), keepdim=True))
        else:
            mask = mask & (conf == conf.max(dim=2, keepdim=True).values) & (conf == conf.max(dim=1, keepdim=True).values)

        mask_v, all_j_ids = mask.max(dim=2)
        b_ids, i_ids = torch.where(mask_v)
        j_ids = all_j_ids[b_ids, i_ids]
        mconf = conf[b_ids, i_ids, j_ids]

        W_f = self.W_f
        scale_f_c = hf0 // hc0
        c0_f = torch.stack((i_ids_c % wc0, torch.div(i_ids_c, wc0, rounding_mode="trunc")), dim=1) * scale_f_c
        c1_f = torch.stack((j_ids_c % wc1, torch.div(j_ids_c, wc1, rounding_mode="trunc")), dim=1) * scale_f_c
        w0 = torch.stack((i_ids % W_f, torch.div(i_ids, W_f, rounding_mode="trunc")), dim=1)
        w1 = torch.stack((j_ids % W_f, torch.div(j_ids, W_f, rounding_mode="trunc")), dim=1)

        sub_in = torch.cat((feat_f0_unfold[b_ids, i_ids], feat_f1_unfold[b_ids, j_ids]), dim=-1)
        sub_dtype = self.subpixel_mlp[0].weight.dtype
        sub_ref = self.subpixel_mlp(sub_in.to(sub_dtype)).float()
        sub_ref0, sub_ref1 = sub_ref.chunk(2, dim=-1)
        sub_ref0 = torch.tanh(sub_ref0) * 0.5
        sub_ref1 = torch.tanh(sub_ref1) * 0.5

        scale = h0 / hf0
        half_w = W_f // 2
        mkpts0 = (w0 + c0_f[b_ids] - half_w + sub_ref0) * scale
        mkpts1 = (w1 + c1_f[b_ids] - half_w + sub_ref1) * scale

        keep = mconf != 0
        return mkpts0[keep], mkpts1[keep], mconf[keep]


class XoFTR(ImageMatcher, nn.Module):
    """Module paths follow the official ``weights_xoftr_640.ckpt`` checkpoint (``backbone``, ``loftr_coarse``,
    ``coarse_matching``, ``fine_process``, ``fine_matching``); loads strictly. ``match(img_a, img_b)`` ->
    :class:`Matches` in each original image's pixel coordinates (default working long side: 640 px)."""

    div_by = 8
    match_defaults: Mapping[str, Any] = {"resize": 640}
    fold_temperature = False  # see spec.py's fold_similarity_temperature pass
    preset: str | None = None
    engine: str | None = None
    quant_report = None
    calibration_metadata = None
    fp32_modules: tuple[str, ...] = ()
    compute_dtype = torch.float32

    def __init__(self, config: XoFTRConfig | None = None, **overrides) -> None:
        super().__init__()
        cfg = config or XoFTRConfig()
        if overrides:
            cfg = XoFTRConfig(**{**cfg.__dict__, **overrides})
        self.config = cfg
        self.threshold = cfg.coarse_threshold
        self.fine_threshold = cfg.fine_threshold
        self.denser = cfg.denser
        self.border_rm = cfg.border_rm
        self.coarse_temperature = cfg.coarse_temperature
        self.fine_temperature = cfg.fine_temperature
        self.match_defaults = {"resize": cfg.default_resize}

        self.backbone = ResNet_8_2(cfg.initial_dim, cfg.block_dims)
        self.loftr_coarse = LocalFeatureTransformer(cfg.d_model, cfg.coarse_nhead, list(cfg.coarse_layer_names))
        self.coarse_matching = CoarseMatching(cfg.d_model)
        self.fine_process = FineProcess(cfg)
        self.fine_matching = FineSubMatching(cfg)
        self._pe: dict[tuple[int, int, torch.device], Tensor] = {}

    @classmethod
    def from_pretrained(
        cls,
        *,
        checkpoint: str | Path | None = None,
        device: torch.device | str | None = None,
        **kwargs,
    ) -> XoFTR:
        with torch.device("meta"):
            model = cls(**kwargs)
        model.load_state_dict(weights.load_state_dict(checkpoint), strict=True, assign=True)
        return model.to(torch.device(device or default_device())).eval()

    @property
    def device(self) -> torch.device:
        return self.backbone.conv1.weight.device

    def set_compute_dtype(self, dtype: torch.dtype) -> None:
        self.to(dtype)
        for path in self.fp32_modules:
            self.get_submodule(path).float()
        self.compute_dtype = dtype

    def batchnorm_pairs(self) -> list[tuple[str, str]]:
        pairs = [("backbone.conv1", "backbone.bn1")]
        for layer in ("layer1", "layer2", "layer3"):
            for block in (0, 1):
                base = f"backbone.{layer}.{block}"
                pairs += [(f"{base}.conv1", f"{base}.bn1"), (f"{base}.conv2", f"{base}.bn2")]
                if self.get_submodule(base).downsample is not None:
                    pairs.append((f"{base}.downsample.0", f"{base}.downsample.1"))
        pairs.append(("fine_process.conv_merge.1", "fine_process.conv_merge.2"))
        return pairs

    fold_batchnorm = sparse.SparseExtractor.fold_batchnorm

    def positions(self, h: int, w: int) -> Tensor:
        key = (h, w, self.device)
        if key not in self._pe:
            self._pe[key] = position_encoding(self.config.d_model, h, w, True, self.device)
        return self._pe[key]

    @staticmethod
    def _normalize_gray(gray: Tensor, eps: float = 1e-6) -> Tensor:
        g = gray.float()
        mean = g.mean(dim=(2, 3), keepdim=True)
        std = g.std(dim=(2, 3), keepdim=True)
        return (g - mean) / (std + eps)

    def extract_features(
        self, gray0: Tensor, gray1: Tensor
    ) -> tuple[tuple[Tensor, Tensor, Tensor], tuple[Tensor, Tensor, Tensor]]:
        n0 = self._normalize_gray(gray0).to(self.compute_dtype)
        n1 = self._normalize_gray(gray1).to(self.compute_dtype)
        if n0.shape[2:] == n1.shape[2:]:
            fc, fm, ff = self.backbone(torch.cat((n0, n1), dim=0))
            (c0, c1), (m0, m1), (f0, f1) = fc.chunk(2, dim=0), fm.chunk(2, dim=0), ff.chunk(2, dim=0)
            return (c0, m0, f0), (c1, m1, f1)
        return self.backbone(n0), self.backbone(n1)

    @torch.inference_mode()
    def match_prepared(self, gray0: Tensor, gray1: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """``[1, 1, H, W]`` grayscale pair (sides multiples of 8) -> ``(kpts0 [M, 2], kpts1 [M, 2], conf [M])`` in the
        inputs' pixel coordinates."""
        h0 = gray0.shape[-2]
        (c0, m0, f0), (c1, m1, f1) = self.extract_features(gray0, gray1)
        (hc0, wc0), (hc1, wc1) = c0.shape[-2:], c1.shape[-2:]

        tokens0 = (c0.float() + self.positions(hc0, wc0)).to(c0.dtype).permute(0, 2, 3, 1).reshape(1, -1, c0.shape[1])
        tokens1 = (c1.float() + self.positions(hc1, wc1)).to(c1.dtype).permute(0, 2, 3, 1).reshape(1, -1, c1.shape[1])
        tokens0, tokens1 = self.loftr_coarse(tokens0, tokens1)

        i_ids, j_ids = self.coarse_matching(
            tokens0,
            tokens1,
            hc0,
            wc0,
            hc1,
            wc1,
            threshold=self.threshold,
            border_rm=self.border_rm,
            temperature=self.coarse_temperature,
            fold_temperature=self.fold_temperature,
        )
        if i_ids.numel() == 0:
            empty_pts = tokens0.new_empty((0, 2), dtype=torch.float32)
            return empty_pts, empty_pts, tokens0.new_empty((0,), dtype=torch.float32)

        f0_win, f1_win = self.fine_process(f0, f1, m0, m1, tokens0, tokens1, c0, c1, i_ids, j_ids)
        return self.fine_matching(
            f0_win,
            f1_win,
            i_ids,
            j_ids,
            hc0,
            wc0,
            wc1,
            f0.shape[-2],
            h0,
            fine_threshold=self.fine_threshold,
            temperature=self.fine_temperature,
            denser=self.denser,
        )

    def _match(
        self,
        img_a: ImageLike,
        img_b: ImageLike,
        *,
        resize: int | None,
        max_matches: int | None,
        threshold: float | None,
    ) -> Matches:
        """``resize``: long side of the working resolution (default 640, sides floored to multiples of 8);
        ``max_matches``: keep the most confident matches (``None``: all); ``threshold``: coarse confidence threshold
        (default 0.3)."""
        images = [load_image(im, self.device) for im in (img_a, img_b)]
        prepared, sizes = [], []
        for image in images:
            height, width = image.shape[-2:]
            new_h, new_w = sparse.target_size(height, width, resize, self.div_by)
            if (new_h, new_w) != (height, width):
                image = F.interpolate(image, size=(new_h, new_w), mode="bilinear", align_corners=False, antialias=True)
            prepared.append(sparse.rgb_to_gray(image))
            sizes.append((height / new_h, width / new_w))
        previous = self.threshold
        self.threshold = previous if threshold is None else threshold
        try:
            k0, k1, conf = self.match_prepared(*prepared)
        finally:
            self.threshold = previous
        if max_matches is not None and max_matches < len(conf):
            conf, index = conf.topk(max_matches)
            k0, k1 = k0[index], k1[index]
        k0 = k0 * device_constant([sizes[0][1], sizes[0][0]], k0)
        k1 = k1 * device_constant([sizes[1][1], sizes[1][0]], k1)
        return Matches(k0, k1, conf)
