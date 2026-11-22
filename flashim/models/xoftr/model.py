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

    low_precision_similarity = False

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
        c = feat_c0.shape[-1]
        if self.low_precision_similarity and feat_c0.dtype != torch.float32:
            a = self.final_proj(feat_c0)[0]
            b = self.final_proj(feat_c1)[0]
            sim = ops.matmul_nt_fp32(a, b, 1.0 / (c * temperature))
        else:
            a = self.final_proj(feat_c0).float() / (c**0.5)
            b = self.final_proj(feat_c1).float() / (c**0.5)
            if fold_temperature:
                sim = torch.einsum("nlc,nsc->nls", a / temperature, b)[0]
            else:
                sim = torch.einsum("nlc,nsc->nls", a, b)[0] / temperature

        # Union of one-way softmax thresholded nearest neighbours (single-pass Triton kernel on CUDA).
        row_prob, row_j, col_prob, col_i = ops.union_softmax_match(sim)
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
        self._qkv_weight: Tensor | None = None

    def pack_qkv(self) -> None:
        """Pack ``q_proj`` + ``kv_proj`` weights into one ``[3*C, C]`` GEMM (exact)."""
        if isinstance(self.q_proj, nn.Linear) and isinstance(self.kv_proj, nn.Linear):
            self._qkv_weight = torch.cat((self.q_proj.weight.detach(), self.kv_proj.weight.detach()), dim=0).contiguous()

    def forward(self, x_q: Tensor, x_kv: Tensor | None = None, *, query_len: int | None = None) -> Tensor:
        bs, seq_len, dim = x_q.shape
        out_len = seq_len if query_len is None else query_len
        if x_kv is None and self._qkv_weight is not None:
            qkv = F.linear(x_q, self._qkv_weight).reshape(bs, seq_len, 3, self.num_heads, self.head_dim)
            q = (qkv[:, :out_len, 0] if out_len < seq_len else qkv[:, :, 0]).permute(0, 2, 1, 3)
            k = qkv[:, :, 1].permute(0, 2, 1, 3)
            v = qkv[:, :, 2].permute(0, 2, 1, 3)
        else:
            if x_kv is None:
                x_kv = x_q
            kv = self.kv_proj(x_kv).reshape(bs, -1, 2, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
            q_in = x_q[:, :out_len] if out_len < seq_len else x_q
            q = self.q_proj(q_in).reshape(bs, out_len, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
            k, v = kv[0], kv[1]
        out = ops.sdpa(q, k, v).transpose(1, 2).reshape(bs, out_len, dim)
        return self.merge(out)


class CrossBidirectionalAttention(nn.Module):
    """Shared-QK bidirectional cross-attention between matched local windows (``ops.bidirectional_cross_attention``)."""

    def __init__(self, dim: int, num_heads: int) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.qk_proj = nn.Linear(dim, dim, bias=False)
        self.v_proj = nn.Linear(dim, dim, bias=False)
        self.merge = nn.Linear(dim, dim, bias=False)
        # Present in upstream checkpoints (unused in forward); kept for strict=True loading.
        self.temperature = nn.Parameter(torch.zeros(1))
        self._qkv_weight: Tensor | None = None

    def pack_qkv(self) -> None:
        """Pack ``qk_proj`` + ``v_proj`` weights into one ``[2*C, C]`` GEMM (exact)."""
        if isinstance(self.qk_proj, nn.Linear) and isinstance(self.v_proj, nn.Linear):
            self._qkv_weight = torch.cat((self.qk_proj.weight.detach(), self.v_proj.weight.detach()), dim=0).contiguous()

    def forward(self, x0: Tensor, x1: Tensor) -> tuple[Tensor, Tensor]:
        qk0, qk1 = self.qk_proj(x0), self.qk_proj(x1)
        v0, v1 = self.v_proj(x0), self.v_proj(x1)
        out0, out1 = ops.bidirectional_cross_attention(qk0, qk1, v0, v1, self.num_heads)
        return self.merge(out0), self.merge(out1)


class WindowPosEmbMLP(nn.Module):
    """Relative 2-D grid positional embedding MLP for a square window, cached per ``(seq_length, device, dtype)``."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.pos_mlp = nn.Sequential(nn.Linear(2, 512, bias=True), nn.ReLU(), nn.Linear(512, dim, bias=False))
        self._cache: dict[tuple[int, torch.device, torch.dtype], Tensor] = {}

    def get_pos(self, seq_length: int, device: torch.device) -> Tensor:
        dtype = self.pos_mlp[0].weight.dtype
        key = (seq_length, device, dtype)
        pos = self._cache.get(key)
        if pos is None:
            side = int(seq_length**0.5)
            coords = torch.arange(0, side, device=device, dtype=dtype)
            grid = torch.stack(torch.meshgrid(coords, coords, indexing="ij")).unsqueeze(0)
            half = side // 2
            grid = (grid - half) / half
            pos = self.pos_mlp(grid.flatten(2).transpose(1, 2))
            self._cache[key] = pos
        return pos

    def forward(self, x: Tensor) -> Tensor:
        return x + self.get_pos(x.shape[1], x.device).to(x.dtype)


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
        self._gamma_folded = False

    def fold_gamma(self) -> None:
        if not self._gamma_folded:
            with torch.no_grad():
                g = self.gamma.to(self.norm1.weight.dtype)
                self.norm1.weight.mul_(g)
                self.norm1.bias.mul_(g)
                self.gamma.fill_(1.0)
            self._gamma_folded = True

    def forward(self, x: Tensor, x_pre: Tensor, *, drop_pre: bool = False) -> tuple[Tensor, Tensor | None]:
        ww, ww_pre = x.shape[1], x_pre.shape[1]
        pos = self.pos_embed.get_pos(ww, x.device)
        pos_pre = (
            self.pos_embed_pre.get_pos(ww_pre, x_pre.device)
            if isinstance(self.pos_embed_pre, WindowPosEmbMLP)
            else None
        )
        x_base, x_norm = ops.pos_cat_layernorm(
            x, x_pre, pos, pos_pre, self.norm2.weight, self.norm2.bias, self.norm2.eps, return_all_pos=not drop_pre
        )
        attn_out = self.attn(x_norm, query_len=ww if drop_pre else None)
        if self._gamma_folded and isinstance(self.mlp.fc1, nn.Linear) and isinstance(self.mlp.fc2, nn.Linear):
            x_out = ops.cat_gelu_mlp(
                x_base,
                attn_out,
                self.mlp.fc1.weight,
                self.mlp.fc1.bias,
                self.mlp.fc2.weight,
                self.mlp.fc2.bias,
                self.norm1.weight,
                self.norm1.bias,
                self.norm1.eps,
            )
        else:
            mlp_out = self.mlp(torch.cat((x_base, attn_out), dim=-1))
            if self._gamma_folded:
                x_out = ops.residual_layernorm(mlp_out, x_base, self.norm1.weight, self.norm1.bias, self.norm1.eps)
            else:
                x_out = x_base + self.gamma * self.norm1(mlp_out)
        if drop_pre:
            return x_out, None
        return x_out.split((ww, ww_pre), dim=1)


class WindowCrossAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int, mlp_hidden_coef: int) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = WindowMlp(dim * 2, dim * mlp_hidden_coef, dim)
        self.cross_attn = CrossBidirectionalAttention(dim, num_heads)
        self.gamma = nn.Parameter(torch.ones(dim))
        self._gamma_folded = False

    def fold_gamma(self) -> None:
        if not self._gamma_folded:
            with torch.no_grad():
                g = self.gamma.to(self.norm2.weight.dtype)
                self.norm2.weight.mul_(g)
                self.norm2.bias.mul_(g)
                self.gamma.fill_(1.0)
            self._gamma_folded = True

    def forward_stacked(self, x: Tensor) -> Tensor:
        n = ops.residual_layernorm(x, None, self.norm1.weight, self.norm1.bias, self.norm1.eps)
        ca = self.cross_attn
        if ca._qkv_weight is not None:
            qkv = F.linear(n, ca._qkv_weight)
            qk, v = qkv.chunk(2, dim=-1)
            qk0, qk1 = qk.chunk(2, dim=0)
            v0, v1 = v.chunk(2, dim=0)
        else:
            qk0, qk1 = ca.qk_proj(n).chunk(2, dim=0)
            v0, v1 = ca.v_proj(n).chunk(2, dim=0)
        a0, a1 = ops.bidirectional_cross_attention(qk0, qk1, v0, v1, ca.num_heads)
        m = ca.merge(torch.cat((a0, a1), dim=0))
        if self._gamma_folded and isinstance(self.mlp.fc1, nn.Linear) and isinstance(self.mlp.fc2, nn.Linear):
            return ops.cat_gelu_mlp(
                x,
                m,
                self.mlp.fc1.weight,
                self.mlp.fc1.bias,
                self.mlp.fc2.weight,
                self.mlp.fc2.bias,
                self.norm2.weight,
                self.norm2.bias,
                self.norm2.eps,
            )
        mlp_out = self.mlp(torch.cat((x, m), dim=-1))
        if self._gamma_folded:
            return ops.residual_layernorm(mlp_out, x, self.norm2.weight, self.norm2.bias, self.norm2.eps)
        return x + self.gamma * self.norm2(mlp_out)

    def forward(self, x0: Tensor, x1: Tensor) -> tuple[Tensor, Tensor]:
        return self.forward_stacked(torch.cat((x0, x1), dim=0)).chunk(2, dim=0)


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
        if (hc0, wc0) == (hc1, wc1):
            c_both = self.conv_merge(
                torch.cat((torch.cat((c0_2d, feat_c0_pre), dim=1), torch.cat((c1_2d, feat_c1_pre), dim=1)), dim=0)
            )
            c0_m, c1_m = c_both.permute(0, 2, 3, 1).reshape(2, -1, 1, d2).unbind(0)
            c0_sel, c1_sel = c0_m[i_ids], c1_m[j_ids]
        else:
            c0_m = self.conv_merge(torch.cat((c0_2d, feat_c0_pre), dim=1)).permute(0, 2, 3, 1).reshape(-1, 1, d2)
            c1_m = self.conv_merge(torch.cat((c1_2d, feat_c1_pre), dim=1)).permute(0, 2, 3, 1).reshape(-1, 1, d2)
            c0_sel, c1_sel = c0_m[i_ids], c1_m[j_ids]

        if feat_m0.shape[2:] == feat_m1.shape[2:]:
            feat_m0, feat_m1 = self.out_conv_m(torch.cat((feat_m0, feat_m1), dim=0)).chunk(2, dim=0)
            feat_f0, feat_f1 = self.out_conv_f(torch.cat((feat_f0, feat_f1), dim=0)).chunk(2, dim=0)
        else:
            feat_m0, feat_m1 = self.out_conv_m(feat_m0), self.out_conv_m(feat_m1)
            feat_f0, feat_f1 = self.out_conv_f(feat_f0), self.out_conv_f(feat_f1)

        stride_m = feat_m0.shape[2] // hc0
        stride_f = feat_f0.shape[2] // hc0

        m0 = ops.gather_windows(feat_m0, i_ids, wc0, self.W_m, stride_m)
        m1 = ops.gather_windows(feat_m1, j_ids, wc1, self.W_m, stride_m)
        f0 = ops.gather_windows(feat_f0, i_ids, wc0, self.W_f, stride_f)
        f1 = ops.gather_windows(feat_f1, j_ids, wc1, self.W_f, stride_f)

        # Self + cross attention at medium level (1/4, 3x3 + coarse token), kept stacked [2*M, L, C]
        m_both, _ = self.self_attn_m(
            torch.cat((m0, m1), dim=0), torch.cat((c0_sel, c1_sel), dim=0), drop_pre=True
        )
        m_both = self.cross_attn_m.forward_stacked(m_both)

        # Down-project medium window to fine dim, then self + cross attention at fine level (1/2, 5x5 + 3x3)
        m_both = self.down_proj_m_f(m_both)
        f_both, _ = self.self_attn_f(torch.cat((f0, f1), dim=0), m_both, drop_pre=True)
        f_both = self.cross_attn_f.forward_stacked(f_both)
        return f_both.chunk(2, dim=0)


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

        f0_proj = self.fine_proj(feat_f0_unfold)
        f1_proj = self.fine_proj(feat_f1_unfold)
        c_dim = feat_f0_unfold.shape[-1]
        if not denser:
            best_conf, best_i, best_j = ops.batched_dual_softmax_argmax(f0_proj, f1_proj, 1.0 / (c_dim * temperature))
            valid = best_conf > fine_threshold
            if not valid.any():
                b_ids = torch.zeros(1, device=best_conf.device, dtype=torch.long)
                i_ids = torch.zeros(1, device=best_conf.device, dtype=torch.long)
                j_ids = torch.zeros(1, device=best_conf.device, dtype=torch.long)
                mconf = torch.ones(1, device=best_conf.device, dtype=torch.float32)
            else:
                b_ids = torch.nonzero(valid)[:, 0]
                i_ids = best_i[b_ids]
                j_ids = best_j[b_ids]
                mconf = best_conf[b_ids]
        else:
            f0 = f0_proj.float() / (c_dim**0.5)
            f1 = f1_proj.float() / (c_dim**0.5)
            sim = torch.einsum("nlc,nsc->nls", f0, f1) / temperature
            conf = F.softmax(sim, dim=1) * F.softmax(sim, dim=2)
            mask = conf > fine_threshold
            if not mask.any():
                mask[0, 0, 0] = True
                conf[0, 0, 0] = 1.0
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


def _run_coarse_transformer_fused(
    transformer: LocalFeatureTransformer, feat0: Tensor, feat1: Tensor
) -> tuple[Tensor, Tensor]:
    """Fused execution of ``loftr_coarse``: packed QKV/KV GEMMs, batched self-attention when ``L0 == L1``, and
    ``ops.elu_linear_attention`` (5-7x faster than PyTorch ``linear_attention``). Leaves ``LoFTREncoderLayer`` and
    LoFTR itself untouched."""
    same_len = feat0.shape[1] == feat1.shape[1]
    for layer, name in zip(transformer.layers, transformer.layer_names):
        nhead, dim = layer.nhead, layer.dim
        gemm_dt = layer.attention_gemm_dtype
        if name == "self":
            qkv_w = getattr(layer, "_qkv_w", None)
            if same_len:
                x = torch.cat((feat0, feat1), dim=0)  # [2, L, C]
                bs, seq_len, c = x.shape
                if qkv_w is not None:
                    q, k, v = F.linear(x, qkv_w).reshape(bs, seq_len, 3, nhead, dim).unbind(2)
                else:
                    q = layer.q_proj(x).reshape(bs, seq_len, nhead, dim)
                    k = layer.k_proj(x).reshape(bs, seq_len, nhead, dim)
                    v = layer.v_proj(x).reshape(bs, seq_len, nhead, dim)
                msg = ops.elu_linear_attention(q, k, v, 1e-6, gemm_dt).reshape(bs, seq_len, c)
                mlp_in = ops.cat_layernorm(x, layer.merge(msg), layer.norm1.weight, layer.norm1.bias, layer.norm1.eps)
                feat0, feat1 = ops.residual_layernorm(
                    layer.mlp(mlp_in), x, layer.norm2.weight, layer.norm2.bias, layer.norm2.eps
                ).chunk(2, dim=0)
            else:
                outs = []
                for x in (feat0, feat1):
                    bs, seq_len, c = x.shape
                    if qkv_w is not None:
                        q, k, v = F.linear(x, qkv_w).reshape(bs, seq_len, 3, nhead, dim).unbind(2)
                    else:
                        q = layer.q_proj(x).reshape(bs, seq_len, nhead, dim)
                        k = layer.k_proj(x).reshape(bs, seq_len, nhead, dim)
                        v = layer.v_proj(x).reshape(bs, seq_len, nhead, dim)
                    msg = ops.elu_linear_attention(q, k, v, 1e-6, gemm_dt).reshape(bs, seq_len, c)
                    mlp_in = ops.cat_layernorm(x, layer.merge(msg), layer.norm1.weight, layer.norm1.bias, layer.norm1.eps)
                    outs.append(ops.residual_layernorm(layer.mlp(mlp_in), x, layer.norm2.weight, layer.norm2.bias, layer.norm2.eps))
                feat0, feat1 = outs[0], outs[1]
        else:
            kv_w = getattr(layer, "_kv_w", None)
            for idx in (0, 1):
                x, src = (feat0, feat1) if idx == 0 else (feat1, feat0)
                bs, l_q, c = x.shape
                l_s = src.shape[1]
                q = layer.q_proj(x).reshape(bs, l_q, nhead, dim)
                if kv_w is not None:
                    k, v = F.linear(src, kv_w).reshape(bs, l_s, 2, nhead, dim).unbind(2)
                else:
                    k = layer.k_proj(src).reshape(bs, l_s, nhead, dim)
                    v = layer.v_proj(src).reshape(bs, l_s, nhead, dim)
                msg = ops.elu_linear_attention(q, k, v, 1e-6, gemm_dt).reshape(bs, l_q, c)
                mlp_in = ops.cat_layernorm(x, layer.merge(msg), layer.norm1.weight, layer.norm1.bias, layer.norm1.eps)
                updated = ops.residual_layernorm(layer.mlp(mlp_in), x, layer.norm2.weight, layer.norm2.bias, layer.norm2.eps)
                if idx == 0:
                    feat0 = updated
                else:
                    feat1 = updated
    return feat0, feat1


class XoFTR(ImageMatcher, nn.Module):
    """Module paths follow the official ``weights_xoftr_640.ckpt`` checkpoint (``backbone``, ``loftr_coarse``,
    ``coarse_matching``, ``fine_process``, ``fine_matching``); loads strictly. ``match(img_a, img_b)`` ->
    :class:`Matches` in each original image's pixel coordinates (default working long side: 640 px)."""

    div_by = 8
    match_defaults: Mapping[str, Any] = {"resize": 640}
    fold_temperature = False  # see spec.py's fold_similarity_temperature pass
    fused_coarse = False  # see spec.py's fused_coarse_transformer pass
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
        if self.fused_coarse:
            tokens0, tokens1 = _run_coarse_transformer_fused(self.loftr_coarse, tokens0, tokens1)
        else:
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
