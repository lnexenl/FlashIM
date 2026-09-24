"""RoMaV2 coarse matcher: multi-view ViT over both images + DPT warp/confidence head."""

from __future__ import annotations

import torch
from einops import rearrange
from torch import Tensor, nn

from flashim.models.romav2.dpt import DPTHead
from flashim.models.romav2.geometry import normalized_grid
from flashim.models.romav2.vit import Block, RopeEmbedding


class MatchTransformer(nn.Module):
    """ViT-B over the tokens of both views, alternating global and per-view attention.

    Even blocks attend jointly over both views without RoPE; odd blocks attend within
    each view with RoPE (BF16 tables).
    """

    multiview = True
    mv_attention_mode = "alternating"

    def __init__(self, in_dim: int, out_dim: int, dim: int = 768, depth: int = 12, num_heads: int = 12) -> None:
        super().__init__()
        self.projector = nn.Linear(in_dim, dim)
        self.output_projector = nn.Linear(dim, out_dim)
        self.blocks = nn.ModuleList(
            Block(dim, num_heads, eps=1e-6, layer_scale=False, mask_k_bias=False, rope_impl=None)
            for _ in range(depth)
        )
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.rope_embed = RopeEmbedding(dim // num_heads, base=100.0, compute_dtype=torch.bfloat16)

    def forward(self, x: Tensor) -> Tensor:
        """``[B, 2, h, w, in_dim]`` -> ``[B, 2*h*w, out_dim]`` (``x_norm_patchtokens``)."""
        _, _, h, w, _ = x.shape
        x = self.projector(x.flatten(1, -2))
        rope = self.rope_embed(h, w)
        for index, block in enumerate(self.blocks):
            if index % 2 == 1:
                x = rearrange(x, "B (V H W) D -> (B V) (H W) D", V=2, H=h, W=w)
                x = block(x, rope)
                x = rearrange(x, "(B V) (H W) D -> B (V H W) D", V=2, H=h, W=w)
            else:
                x = block(x, None)
        return self.output_projector(self.norm(x))


def _cosine_similarity(f_a: Tensor, f_b: Tensor) -> Tensor:
    f_a = f_a / f_a.norm(dim=-1, keepdim=True)
    f_b = f_b / f_b.norm(dim=-1, keepdim=True)
    return torch.einsum("bijd,bkld->bijkl", f_a, f_b)


class Matcher(nn.Module):
    def __init__(self, dim: int = 1024, feat_dim: int = 1024, num_feature_layers: int = 2) -> None:
        super().__init__()
        self.dim = dim
        self.register_buffer("omega", 2 * torch.pi * torch.randn(dim // 2, 2))
        self.register_buffer("scale", torch.tensor(1))
        self.register_buffer("temp", torch.tensor(0.1))
        self.mv_vit = MatchTransformer(in_dim=feat_dim * num_feature_layers, out_dim=dim)
        self.head = DPTHead(dim_in=dim, out_dim=3, down_ratio=4)

    def _match_embedding(self, f_a: Tensor, f_b: Tensor, pos_emb: Tensor) -> tuple[Tensor, Tensor]:
        batch, h, w, _ = f_a.shape
        logits = (1 / self.temp * _cosine_similarity(f_a, f_b)).reshape(batch, h * w, h * w)
        attn = torch.softmax(logits, dim=2).reshape(batch, h, w, h, w)
        match_emb = torch.einsum("bijkl,bkld->bijd", attn, pos_emb)
        return logits.reshape(batch, h, w, h, w), match_emb

    def _head(self, f_list: list[Tensor], f_mv: Tensor, match_emb: Tensor) -> tuple[Tensor, Tensor]:
        head_input = [*f_list[:-1], f_list[-1] + f_mv + match_emb]
        out = self.head(head_input)
        return out[..., :2], out[..., 2:]

    def forward(self, f_list_a: list[Tensor], f_list_b: list[Tensor], bidirectional: bool) -> dict[str, Tensor | None]:
        f_a = torch.cat(f_list_a, dim=-1)
        f_b = torch.cat(f_list_b, dim=-1)
        batch, h, w, _ = f_a.shape
        if f_b.shape[1:3] != (h, w):
            raise ValueError("RoMaV2 matcher requires equally sized coarse grids")
        grid = normalized_grid(batch, h, w, f_a.device)
        x_emb = nn.functional.linear(grid.reshape(batch, h * w, 2), self.scale * self.omega).reshape(
            batch, h, w, -1
        )
        pos_emb = torch.cat((x_emb.sin(), x_emb.cos()), dim=-1)

        with torch.autocast(f_a.device.type, torch.bfloat16):
            f_mv = self.mv_vit(torch.stack((f_a, f_b), dim=1)).reshape(batch, 2, h, w, self.dim)
        f_mv_a, f_mv_b = f_mv[:, 0].float(), f_mv[:, 1].float()

        logits_ab, emb_ab = self._match_embedding(f_mv_a, f_mv_b, pos_emb)
        warp_ab, conf_ab = self._head(f_list_a, f_mv_a, emb_ab)
        out = {"attn_AB_logits": logits_ab, "warp_AB": warp_ab, "confidence_AB": conf_ab}
        if bidirectional:
            logits_ba, emb_ba = self._match_embedding(f_mv_b, f_mv_a, pos_emb)
            warp_ba, conf_ba = self._head(f_list_b, f_mv_b, emb_ba)
            out.update(attn_BA_logits=logits_ba, warp_BA=warp_ba, confidence_BA=conf_ba)
        else:
            out.update(attn_BA_logits=None, warp_BA=None, confidence_BA=None)
        return out
