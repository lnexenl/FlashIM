"""Reusable CroCo / DUSt3R stereo encoder, cross-attention decoder, 2D RoPE and DPT head.

Built from :class:`flashim.layers.TransformerBlock` / :class:`flashim.layers.Attention` /
:class:`flashim.layers.Mlp` with exact upstream parameter names so MASt3R and DUSt3R checkpoints load with
``strict=True`` and FlashIM's transformer quantization/fusion passes apply by layer type.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from flashim import ops
from flashim.layers.transformer import Attention, Mlp, TransformerBlock


class RoPE2D(nn.Module):
    """CroCo-style 2D axial rotary embedding tables cached per ``(head_dim, h, w, device, dtype)``.

    Each head vector of width ``D`` is split into ``(y, x)`` halves of width ``D // 2``, each rotated with
    frequencies ``F0 / (base ** (arange(0, D//2, 2) / (D//2)))`` indexed by integer patch coordinates ``(y, x)``.
    Returns ``(sin, cos)`` of shape ``[h * w, D]`` matching :data:`flashim.ops.rope_qk_2d`.
    """

    def __init__(self, freq: float = 100.0, f0: float = 1.0) -> None:
        super().__init__()
        self.base = float(freq)
        self.f0 = float(f0)
        self._grid_cache: dict[tuple[int, int, int, torch.device, torch.dtype], tuple[Tensor, Tensor]] = {}
        self._1d_cache: dict[tuple[int, int, torch.device, torch.dtype], tuple[Tensor, Tensor]] = {}

    def _cos_sin_1d(self, half_dim: int, length: int, device: torch.device, dtype: torch.dtype) -> tuple[Tensor, Tensor]:
        key = (half_dim, length, device, dtype)
        cached = self._1d_cache.get(key)
        if cached is not None:
            return cached
        inv_freq = self.f0 / (self.base ** (torch.arange(0, half_dim, 2, device=device, dtype=torch.float32) / half_dim))
        t = torch.arange(length, device=device, dtype=torch.float32)
        freqs = torch.einsum("i,j->ij", t, inv_freq).to(dtype)
        freqs = torch.cat((freqs, freqs), dim=-1)
        table = (freqs.sin(), freqs.cos())
        self._1d_cache[key] = table
        return table

    def for_grid(self, head_dim: int, h: int, w: int, device: torch.device, dtype: torch.dtype) -> tuple[Tensor, Tensor]:
        """``(sin, cos)`` of shape ``[h * w, head_dim]`` for an ``h x w`` row-major patch grid."""
        key = (head_dim, h, w, device, dtype)
        cached = self._grid_cache.get(key)
        if cached is not None:
            return cached
        half_dim = head_dim // 2
        sin_1d, cos_1d = self._cos_sin_1d(half_dim, max(h, w), device, dtype)
        y_idx = torch.arange(h, device=device)
        x_idx = torch.arange(w, device=device)
        gy, gx = torch.meshgrid(y_idx, x_idx, indexing="ij")
        gy_flat, gx_flat = gy.reshape(-1), gx.reshape(-1)
        sin_2d = torch.cat((sin_1d[gy_flat], sin_1d[gx_flat]), dim=-1)
        cos_2d = torch.cat((cos_1d[gy_flat], cos_1d[gx_flat]), dim=-1)
        self._grid_cache[key] = (sin_2d, cos_2d)
        return sin_2d, cos_2d


class PatchEmbed(nn.Module):
    """Non-overlapping conv patch embedding (upstream ``PatchEmbedDust3R`` parameter names: ``proj``)."""

    def __init__(self, patch_size: int = 16, in_chans: int = 3, embed_dim: int = 1024) -> None:
        super().__init__()
        self.patch_size = (patch_size, patch_size)
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = nn.Identity()

    def forward(self, x: Tensor) -> tuple[Tensor, tuple[int, int]]:
        _, _, h, w = x.shape
        ph, pw = self.patch_size
        if h % ph != 0 or w % pw != 0:
            raise ValueError(f"Input image size ({h}, {w}) must be divisible by patch_size ({ph}, {pw})")
        y = self.proj(x)
        grid = (y.shape[2], y.shape[3])
        return self.norm(y.flatten(2).transpose(1, 2).contiguous()), grid


class CrossAttention(nn.Module):
    """Multi-head cross-attention with CroCo parameter names (``projq`` / ``projk`` / ``projv`` / ``proj``) and
    2D RoPE via :data:`flashim.ops.rope_qk_2d` + :data:`flashim.ops.sdpa`."""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        *,
        qkv_bias: bool = True,
        rope_impl: str | None = None,
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.projq = nn.Linear(dim, dim, bias=qkv_bias)
        self.projk = nn.Linear(dim, dim, bias=qkv_bias)
        self.projv = nn.Linear(dim, dim, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)
        self.rope_impl = rope_impl
        self.sdpa_impl: str | None = "reference"

    def forward(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        q_rope: tuple[Tensor, Tensor] | None = None,
        k_rope: tuple[Tensor, Tensor] | None = None,
    ) -> Tensor:
        batch, nq, channels = query.shape
        nk = key.shape[1]
        head_dim = channels // self.num_heads
        q = self.projq(query).reshape(batch, nq, self.num_heads, head_dim).permute(0, 2, 1, 3)
        k = self.projk(key).reshape(batch, nk, self.num_heads, head_dim).permute(0, 2, 1, 3)
        v = self.projv(value).reshape(batch, nk, self.num_heads, head_dim).permute(0, 2, 1, 3)
        if q_rope is not None:
            if k_rope is None:
                k_rope = q_rope
            q, k = ops.rope_qk_2d.get(self.rope_impl)(q, k, q_rope[0], q_rope[1], k_rope[0], k_rope[1])
        out = ops.sdpa.get(self.sdpa_impl)(q, k, v)
        return self.proj(out.transpose(1, 2).reshape(batch, nq, channels))


class DecoderBlock(nn.Module):
    """CroCo cross-attention decoder block: self-attention -> cross-attention (normalized memory) -> MLP."""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        *,
        mlp_ratio: int = 4,
        eps: float = 1e-6,
        norm_mem: bool = True,
        rope_impl: str | None = None,
    ) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, eps=eps)
        self.attn = Attention(dim, num_heads, rope_impl=rope_impl, rope_2d=True)
        self.cross_attn = CrossAttention(dim, num_heads, qkv_bias=True, rope_impl=rope_impl)
        self.norm2 = nn.LayerNorm(dim, eps=eps)
        self.norm3 = nn.LayerNorm(dim, eps=eps)
        self.mlp = Mlp(dim, int(dim * mlp_ratio))
        self.norm_y = nn.LayerNorm(dim, eps=eps) if norm_mem else nn.Identity()

    def forward(
        self,
        x: Tensor,
        y: Tensor,
        x_rope: tuple[Tensor, Tensor],
        y_rope: tuple[Tensor, Tensor],
    ) -> Tensor:
        x = x + self.attn(self.norm1(x), x_rope)
        y_norm = self.norm_y(y)
        x = x + self.cross_attn(self.norm2(x), y_norm, y_norm, x_rope, y_rope)
        return x + self.mlp(self.norm3(x))


class CroCoStereoBackbone(nn.Module):
    """Siamese CroCo ViT encoder + asymmetric two-branch cross-attention decoder shared by DUSt3R and MASt3R."""

    def __init__(
        self,
        *,
        patch_size: int = 16,
        enc_embed_dim: int = 1024,
        enc_depth: int = 24,
        enc_num_heads: int = 16,
        dec_embed_dim: int = 768,
        dec_depth: int = 12,
        dec_num_heads: int = 12,
        mlp_ratio: int = 4,
        norm_eps: float = 1e-6,
        norm_im2_in_dec: bool = True,
        rope_freq: float = 100.0,
    ) -> None:
        super().__init__()
        self.patch_size = patch_size
        self.enc_embed_dim = enc_embed_dim
        self.enc_depth = enc_depth
        self.enc_num_heads = enc_num_heads
        self.dec_embed_dim = dec_embed_dim
        self.dec_depth = dec_depth
        self.dec_num_heads = dec_num_heads

        self.patch_embed = PatchEmbed(patch_size=patch_size, in_chans=3, embed_dim=enc_embed_dim)
        self.rope = RoPE2D(freq=rope_freq)

        self.enc_blocks = nn.ModuleList(
            TransformerBlock(
                enc_embed_dim,
                enc_num_heads,
                eps=norm_eps,
                layer_scale=False,
                rope_impl="reference",
                rope_2d=True,
                mlp_ratio=mlp_ratio,
            )
            for _ in range(enc_depth)
        )
        self.enc_norm = nn.LayerNorm(enc_embed_dim, eps=norm_eps)

        # Retained from CroCo pretraining checkpoints so strict=True loading succeeds.
        self.mask_token = nn.Parameter(torch.zeros(1, 1, dec_embed_dim))

        self.decoder_embed = nn.Linear(enc_embed_dim, dec_embed_dim, bias=True)
        self.dec_blocks = nn.ModuleList(
            DecoderBlock(
                dec_embed_dim,
                dec_num_heads,
                mlp_ratio=mlp_ratio,
                eps=norm_eps,
                norm_mem=norm_im2_in_dec,
                rope_impl="reference",
            )
            for _ in range(dec_depth)
        )
        self.dec_blocks2 = nn.ModuleList(
            DecoderBlock(
                dec_embed_dim,
                dec_num_heads,
                mlp_ratio=mlp_ratio,
                eps=norm_eps,
                norm_mem=norm_im2_in_dec,
                rope_impl="reference",
            )
            for _ in range(dec_depth)
        )
        self.dec_norm = nn.LayerNorm(dec_embed_dim, eps=norm_eps)

    def encode_image(self, image: Tensor) -> tuple[Tensor, tuple[int, int]]:
        """Encode ``[B, 3, H, W]`` image -> ``(tokens [B, N, enc_dim], (h_patches, w_patches))``."""
        x, grid = self.patch_embed(image)
        rope = self.rope.for_grid(self.enc_embed_dim // self.enc_num_heads, grid[0], grid[1], x.device, x.dtype)
        for blk in self.enc_blocks:
            x = blk(x, rope)
        return self.enc_norm(x), grid

    def encode_pair(self, img1: Tensor, img2: Tensor) -> tuple[Tensor, Tensor, tuple[int, int], tuple[int, int]]:
        """Encode an image pair, batching the encoder pass when both images share spatial shape."""
        if img1.shape[-2:] == img2.shape[-2:]:
            out, grid = self.encode_image(torch.cat((img1, img2), dim=0))
            feat1, feat2 = out.chunk(2, dim=0)
            return feat1, feat2, grid, grid
        feat1, grid1 = self.encode_image(img1)
        feat2, grid2 = self.encode_image(img2)
        return feat1, feat2, grid1, grid2

    def decode(
        self,
        feat1: Tensor,
        grid1: tuple[int, int],
        feat2: Tensor,
        grid2: tuple[int, int],
    ) -> tuple[list[Tensor], list[Tensor]]:
        """Run the two-branch cross-attention decoder and return per-layer token lists ``(dec1, dec2)``
        of length ``dec_depth + 1`` (index 0 = encoder tokens, index ``dec_depth`` = final ``dec_norm``'ed tokens)."""
        dec_head_dim = self.dec_embed_dim // self.dec_num_heads
        f1 = self.decoder_embed(feat1)
        f2 = self.decoder_embed(feat2)
        rope1 = self.rope.for_grid(dec_head_dim, grid1[0], grid1[1], f1.device, f1.dtype)
        rope2 = rope1 if grid2 == grid1 else self.rope.for_grid(dec_head_dim, grid2[0], grid2[1], f2.device, f2.dtype)

        dec1: list[Tensor] = [feat1]
        dec2: list[Tensor] = [feat2]
        cur1, cur2 = f1, f2
        for blk1, blk2 in zip(self.dec_blocks, self.dec_blocks2):
            nxt1 = blk1(cur1, cur2, rope1, rope2)
            nxt2 = blk2(cur2, cur1, rope2, rope1)
            cur1, cur2 = nxt1, nxt2
            dec1.append(cur1)
            dec2.append(cur2)
        dec1[-1] = self.dec_norm(dec1[-1])
        dec2[-1] = self.dec_norm(dec2[-1])
        return dec1, dec2


# -- CroCo / DUSt3R DPT head & post-processing (reusable by DUSt3R) --------------------------------------


class ResidualConvUnit(nn.Module):
    """CroCo ``ResidualConvUnit_custom`` (out-of-place ReLU so skip input is not rectified)."""

    def __init__(self, features: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(features, features, 3, padding=1, bias=True)
        self.conv2 = nn.Conv2d(features, features, 3, padding=1, bias=True)
        self.activation = nn.ReLU(inplace=False)

    def forward(self, x: Tensor) -> Tensor:
        out = self.conv2(self.activation(self.conv1(self.activation(x))))
        return out + x


class FeatureFusionBlock(nn.Module):
    """CroCo ``FeatureFusionBlock_custom`` (both ``resConfUnit1`` and ``resConfUnit2`` instantiated to match
    upstream checkpoints strictly, even on ``refinenet4`` where ``resConfUnit1`` is unused)."""

    def __init__(self, features: int) -> None:
        super().__init__()
        self.out_conv = nn.Conv2d(features, features, 1, bias=True)
        self.resConfUnit1 = ResidualConvUnit(features)
        self.resConfUnit2 = ResidualConvUnit(features)

    def forward(self, *xs: Tensor) -> Tensor:
        output = xs[0]
        if len(xs) == 2:
            output = output + self.resConfUnit1(xs[1])
        output = self.resConfUnit2(output)
        output = F.interpolate(output, scale_factor=2, mode="bilinear", align_corners=True)
        return self.out_conv(output)


class Interpolate(nn.Module):
    def __init__(self, scale_factor: float, mode: str = "bilinear", align_corners: bool = True) -> None:
        super().__init__()
        self.scale_factor = scale_factor
        self.mode = mode
        self.align_corners = align_corners

    def forward(self, x: Tensor) -> Tensor:
        return F.interpolate(x, scale_factor=self.scale_factor, mode=self.mode, align_corners=self.align_corners)


class DPTOutputAdapter(nn.Module):
    """CroCo / DUSt3R ``DPTOutputAdapter_fix`` with exact checkpoint parameter names."""

    def __init__(
        self,
        *,
        num_channels: int = 4,
        patch_size: int = 16,
        hooks: tuple[int, ...] = (0, 6, 9, 12),
        dim_tokens: tuple[int, ...] = (1024, 768, 768, 768),
        layer_dims: tuple[int, ...] = (96, 192, 384, 768),
        feature_dim: int = 256,
        last_dim: int = 128,
    ) -> None:
        super().__init__()
        self.patch_size = patch_size
        self.hooks = tuple(hooks)

        scratch = nn.Module()
        for idx, dim in enumerate(layer_dims, start=1):
            setattr(scratch, f"layer{idx}_rn", nn.Conv2d(dim, feature_dim, 3, padding=1, bias=False))
        # Upstream registers layer_rn aliasing layer{1..4}_rn so state_dict holds both keys.
        scratch.layer_rn = nn.ModuleList(
            [scratch.layer1_rn, scratch.layer2_rn, scratch.layer3_rn, scratch.layer4_rn]
        )
        scratch.refinenet1 = FeatureFusionBlock(feature_dim)
        scratch.refinenet2 = FeatureFusionBlock(feature_dim)
        scratch.refinenet3 = FeatureFusionBlock(feature_dim)
        scratch.refinenet4 = FeatureFusionBlock(feature_dim)
        self.scratch = scratch

        self.head = nn.Sequential(
            nn.Conv2d(feature_dim, feature_dim // 2, 3, padding=1),
            Interpolate(scale_factor=2, mode="bilinear", align_corners=True),
            nn.Conv2d(feature_dim // 2, last_dim, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(last_dim, num_channels, 1),
        )

        self.act_postprocess = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(dim_tokens[0], layer_dims[0], 1),
                    nn.ConvTranspose2d(layer_dims[0], layer_dims[0], 4, stride=4),
                ),
                nn.Sequential(
                    nn.Conv2d(dim_tokens[1], layer_dims[1], 1),
                    nn.ConvTranspose2d(layer_dims[1], layer_dims[1], 2, stride=2),
                ),
                nn.Sequential(nn.Conv2d(dim_tokens[2], layer_dims[2], 1)),
                nn.Sequential(
                    nn.Conv2d(dim_tokens[3], layer_dims[3], 1),
                    nn.Conv2d(layer_dims[3], layer_dims[3], 3, stride=2, padding=1),
                ),
            ]
        )

    def forward(self, tokens: list[Tensor], image_size: tuple[int, int]) -> Tensor:
        h, w = image_size
        nh, nw = h // self.patch_size, w // self.patch_size
        layers = [tokens[hook] for hook in self.hooks]
        if self.act_postprocess[0][0].weight.is_contiguous(memory_format=torch.channels_last):
            # Tokens are [B, nh*nw, C] contiguous = NHWC memory layout: zero-copy channels_last view.
            layers = [t.reshape(t.shape[0], nh, nw, t.shape[2]).permute(0, 3, 1, 2) for t in layers]
        else:
            layers = [t.transpose(1, 2).reshape(t.shape[0], t.shape[2], nh, nw) for t in layers]
        layers = [post(feat) for post, feat in zip(self.act_postprocess, layers)]
        layers = [rn(feat) for rn, feat in zip(self.scratch.layer_rn, layers)]

        path_4 = self.scratch.refinenet4(layers[3])[:, :, : layers[2].shape[2], : layers[2].shape[3]]
        path_3 = self.scratch.refinenet3(path_4, layers[2])
        path_2 = self.scratch.refinenet2(path_3, layers[1])
        path_1 = self.scratch.refinenet1(path_2, layers[0])
        return self.head(path_1)


def reg_dense_depth(xyz: Tensor, mode: tuple[str, float, float]) -> Tensor:
    """Convert raw 3-channel head output ``[..., 3]`` into metric 3D points (DUSt3R/MASt3R convention)."""
    kind, vmin, vmax = mode
    if kind == "linear":
        return xyz if (vmin == float("-inf") and vmax == float("inf")) else xyz.clamp(min=vmin, max=vmax)
    d = xyz.norm(dim=-1, keepdim=True)
    unit = xyz / d.clamp(min=1e-8)
    if kind == "square":
        return unit * d.square()
    if kind == "exp":
        return unit * torch.expm1(d)
    raise ValueError(f"Unknown depth_mode {kind!r}")


def reg_dense_conf(x: Tensor, mode: tuple[str, float, float]) -> Tensor:
    """Convert raw confidence map into positive confidence values."""
    kind, vmin, vmax = mode
    if kind == "exp":
        return vmin + x.exp().clamp(max=vmax - vmin)
    if kind == "sigmoid":
        return (vmax - vmin) * torch.sigmoid(x) + vmin
    raise ValueError(f"Unknown conf_mode {kind!r}")
