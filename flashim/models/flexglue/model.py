"""FlexGlue matcher (native FlashIM port of cvg/FlexGlue).

Sparse-to-dense matcher: dense DINOv3-ConvNeXt search maps from :mod:`~flashim.models.flexglue.descriptor`, a
9-layer transformer over the search-map tokens, and per-layer correspondence heads that regress the position of a
query point in the other image through a 1.5-entmax readout of the search map. Matches keypoints of one image to
sub-pixel correspondences in the other and keeps the mutual nearest neighbours; without keypoints it samples
balanced correspondences from the dense warp.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from flashim import ops
from flashim.models.base import ImageMatcher
from flashim.models.flexglue import weights
from flashim.models.flexglue.config import FlexGlueConfig
from flashim.models.flexglue.descriptor import DINOv3Descriptor
from flashim.models.romav2.model import ImageLike, load_image
from flashim.platforms import default_device
from flashim.types import Matches
from flashim.utils import device_constant


def normalize_keypoints(keypoints: Tensor, image_size: Tensor) -> Tensor:
    """Points ``(b, ..., 2)`` of a ``(w, h)`` image with ``size (b, 2)`` to ``[-1, 1]``."""
    size = image_size.view(image_size.shape[0], *([1] * (keypoints.dim() - 2)), 2)
    return keypoints / size * 2 - 1


def denormalize_keypoints(keypoints: Tensor, image_size: Tensor) -> Tensor:
    """Points ``(b, ..., 2)`` from ``[-1, 1]`` to pixels of a ``(w, h)`` image."""
    size = image_size.view(image_size.shape[0], *([1] * (keypoints.dim() - 2)), 2)
    return (keypoints + 1) / 2 * size


def cell_centers(h: int, w: int, scale: tuple, device: torch.device) -> Tensor:
    """Normalized ``(x, y)`` centers ``(h*w, 2)`` of an ``h x w`` grid, divided by ``(sx, sy)``."""
    y, x = torch.meshgrid(torch.arange(h, device=device), torch.arange(w, device=device), indexing="ij")
    xy = torch.stack([(x + 0.5) / scale[0], (y + 0.5) / scale[1]], -1)
    return xy.flatten(0, 1) * 2 - 1


def sample_features(features: Tensor, points_n: Tensor) -> Tensor:
    """Bilinearly sample ``(B, C, H, W)`` features at points ``(B, N, 2)`` in ``[-1, 1]`` -> ``(B, N, C)``."""
    points_n = points_n.to(features.dtype)
    out = F.grid_sample(features, points_n[:, None], "bilinear", padding_mode="border", align_corners=False)
    return out[:, :, 0].transpose(1, 2)


def rotate_half(x: Tensor) -> Tensor:
    x1, x2 = x.unflatten(-1, (-1, 2)).unbind(-1)
    return torch.stack((-x2, x1), -1).flatten(-2)


def apply_rotary(encoding: Tensor, t: Tensor) -> Tensor:
    """``t * cos + rotate_half(t) * sin`` (interleaved-pair RoPE).

    ``t`` is ``(B, N, H, D)``; ``encoding`` is ``[2, B, N, 1, D]`` (cos, sin) and broadcasts over the heads.
    """
    return t * encoding[0] + rotate_half(t) * encoding[1]


def _mlp(dims: list) -> nn.Sequential:
    layers = []
    for i in range(1, len(dims)):
        layers.append(nn.Linear(dims[i - 1], dims[i]))
        if i < len(dims) - 1:
            layers += [nn.LayerNorm(dims[i]), nn.ReLU(inplace=True)]
    return nn.Sequential(*layers)


def _ffn(dim: int) -> nn.Sequential:
    return nn.Sequential(nn.Linear(2 * dim, 2 * dim), nn.LayerNorm(2 * dim), nn.GELU(), nn.Linear(2 * dim, dim))


def _attention(q: Tensor, k: Tensor, v: Tensor, impl: str = "reference") -> Tensor:
    """Attention (``ops.sdpa`` impl ``impl``) on token-major ``(B, N, H, D)`` tensors, returns ``(B, N, H * D)``."""
    out = ops.sdpa.get(impl)(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2))
    return out.transpose(1, 2).flatten(-2)


class SelfBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.sdpa_impl = "reference"  # ``optimize.attention`` (spec.set_attention)
        self.Wqkv = nn.Linear(dim, 3 * dim)
        self.q_norm = nn.LayerNorm(dim // num_heads)
        self.k_norm = nn.LayerNorm(dim // num_heads)
        self.out_proj = nn.Linear(dim, dim)
        self.message_norm = nn.LayerNorm(dim)
        self.ffn = _ffn(dim)

    def forward(self, x: Tensor, encoding: Tensor) -> Tensor:
        qkv = self.Wqkv(x)  # token-major (B, N, H, D) q, k, v; checkpoint layout (H, D, 3)
        q, k, v = qkv.unflatten(-1, (self.num_heads, -1, 3)).unbind(-1)
        q, k = self.q_norm(q).to(v.dtype), self.k_norm(k).to(v.dtype)
        encoding = encoding.transpose(-3, -2)  # broadcast the tables over (N, H) instead of (1, N)
        q, k = apply_rotary(encoding, q), apply_rotary(encoding, k)
        q, k = q.to(v.dtype), k.to(v.dtype)  # the fp32 tables promote the rope back to fp32
        message = self.out_proj(_attention(q, k, v.contiguous(), self.sdpa_impl))
        return x + self.ffn(torch.cat([x, self.message_norm(message)], -1))


class CrossBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.sdpa_impl = "reference"
        self.to_qk = nn.Linear(dim, dim)
        self.to_v = nn.Linear(dim, dim)
        self.to_out = nn.Linear(dim, dim)
        self.qk_norm = nn.LayerNorm(dim // num_heads)
        self.message_norm = nn.LayerNorm(dim)
        self.ffn = _ffn(dim)

    def forward(self, x0: Tensor, x1: Tensor) -> tuple[Tensor, Tensor]:
        qk0, qk1, v0, v1 = (  # token-major (B, N, H, D)
            t.unflatten(-1, (self.num_heads, -1))
            for t in (self.to_qk(x0), self.to_qk(x1), self.to_v(x0), self.to_v(x1))
        )
        qk0, qk1 = self.qk_norm(qk0).to(v0.dtype), self.qk_norm(qk1).to(v1.dtype)
        m0 = self.to_out(_attention(qk0, qk1, v1, self.sdpa_impl))
        m1 = self.to_out(_attention(qk1, qk0, v0, self.sdpa_impl))
        x0 = x0 + self.ffn(torch.cat([x0, self.message_norm(m0)], -1))
        x1 = x1 + self.ffn(torch.cat([x1, self.message_norm(m1)], -1))
        return x0, x1

    def forward_stacked(self, x: Tensor) -> Tensor:
        """``forward`` on both images stacked on the batch, ``x = cat([x0, x1])`` of equal token counts: one call
        per linear and one attention over the 2B pairs (image 0 attends to image 1 and vice versa)."""
        b = x.shape[0] // 2
        qk = self.qk_norm(self.to_qk(x).unflatten(-1, (self.num_heads, -1)))
        v = self.to_v(x).unflatten(-1, (self.num_heads, -1))
        qk = qk.to(v.dtype)
        swap = lambda t: torch.cat([t[b:], t[:b]])  # the other image of each pair
        message = self.to_out(_attention(qk, swap(qk), swap(v), self.sdpa_impl))
        return x + self.ffn(torch.cat([x, self.message_norm(message)], -1))


class TransformerLayer(nn.Module):
    def __init__(self, dim: int, num_heads: int) -> None:
        super().__init__()
        self.self_attn = SelfBlock(dim, num_heads)
        self.cross_attn = CrossBlock(dim, num_heads)

    def forward(self, x0: Tensor, x1: Tensor, enc0: Tensor, enc1: Tensor) -> tuple[Tensor, Tensor]:
        if x0.shape == x1.shape and enc0.shape == enc1.shape:
            # equal grids (always, at a fixed describe_resize): the encodings depend on the grid only, so both
            # images run stacked on the batch (half the kernel launches, twice the GEMM rows)
            x = self.cross_attn.forward_stacked(self.self_attn(torch.cat([x0, x1]), enc0))
            return x.chunk(2)
        return self.cross_attn(self.self_attn(x0, enc0), self.self_attn(x1, enc1))


class OffsetHead(nn.Module):
    """Sub-cell offset as a soft-argmax over bins spanning +-1 cell."""

    def __init__(self, dim: int, hidden: int = 256, bins: int = 16) -> None:
        super().__init__()
        self.bins = bins
        self.query_encoder = _mlp([dim, hidden, hidden])
        self.match_encoder = _mlp([dim, hidden, hidden])
        self.head = _mlp([2 * hidden, 2 * hidden, 2 * (bins + 2)])  # last bin unused

    def forward(self, f_query: Tensor, f_match: Tensor, hw: tuple[int, int]) -> Tensor:
        x = torch.cat([self.query_encoder(f_query), self.match_encoder(f_match)], -1)
        logits = self.head(x).unflatten(-1, (2, self.bins + 2))[..., : self.bins + 1]
        bins = torch.arange(self.bins + 1, device=x.device, dtype=x.dtype)
        delta = ((logits.softmax(-1) * bins).sum(-1) / self.bins - 0.5) * 2
        return torch.stack([delta[..., 0] * 2 / hw[1], delta[..., 1] * 2 / hw[0]], -1)


class MatchHead(nn.Module):
    """Regress the correspondence of each point in the other image's search map."""

    def __init__(self, dim: int, search_dim: int, query_chunk_size: int | None, cross: bool = True) -> None:
        super().__init__()
        self.query_chunk_size = query_chunk_size
        self.temperature = nn.Parameter(torch.tensor(10.0))
        self.to_search = nn.Linear(dim, search_dim)
        # layer 0 sees no cross-view context: no matchability, no covisibility
        self.matchability = nn.Linear(dim, 1) if cross else None
        self.offset_head = OffsetHead(dim)
        self.covisibility = nn.Linear(dim, 1) if cross else None

    def forward(self, points_ni: Tensor, cf_i: Tensor, cf_j: Tensor) -> tuple[Tensor, Tensor]:
        """Normalized points ``(B, N, 2)`` of i -> predicted points in j and confidence."""
        search = F.normalize(self.to_search(cf_j.flatten(2).mT), dim=-1)
        if self.query_chunk_size is None:
            return self.match_points(points_ni, cf_i, cf_j, search)
        chunks = [self.match_points(points, cf_i, cf_j, search) for points in points_ni.split(self.query_chunk_size, dim=1)]
        if len(chunks) == 1:
            return chunks[0]
        p2di_nj, confidence = zip(*chunks)
        return torch.cat(p2di_nj, 1), torch.cat(confidence, 1)

    def match_points(self, points_ni, cf_i, cf_j, search) -> tuple[Tensor, Tensor]:
        """forward on one chunk of points, with the normalized search features of j."""
        hw = cf_j.shape[-2:]
        f_query = sample_features(cf_i, points_ni)
        query = F.normalize(self.to_search(f_query), dim=-1)
        logits = query @ search.mT * self.temperature
        grid = cell_centers(*hw, scale=(hw[1], hw[0]), device=logits.device)
        p2di_nj, window_mass = ops.flexglue_entmax_readout(logits, grid, hw)
        f_match = sample_features(cf_j, p2di_nj)
        p2di_nj = p2di_nj + self.offset_head(f_query, f_match, hw)
        if self.matchability is None:
            return p2di_nj, window_mass
        matchability = torch.sigmoid(self.matchability(f_query).squeeze(-1))
        return p2di_nj, matchability * window_mass


def _mutual_nn(dist: Tensor) -> tuple[Tensor, Tensor]:
    """Mutual nearest neighbors of a ``(B, M, N)`` distance matrix, -1 if not mutual."""
    m0, m1 = dist.argmin(2), dist.argmin(1)
    mutual0 = torch.arange(m0.shape[1], device=m0.device) == m1.gather(1, m0)
    mutual1 = torch.arange(m1.shape[1], device=m1.device) == m0.gather(1, m1)
    return torch.where(mutual0, m0, -1), torch.where(mutual1, m1, -1)


def _sq_dist_gemm(x: Tensor, y: Tensor) -> Tensor:
    """Squared distance ``(B, M, N)`` of ``x (B, M, D)`` and ``y (B, N, D)`` in one GEMM, no sqrt."""
    ones_x, ones_y = x.new_ones(*x.shape[:-1], 1), y.new_ones(*y.shape[:-1], 1)
    xa = torch.cat([-2 * x, x.square().sum(-1, keepdim=True), ones_x], -1)
    ya = torch.cat([y, ones_y, y.square().sum(-1, keepdim=True)], -1)
    return xa @ ya.mT


def _grid_sources(h: int, w: int, device: torch.device) -> Tensor:
    """Normalized source position ``(h*w, 2)`` of every cell of an ``h x w`` grid."""
    return cell_centers(h, w, scale=(w, h), device=device)


def _kde_balance_weights(dense_p2d0_n1, dense_p2d1_n0, scores, num_samples, std=0.1, min_density=5e-4, expansion=4):
    """RoMaV2's balanced sampling weights ``(B, N)`` of the valid (score > 0) cells."""
    valid = (scores > 0).float()
    pool = valid.sum(1).clamp(max=expansion * num_samples)
    scale = pool / valid.sum(1).clamp_min(1)
    min_support = (min_density * pool)[:, None]
    points = []
    for i, grid in enumerate((dense_p2d0_n1, dense_p2d1_n0)):
        h, w = grid.shape[1:3]
        src = _grid_sources(h, w, grid.device).view(1, h, w, 2).expand_as(grid).flatten(1, 2)
        pair = [src, grid.flatten(1, 2)]
        points.append(torch.cat([pair[i], pair[1 - i]], -1))  # (x_i, x_j) normalized
    # upstream weights the density by ``valid``: park the gated-out points far outside [-1, 1]^4 so they add
    # nothing to the valid ones (their own density is masked below)
    points = torch.where(valid[..., None] > 0, torch.cat(points, 1), 10.0)
    density = ops.kde_density.get("reference")(points, std=std) * scale[:, None]
    balance = torch.where(density < min_support, 1e-7, 1 / (density + 1))
    return balance * valid  # valid cells only, no extra confidence weighting


class FlexGlue(ImageMatcher, nn.Module):
    """Sparse-to-dense matcher: dense search maps + 9 transformer layers + per-layer correspondence heads.

    ``match(a, b)`` describes both images densely and samples up to ``max_matches`` balanced correspondences
    from the dense warp. ``match_keypoints(a, kpts_a, b, kpts_b)`` matches externally detected keypoints
    (RDD / ALIKED / ...): each keypoint is mapped to a sub-pixel correspondence in the other image and the mutual
    nearest neighbours above ``threshold`` are kept. FlexGlue reads the dense maps, so the images are always
    required (``match_features`` on bare ``SparseFeatures`` is not supported).
    """

    match_defaults: Mapping[str, int] = {"max_matches": 5000}

    def __init__(self, config: FlexGlueConfig | None = None) -> None:
        super().__init__()
        self.config = cfg = config or FlexGlueConfig()
        d, h = cfg.descriptor_dim, cfg.num_heads
        self.input_proj = nn.Linear(cfg.input_dim, d)
        self.posenc_wr = nn.Linear(2, d // h // 2, bias=False)
        self.transformers = nn.ModuleList(TransformerLayer(d, h) for _ in range(cfg.num_layers))
        # heads[k] reads the features after k transformer layers, heads[0] the raw ones
        self.heads = nn.ModuleList(
            MatchHead(d, cfg.search_dim, cfg.query_chunk_size, cross=k > 0) for k in range(cfg.num_layers + 1)
        )
        self.descriptor = DINOv3Descriptor()
        # The correspondence heads read out sub-pixel positions through the entmax; they stay in FP32 under a
        # reduced compute dtype (like the assignment heads of LightGlue), which measurably tracks FP32 far better
        # than BF16 heads (the entmax logits are coordinate-sensitive).
        self.heads_dtype = torch.float32
        # Filled in by the builder (``flashim.build``).
        self.preset: str | None = None
        self.engine: str | None = None
        self.quant_report = None
        self.calibration_metadata: dict[str, str] | None = None
        self.compute_dtype = torch.float32

    @classmethod
    def from_pretrained(
        cls,
        config: FlexGlueConfig | None = None,
        *,
        checkpoint: str | Path | None = None,
        device: torch.device | str | None = None,
        state_dict: dict[str, Tensor] | None = None,
    ) -> FlexGlue:
        device = torch.device(device or default_device())
        config = config or FlexGlueConfig()
        model = cls(config)
        if state_dict is None:
            state_dict = weights.load_state_dict(checkpoint, map_location="cpu")
        model.load_state_dict(weights.truncate(state_dict, config.num_layers), strict=True)
        return model.to(device).eval()

    @property
    def device(self) -> torch.device:
        return self.input_proj.weight.device

    def set_compute_dtype(self, dtype: torch.dtype) -> None:
        """Run the descriptor and transformer in ``dtype``; the correspondence heads stay FP32 (the entmax readout
        is coordinate-sensitive), sampling / positions stay FP32."""
        self.descriptor.to(dtype)
        for module in (self.input_proj, self.transformers):
            module.to(dtype)
        self.heads.to(self.heads_dtype)
        self.compute_dtype = dtype

    # -- dense description -------------------------------------------------------------------

    def _encode(self, image: Tensor) -> Tensor:
        """Dense stride-8 search map ``[1, C, h, w]`` (FP32) of a ``[1, 3, H, W]`` image in ``[0, 1]``."""
        resized = F.interpolate(image, size=self.config.describe_resize, mode="bilinear", align_corners=False)
        return self.descriptor(resized.to(self.compute_dtype)).float()

    def _posenc(self, f: Tensor) -> Tensor:
        """Positional encodings of a search map's ``(h, w)`` cells: ``[2, 1, 1, h*w, D/h]`` (cos, sin)."""
        h, w = f.shape[-2:]
        projected = F.linear(cell_centers(h, w, scale=(max(h, w),) * 2, device=f.device), self.posenc_wr.weight)
        emb = torch.stack([projected.cos(), projected.sin()], 0).unsqueeze(-3).unsqueeze(-3)
        return emb.repeat_interleave(2, dim=-1)

    # -- pair decoding -----------------------------------------------------------------------

    def covisibility_score(self, x0: Tensor, x1: Tensor, layer: int) -> Tensor:
        """Covisibility score ``(B,)`` of the tokens after ``layer``; 1 for the raw features (head 0)."""
        covisibility = self.heads[layer].covisibility
        if covisibility is None:
            return x0.new_ones(x0.shape[0])
        tokens = torch.cat([x0, x1], 1).to(covisibility.weight.dtype)
        return torch.sigmoid(covisibility(tokens)).float().mean((1, 2))

    def decode_pair(self, f0: Tensor, f1: Tensor) -> tuple[int, Tensor, Tensor, Tensor]:
        """The last executed layer, its covisibility score and the fp32 cross features of both images.

        Runs the transformer layers with the early-stop gate when ``config.early_stop_decay`` is set (a host sync
        per checked layer, as upstream); otherwise every layer runs (deterministic, the upstream default).
        """
        x0, x1 = (self.input_proj(f.flatten(2).mT.to(self.compute_dtype)) for f in (f0, f1))
        enc0, enc1 = (self._posenc(f) for f in (f0, f1))
        cfg = self.config
        layer, score = cfg.num_layers, None
        for k, block in enumerate(self.transformers[: cfg.num_layers], 1):
            x0, x1 = block(x0, x1, enc0, enc1)
            if cfg.early_stop_decay is None or k == cfg.num_layers:
                continue
            # stop once the whole batch is a confident non-pair
            score = self.covisibility_score(x0, x1, k)
            gamma = (1 - cfg.early_stop_decay) ** (len(self.transformers) - k)
            if (score < cfg.min_covisibility * gamma).all():
                layer = k
                break
        # on a stop, ``score`` is the stopping layer's; otherwise recompute from the final tokens with the head the
        # matches will use (upstream's ``pair_context``), so ``is_covisible`` is never stale
        if layer == cfg.num_layers or score is None:
            score = self.covisibility_score(x0, x1, layer)
        cf0 = x0.mT.unflatten(-1, f0.shape[-2:]).float()
        cf1 = x1.mT.unflatten(-1, f1.shape[-2:]).float()
        return layer, score, cf0, cf1

    def run_head(self, layer: int, points_ni: Tensor, cf_i: Tensor, cf_j: Tensor) -> tuple[Tensor, Tensor]:
        """Head of ``layer`` on normalized points ``(B, N, 2)`` -> predicted points in j and confidence (FP32)."""
        cf_i, cf_j = cf_i.to(self.heads_dtype), cf_j.to(self.heads_dtype)
        p2di_nj, confidence = self.heads[layer](points_ni, cf_i, cf_j)
        return p2di_nj.float(), confidence.float()

    # -- matching ----------------------------------------------------------------------------

    def _encode_pair(self, images: list[Tensor]) -> list[Tensor]:
        """``_encode`` of both images in one descriptor batch (all are resized to ``describe_resize``)."""
        resized = torch.cat([
            F.interpolate(image, size=self.config.describe_resize, mode="bilinear", align_corners=False)
            for image in images
        ])
        return list(self.descriptor(resized.to(self.compute_dtype)).float().split(1))

    def _prepare(self, images: list[Tensor]) -> tuple[list[Tensor], list[Tensor], list[Tensor]]:
        features = self._encode_pair(images)
        sizes_wh = [
            device_constant([image.shape[-1], image.shape[-2]], image, torch.float32)[None] for image in images
        ]
        # the match gates operate in the descriptor's squashed (describe_resize) pixel frame, per axis
        resized_wh = [
            device_constant(list(self.config.describe_resize)[::-1], image, torch.float32)[None] for image in images
        ]
        return features, sizes_wh, resized_wh

    def _to_resized(self, points: Tensor, image_index: int, sizes_wh: list[Tensor], resized_wh: list[Tensor]) -> Tensor:
        """Points ``(K, 2)`` in original-image pixels -> the descriptor's resized pixel frame (per axis)."""
        return points * (resized_wh[image_index][0] / sizes_wh[image_index][0])

    @torch.inference_mode()
    def match_keypoints(self, img_a: ImageLike, kpts_a: Tensor, img_b: ImageLike, kpts_b: Tensor,
                        *, threshold: float | None = None) -> Matches:
        """Match two images at externally detected keypoints ``[K, 2]`` (pixel indices of each original image)."""
        images = [load_image(img_a, self.device), load_image(img_b, self.device)]
        return self._match_keypoints(images, [kpts_a, kpts_b], threshold)

    def _match_keypoints(self, images: list[Tensor], kpts: list[Tensor], threshold: float | None) -> Matches:
        features, sizes_wh, resized_wh = self._prepare(images)
        cfg = self.config
        layer, score, cf0, cf1 = self.decode_pair(*features)
        cf = [cf0, cf1]
        kpts_r, p2d_r, conf, inside = [], [], [], []
        for i in (0, 1):
            j = 1 - i
            kpts_ni = normalize_keypoints(kpts[i][None].float(), sizes_wh[i])
            p2di_nj, confidence = self.run_head(layer, kpts_ni, cf[i], cf[j])
            conf.append(confidence[0])
            kpts_r.append(self._to_resized(kpts[i], i, sizes_wh, resized_wh))
            p2d_r.append(self._to_resized(denormalize_keypoints(p2di_nj, sizes_wh[j])[0], j, sizes_wh, resized_wh))
            inside.append(
                ((kpts_r[i] > 0) & (kpts_r[i] < resized_wh[i][0])).all(-1)
                & ((p2d_r[i] > 0) & (p2d_r[i] < resized_wh[j][0])).all(-1)
            )
        is_covisible = score >= cfg.min_covisibility
        threshold = cfg.filter_threshold if threshold is None else threshold
        return self._sparse_matches(kpts, kpts_r, p2d_r, conf, inside, is_covisible, threshold, resized_wh)

    def _sparse_matches(self, kpts, kpts_r, p2d_r, conf, inside, is_covisible, threshold, resized_wh) -> Matches:
        """Mutual nearest neighbours of keypoints and predicted points, gated and confidence-filtered — all in the
        descriptor's resized pixel frame (upstream's ``sparse_matches`` / ``gate_points``)."""
        cfg = self.config
        valids = [inside[i] & is_covisible[0] for i in (0, 1)]
        # mutual NN on the bidirectional warp in one GEMM, centered on each resized image for fp32 precision
        c0, c1 = resized_wh[0][0] / 2, resized_wh[1][0] / 2
        x = torch.cat([p2d_r[0] - c1, kpts_r[0] - c0], -1)
        y = torch.cat([kpts_r[1] - c1, p2d_r[1] - c0], -1)
        m0, m1 = _mutual_nn(_sq_dist_gemm(x[None].float(), y[None].float()))

        def take(t, m):
            return t.gather(0, m.clamp(0)[:, None].expand(-1, 2))

        dist0 = 0.5 * (
            (p2d_r[0] - take(kpts_r[1], m0[0])).norm(dim=-1) + (kpts_r[0] - take(p2d_r[1], m0[0])).norm(dim=-1)
        )
        dist1 = 0.5 * (
            (take(p2d_r[0], m1[0]) - kpts_r[1]).norm(dim=-1) + (take(kpts_r[0], m1[0]) - p2d_r[1]).norm(dim=-1)
        )
        scores0 = (conf[0] * conf[1].gather(0, m0[0].clamp(0))).sqrt()
        scores1 = (conf[1] * conf[0].gather(0, m1[0].clamp(0))).sqrt()
        valid0 = (m0[0] > -1) & (scores0 >= threshold) & valids[0] & valids[1].gather(0, m0[0].clamp(0))
        valid1 = (m1[0] > -1) & (scores1 >= threshold) & valids[1] & valids[0].gather(0, m1[0].clamp(0))
        if cfg.max_distance_px > 0:  # dist in resized pixels of the (shared) descriptor frame
            valid0 &= dist0 < cfg.max_distance_px
            valid1 &= dist1 < cfg.max_distance_px
        ia = torch.nonzero(valid0)[:, 0]
        ib = m0[0][ia]
        if ia.numel() == 0:
            empty = kpts[0].new_zeros(0, 2)
            return Matches(empty, empty, kpts[0].new_zeros(0), ia, ib)
        return Matches(kpts[0][ia], kpts[1][ib], scores0[ia], ia, ib)

    def _match(self, img_a: ImageLike, img_b: ImageLike, *, resize: int | None, max_matches: int | None,
               threshold: float | None, generator: torch.Generator | None = None) -> Matches:
        """``match(a, b)``: up to ``max_matches`` balanced correspondences sampled from the gated dense warp."""
        if resize is not None:
            raise ValueError("FlexGlue.match takes no resize: its resolution is arch.describe_resize's")
        a, b = load_image(img_a, self.device), load_image(img_b, self.device)
        num = self.match_defaults["max_matches"] if max_matches is None else max_matches
        threshold = self.config.filter_threshold if threshold is None else threshold
        return self._match_sampled(a, b, num, threshold, generator)

    def _match_sampled(self, a: Tensor, b: Tensor, num: int, threshold: float,
                       generator: torch.Generator | None) -> Matches:
        cfg = self.config
        images = [a, b]
        features, sizes_wh, resized_wh = self._prepare(images)
        layer, score, cf0, cf1 = self.decode_pair(*features)
        cf = [cf0, cf1]
        hw = [tuple(f.shape[-2:]) for f in features]
        centers = [_grid_sources(h, w, a.device).view(1, h * w, 2) for h, w in hw]
        p2d_n, conf = [], []
        for i in (0, 1):
            j = 1 - i
            p2di_nj, confidence = self.run_head(layer, centers[i], cf[i], cf[j])
            p2d_n.append(p2di_nj.view(1, hw[i][0], hw[i][1], 2))
            conf.append(confidence.view(1, hw[i][0], hw[i][1]))
        # gate_dense (upstream): covisibility, confidence filter, border and cycle checks in the resized frame
        is_covisible = (score >= cfg.min_covisibility).view(1, 1, 1)
        src_n = [centers[i].view(1, *hw[i], 2) for i in (0, 1)]
        src_r = [denormalize_keypoints(src_n[i], resized_wh[i]) for i in (0, 1)]
        p2d_r = [denormalize_keypoints(p2d_n[i], resized_wh[1 - i]) for i in (0, 1)]  # p2d{i} in resized j
        valids = []
        for i in (0, 1):
            j = 1 - i
            valid = is_covisible.expand_as(conf[i]) & (conf[i] >= threshold)
            valid &= ((src_r[i] > 0) & (src_r[i] < resized_wh[i][0].view(1, 1, 2))).all(-1)
            valid &= ((p2d_r[i] > 0) & (p2d_r[i] < resized_wh[j][0].view(1, 1, 2))).all(-1)
            if cfg.max_distance_px > 0:  # cycle error: back through the dense j -> i grid, resized pixels of i
                back_n = sample_features(p2d_n[j].permute(0, 3, 1, 2), p2d_n[i].flatten(1, 2))
                back_r = denormalize_keypoints(back_n.view_as(p2d_n[i]), resized_wh[i])
                valid &= (back_r - src_r[i]).norm(dim=-1) < cfg.max_distance_px
            valids.append(valid)
        for i in (0, 1):  # gate after both cycle checks read the ungated grids
            conf[i] = torch.where(valids[i], conf[i], conf[i].new_zeros(()))
        scores = torch.cat([conf[0], conf[1]], 1).flatten(1)  # (1, 2*h*w), 0 where gated out
        weights = _kde_balance_weights(p2d_n[0], p2d_n[1], scores, num)
        idx = torch.multinomial(weights[0].clamp_min(1e-12), min(num, weights.shape[1]), replacement=False,
                                generator=generator)
        # image-i pixel position of each candidate: the i side of its own grid or the j->i warp (original pixels)
        src = [denormalize_keypoints(centers[i][0], sizes_wh[i]) for i in (0, 1)]
        warp10 = denormalize_keypoints(p2d_n[1].view(1, -1, 2), sizes_wh[0])[0]  # image-1 cells -> image 0
        warp01 = denormalize_keypoints(p2d_n[0].view(1, -1, 2), sizes_wh[1])[0]  # image-0 cells -> image 1
        kpts0 = torch.cat([src[0], warp10])
        kpts1 = torch.cat([warp01, src[1]])
        sample_scores = scores[0].gather(0, idx)
        keep = sample_scores > 0
        return Matches(kpts0[idx][keep], kpts1[idx][keep], sample_scores[keep])
