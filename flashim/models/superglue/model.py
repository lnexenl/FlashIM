"""SuperGlue sparse matcher (native FlashIM port: ``nn.Linear`` + ``ops.sdpa`` + ``ops.log_optimal_transport``)."""

from __future__ import annotations

import logging
from collections import OrderedDict
from collections.abc import Sequence

import torch
from torch import Tensor, nn

from flashim import ops
from flashim.log import warn_once
from flashim.models.superglue import weights
from flashim.models.superglue.config import SuperGlueConfig
from flashim.platforms import default_device
from flashim.types import Matches, SparseFeatures
from flashim.utils import device_constant

logger = logging.getLogger(__name__)


def normalize_keypoints(keypoints: Tensor, image_size: tuple[int, int]) -> Tensor:
    """Center keypoints ``[B, K, 2]`` ``(x, y)`` on ``(W/2, H/2)`` and scale by ``0.7 * max(W, H)`` (FP32)."""
    height, width = image_size
    size = device_constant((float(width), float(height)), keypoints, torch.float32)
    center = size * 0.5
    scale = max(float(width), float(height)) * 0.7
    return (keypoints.float() - center[None, None, :]) / scale


def make_mlp(channels: Sequence[int], *, do_bn: bool = True) -> nn.Sequential:
    """Pointwise MLP matching the sequential indices of the SuperGlue checkpoints."""
    layers: list[nn.Module] = []
    n = len(channels)
    for i in range(1, n):
        layers.append(nn.Linear(channels[i - 1], channels[i], bias=True))
        if i < n - 1:
            if do_bn:
                layers.append(nn.BatchNorm1d(channels[i]))
            layers.append(nn.ReLU())
    return nn.Sequential(*layers)


def _apply_mlp(mlp: nn.Sequential, x: Tensor) -> Tensor:
    """Run ``mlp`` on ``[B, N, C]`` or ``[N, C]``; an unfolded ``BatchNorm1d`` runs channel-first, a folded
    ``Identity`` does not."""
    for layer in mlp:
        if isinstance(layer, nn.BatchNorm1d):
            if x.ndim == 2:
                x = layer(x.unsqueeze(0).transpose(1, 2)).transpose(1, 2).squeeze(0)
            else:
                x = layer(x.transpose(1, 2)).transpose(1, 2)
        else:
            x = layer(x)
    return x


class KeypointEncoder(nn.Module):
    """Joint MLP encoding of normalized keypoint coordinates and detector scores."""

    def __init__(self, feature_dim: int, layers: Sequence[int]) -> None:
        super().__init__()
        self.encoder = make_mlp([3, *layers, feature_dim])
        nn.init.constant_(self.encoder[-1].bias, 0.0)

    def forward(self, kpts: Tensor, scores: Tensor) -> Tensor:
        return _apply_mlp(self.encoder, torch.cat((kpts, scores.unsqueeze(-1)), dim=-1))


class MultiHeadedAttention(nn.Module):
    """Multi-head attention over ``[B, N, C]`` (or concatenated ``[M + N, C]`` stream) via :data:`flashim.ops.sdpa`."""

    def __init__(self, num_heads: int, d_model: int) -> None:
        super().__init__()
        if d_model % num_heads != 0:
            raise ValueError(f"d_model ({d_model}) must be divisible by num_heads ({num_heads})")
        self.dim = d_model // num_heads
        self.num_heads = num_heads
        self.merge = nn.Linear(d_model, d_model)
        self.proj = nn.ModuleList(nn.Linear(d_model, d_model) for _ in range(3))
        self.qkv: nn.Module | None = None
        self.sdpa_impl: str | None = "reference"

    def pack_qkv(self) -> None:
        """Pack ``proj[0..2]`` into a single ``qkv = nn.Linear(C, 3 * C)`` (exact, 3x fewer GEMMs)."""
        if self.qkv is not None:
            return
        d_model = self.num_heads * self.dim
        qkv = nn.Linear(d_model, 3 * d_model, bias=True, device=self.merge.weight.device, dtype=self.merge.weight.dtype)
        with torch.no_grad():
            qkv.weight.copy_(torch.cat([p.weight for p in self.proj], dim=0))
            qkv.bias.copy_(torch.cat([p.bias for p in self.proj], dim=0))
        self.qkv = qkv
        self.proj = nn.ModuleList()

    def _heads(self, x: Tensor) -> Tensor:
        return x.view(x.shape[0], x.shape[1], self.num_heads, self.dim).transpose(1, 2)

    def forward(self, query: Tensor, key: Tensor, value: Tensor) -> Tensor:
        if self.qkv is not None:
            c = self.num_heads * self.dim
            w, b = self.qkv.weight, self.qkv.bias
            q = self._heads(nn.functional.linear(query, w[:c], b[:c]))
            k = self._heads(nn.functional.linear(key, w[c : 2 * c], b[c : 2 * c]))
            v = self._heads(nn.functional.linear(value, w[2 * c :], b[2 * c :]))
        else:
            q = self._heads(self.proj[0](query))
            k = self._heads(self.proj[1](key))
            v = self._heads(self.proj[2](value))
        out = ops.sdpa.get(self.sdpa_impl)(q, k, v)
        return self.merge(out.transpose(1, 2).reshape(query.shape))

    def forward_stream(self, x: Tensor, m: int, n: int, *, cross: bool) -> Tensor:
        """Self or cross attention on a concatenated ``[M + N, C]`` stream (one linear call per projection)."""
        c = self.num_heads * self.dim
        sdpa = ops.sdpa.get(self.sdpa_impl)
        if self.qkv is not None:
            qkv = self.qkv(x)
            if m == n:
                q, k, v = qkv.view(2, m, 3, self.num_heads, self.dim).permute(2, 0, 3, 1, 4)
                if cross:
                    k, v = k.flip(0), v.flip(0)
                o = sdpa(q, k, v).transpose(1, 2).reshape(m + n, c)
            else:
                q0, k0, v0 = qkv[:m].view(1, m, 3, self.num_heads, self.dim).permute(2, 0, 3, 1, 4)
                q1, k1, v1 = qkv[m:].view(1, n, 3, self.num_heads, self.dim).permute(2, 0, 3, 1, 4)
                if cross:
                    o0, o1 = sdpa(q0, k1, v1), sdpa(q1, k0, v0)
                else:
                    o0, o1 = sdpa(q0, k0, v0), sdpa(q1, k1, v1)
                o = torch.cat((o0.transpose(1, 2).reshape(m, c), o1.transpose(1, 2).reshape(n, c)), dim=0)
        else:
            q_all, k_all, v_all = self.proj[0](x), self.proj[1](x), self.proj[2](x)
            if m == n:
                q = q_all.view(2, m, self.num_heads, self.dim).transpose(1, 2)
                k = k_all.view(2, m, self.num_heads, self.dim).transpose(1, 2)
                v = v_all.view(2, m, self.num_heads, self.dim).transpose(1, 2)
                if cross:
                    k, v = k.flip(0), v.flip(0)
                o = sdpa(q, k, v).transpose(1, 2).reshape(m + n, c)
            else:
                q0 = q_all[:m].view(1, m, self.num_heads, self.dim).transpose(1, 2)
                k0 = k_all[:m].view(1, m, self.num_heads, self.dim).transpose(1, 2)
                v0 = v_all[:m].view(1, m, self.num_heads, self.dim).transpose(1, 2)
                q1 = q_all[m:].view(1, n, self.num_heads, self.dim).transpose(1, 2)
                k1 = k_all[m:].view(1, n, self.num_heads, self.dim).transpose(1, 2)
                v1 = v_all[m:].view(1, n, self.num_heads, self.dim).transpose(1, 2)
                if cross:
                    o0, o1 = sdpa(q0, k1, v1), sdpa(q1, k0, v0)
                else:
                    o0, o1 = sdpa(q0, k0, v0), sdpa(q1, k1, v1)
                o = torch.cat((o0.transpose(1, 2).reshape(m, c), o1.transpose(1, 2).reshape(n, c)), dim=0)
        return self.merge(o)

    def forward_pair(self, x0: Tensor, x1: Tensor, *, cross: bool) -> tuple[Tensor, Tensor]:
        """Self or cross attention on a pair ``([1, M, C], [1, N, C])``."""
        m, n = x0.shape[1], x1.shape[1]
        x = torch.cat((x0[0], x1[0]), dim=0)
        out = self.forward_stream(x, m, n, cross=cross)
        return out[:m][None], out[m:][None]


class AttentionalPropagation(nn.Module):
    """Attention message + residual update MLP ``[x, message] -> delta``."""

    def __init__(self, feature_dim: int, num_heads: int) -> None:
        super().__init__()
        self.attn = MultiHeadedAttention(num_heads, feature_dim)
        self.mlp = make_mlp([feature_dim * 2, feature_dim * 2, feature_dim])
        nn.init.constant_(self.mlp[-1].bias, 0.0)

    def forward(self, x: Tensor, source: Tensor) -> Tensor:
        message = self.attn(x, source, source)
        return _apply_mlp(self.mlp, torch.cat((x, message), dim=-1))

    def forward_stream(self, x: Tensor, m: int, n: int, *, cross: bool) -> Tensor:
        msg = self.attn.forward_stream(x, m, n, cross=cross)
        return x + _apply_mlp(self.mlp, torch.cat((x, msg), dim=-1))

    def forward_pair(self, x0: Tensor, x1: Tensor, *, cross: bool) -> tuple[Tensor, Tensor]:
        m, n = x0.shape[1], x1.shape[1]
        out = self.forward_stream(torch.cat((x0[0], x1[0]), dim=0), m, n, cross=cross)
        return out[:m][None], out[m:][None]


class AttentionalGNN(nn.Module):
    """Alternating self- and cross-attention graph neural network."""

    def __init__(self, feature_dim: int, layer_names: Sequence[str], num_heads: int = 4) -> None:
        super().__init__()
        for name in layer_names:
            if name not in ("self", "cross"):
                raise ValueError(f"Unknown GNN layer type {name!r}; expected 'self' or 'cross'")
        self.layers = nn.ModuleList(AttentionalPropagation(feature_dim, num_heads) for _ in layer_names)
        self.names = list(layer_names)

    def forward_stream(self, x: Tensor, m: int, n: int) -> Tensor:
        for layer, name in zip(self.layers, self.names):
            x = layer.forward_stream(x, m, n, cross=(name == "cross"))
        return x

    def forward(self, desc0: Tensor, desc1: Tensor) -> tuple[Tensor, Tensor]:
        m, n = desc0.shape[1], desc1.shape[1]
        out = self.forward_stream(torch.cat((desc0[0], desc1[0]), dim=0), m, n)
        return out[:m][None], out[m:][None]


def filter_matches(scores: Tensor, threshold: float) -> tuple[Tensor, Tensor, Tensor]:
    """Mutual best matches of a log-assignment matrix ``[1, M, N]`` (dustbin stripped) with probability > ``threshold``.

    Returns ``(indices_a, indices_b, confidence)`` as 1-D tensors.
    """
    best0, best1 = scores.max(dim=2), scores.max(dim=1)
    m0, m1 = best0.indices, best1.indices
    mutual = torch.arange(m0.shape[1], device=m0.device)[None] == m1.gather(1, m0)
    prob = best0.values.exp()
    valid = (mutual & (prob > threshold))[0]
    index_a = torch.nonzero(valid)[:, 0]
    return index_a, m0[0][index_a], prob[0][index_a]


class SuperGlue(nn.Module):
    """SuperGlue attentional GNN + log-space Sinkhorn optimal transport matcher.

    Module and parameter paths follow ``superglue_{outdoor,indoor}.pth`` after :func:`weights.convert_state_dict`,
    so checkpoints load with ``strict=True``. ``match_features`` takes two :class:`SparseFeatures` and returns
    mutual matches above ``match_threshold``.
    """

    def __init__(self, config: SuperGlueConfig | None = None) -> None:
        super().__init__()
        self.config = cfg = config or SuperGlueConfig()
        self.kenc = KeypointEncoder(cfg.descriptor_dim, cfg.keypoint_encoder)
        self.gnn = AttentionalGNN(cfg.descriptor_dim, cfg.gnn_layers, cfg.num_heads)
        self.final_proj = nn.Linear(cfg.descriptor_dim, cfg.descriptor_dim, bias=True)
        self.bin_score = nn.Parameter(torch.tensor(1.0))

        self.sinkhorn_iterations = cfg.sinkhorn_iterations
        self.match_threshold = cfg.match_threshold
        self.compute_dtype = torch.float32
        # Optimization flags (set by passes in ``spec.py``).
        self.low_precision_similarity = False
        self.int8_glue = False
        self.graphs = 0  # CUDA graph LRU cache size per (M, N) keypoint shape (0 = eager)
        # Filled by the builder (``flashim.build``).
        self.preset: str | None = None
        self.engine: str | None = None
        self.quant_report = None
        self.calibration_metadata: dict[str, str] | None = None

    @classmethod
    def from_pretrained(
        cls,
        config: SuperGlueConfig | None = None,
        *,
        variant: str = weights.DEFAULT_VARIANT,
        checkpoint: str | None = None,
        device: torch.device | str | None = None,
        state_dict: dict[str, Tensor] | None = None,
    ) -> SuperGlue:
        device = torch.device(device or default_device())
        config = config or SuperGlueConfig()
        with torch.device("meta"):
            model = cls(config)
        state = state_dict or weights.load_state_dict(variant, num_heads=config.num_heads, path=checkpoint)
        model.load_state_dict(state, strict=True, assign=True)
        return model.to(device).eval()

    @property
    def device(self) -> torch.device:
        return self.bin_score.device

    def set_compute_dtype(self, dtype: torch.dtype) -> None:
        """Run the GNN layers and ``final_proj`` in ``dtype``; ``kenc``, ``bin_score`` and Sinkhorn stay FP32."""
        self.gnn.to(dtype)
        self.final_proj.to(dtype)
        self.compute_dtype = dtype

    def fold_batchnorm(self) -> list[str]:
        """Fold every ``(Linear, BatchNorm1d)`` pair in ``kenc`` and ``gnn`` (exact in eval mode)."""
        if self.training:
            raise RuntimeError("fold_batchnorm requires model.eval()")
        folded: list[str] = []
        sequences: list[tuple[str, nn.Sequential]] = [("kenc.encoder", self.kenc.encoder)]
        sequences.extend((f"gnn.layers.{i}.mlp", layer.mlp) for i, layer in enumerate(self.gnn.layers))
        for prefix, seq in sequences:
            for i in range(len(seq) - 1):
                if isinstance(seq[i], nn.Linear) and isinstance(seq[i + 1], nn.BatchNorm1d):
                    seq[i] = nn.utils.fusion.fuse_linear_bn_eval(seq[i], seq[i + 1])
                    seq[i + 1] = nn.Identity()
                    folded.append(f"{prefix}.{i}")
        return folded

    def pack_qkv(self) -> int:
        """Pack ``proj[0..2]`` into one ``qkv`` linear per GNN layer (exact, 3x fewer attention projection GEMMs)."""
        for layer in self.gnn.layers:
            layer.attn.pack_qkv()
        return len(self.gnn.layers)

    def _gnn_and_proj(
        self, kpts0: Tensor, scores0: Tensor, desc0: Tensor, kpts1: Tensor, scores1: Tensor, desc1: Tensor
    ) -> tuple[Tensor, Tensor]:
        """Run ``kenc`` + GNN + ``final_proj`` on already-normalized keypoints; returns ``(mdesc0 [M, C], mdesc1 [N, C])``."""
        m, n = desc0.shape[1], desc1.shape[1]
        kpts = torch.cat((kpts0, kpts1), dim=1)
        sc = torch.cat((scores0.float(), scores1.float()), dim=1)
        x = torch.cat((desc0.float(), desc1.float()), dim=1) + self.kenc(kpts, sc)
        rot_in = getattr(self, "_rot_in", None)
        if rot_in is not None:
            x = x @ rot_in
        stream = x[0].to(self.compute_dtype, memory_format=torch.contiguous_format)
        if self.int8_glue:
            from flashim.models.superglue import int8_glue

            stream = int8_glue.gnn_stream(self, stream, m, n)
        else:
            stream = self.gnn.forward_stream(stream, m, n)
        mdesc = self.final_proj(stream)
        return mdesc[:m], mdesc[m:]

    def encode_features(self, a: SparseFeatures, b: SparseFeatures) -> tuple[Tensor, Tensor]:
        """Projected descriptors ``(mdesc0, mdesc1)`` in FP32 ``[1, M, C]`` and ``[1, N, C]``."""
        kpts0 = normalize_keypoints(a.keypoints, a.image_size)
        kpts1 = normalize_keypoints(b.keypoints, b.image_size)
        m0, m1 = self._gnn_and_proj(kpts0, a.scores, a.descriptors, kpts1, b.scores, b.descriptors)
        return m0.float()[None], m1.float()[None]

    def _scores_from_norm(
        self, kpts0: Tensor, scores0: Tensor, desc0: Tensor, kpts1: Tensor, scores1: Tensor, desc1: Tensor, iters: int
    ) -> Tensor:
        m0, m1 = self._gnn_and_proj(kpts0, scores0, desc0, kpts1, scores1, desc1)
        scale = self.config.descriptor_dim**-0.5
        if self.low_precision_similarity and m0.dtype in (torch.bfloat16, torch.float16):
            sim = ops.matmul_nt_fp32(m0, m1, scale)[None]
        else:
            sim = torch.matmul(m0.float(), m1.float().t())[None] * scale
        return ops.log_optimal_transport(sim, self.bin_score.float(), iters)

    def scores(
        self, a: SparseFeatures, b: SparseFeatures, *, sinkhorn_iterations: int | None = None
    ) -> Tensor:
        """Full ``[1, M + 1, N + 1]`` log-assignment matrix (with dustbin) in FP32."""
        kpts0 = normalize_keypoints(a.keypoints, a.image_size)
        kpts1 = normalize_keypoints(b.keypoints, b.image_size)
        iters = self.sinkhorn_iterations if sinkhorn_iterations is None else sinkhorn_iterations
        return self._scores_from_norm(kpts0, a.scores, a.descriptors, kpts1, b.scores, b.descriptors, iters)

    def _best_matches_core(
        self, kpts0: Tensor, scores0: Tensor, desc0: Tensor, kpts1: Tensor, scores1: Tensor, desc1: Tensor
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Fixed-shape matching core (capturable as a CUDA graph): returns ``(prob [M], index [M], mutual [M])``."""
        log_p = self._scores_from_norm(
            kpts0, scores0, desc0, kpts1, scores1, desc1, self.sinkhorn_iterations
        )[:, :-1, :-1]
        best0, best1 = log_p.max(dim=2), log_p.max(dim=1)
        m0, m1 = best0.indices, best1.indices
        mutual = (torch.arange(m0.shape[1], device=m0.device)[None] == m1.gather(1, m0))[0]
        return best0.values.exp()[0], m0[0], mutual

    def best_matches(self, a: SparseFeatures, b: SparseFeatures) -> tuple[Tensor, Tensor, Tensor]:
        """Per keypoint of image A: ``(prob, index_in_b, is_mutual)`` (CUDA-graph replayed when ``graphs > 0``)."""
        kpts0 = normalize_keypoints(a.keypoints, a.image_size)
        kpts1 = normalize_keypoints(b.keypoints, b.image_size)
        args = (kpts0, a.scores, a.descriptors, kpts1, b.scores, b.descriptors)
        if not self.graphs or not kpts0.is_cuda or torch.is_grad_enabled():
            return self._best_matches_core(*args)
        from flashim.engines.cuda_graph import CudaGraphRunner

        cache = self.__dict__.setdefault("_graph_runners", OrderedDict())
        key = (*((t.shape, t.dtype, t.device) for t in args), self.sinkhorn_iterations)
        runner = cache.get(key)
        if runner is not None:
            cache.move_to_end(key)
            return runner(*args)
        runner = CudaGraphRunner(self._best_matches_core)
        try:
            outputs = runner(*args)
        except RuntimeError as exc:
            warn_once(
                logger,
                f"superglue-graph-{id(self)}",
                "SuperGlue: CUDA graph capture failed, running eagerly from now on (%s)",
                exc,
            )
            self.graphs = 0
            torch.cuda.synchronize()
            return self._best_matches_core(*args)
        cache[key] = runner
        while len(cache) > self.graphs:
            cache.popitem(last=False)
        return outputs

    @torch.inference_mode()
    def match_features(
        self, a: SparseFeatures, b: SparseFeatures, threshold: float | None = None
    ) -> Matches:
        if a.keypoints.shape[0] != 1 or b.keypoints.shape[0] != 1:
            raise ValueError("SuperGlue matches one image pair at a time (batch size 1)")
        if a.keypoints.shape[1] == 0 or b.keypoints.shape[1] == 0:
            empty = a.keypoints.new_zeros(0, 2)
            return Matches(empty, empty, a.keypoints.new_zeros(0))
        prob, index, mutual = self.best_matches(a, b)
        thresh = self.match_threshold if threshold is None else threshold
        ia = torch.nonzero(mutual & (prob > thresh))[:, 0]
        ib = index[ia]
        return Matches(a.keypoints[0][ia], b.keypoints[0][ib], prob[ia], ia, ib)
