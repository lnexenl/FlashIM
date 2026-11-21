"""SuperGlue sparse matcher (native FlashIM port: ``nn.Linear`` + ``ops.sdpa`` + ``ops.log_optimal_transport``)."""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import Tensor, nn

from flashim import ops
from flashim.models.superglue import weights
from flashim.models.superglue.config import SuperGlueConfig
from flashim.platforms import default_device
from flashim.types import Matches, SparseFeatures
from flashim.utils import device_constant


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
    """Run ``mlp`` on ``[B, N, C]``; an unfolded ``BatchNorm1d`` runs channel-first, a folded ``Identity`` does not."""
    for layer in mlp:
        if isinstance(layer, nn.BatchNorm1d):
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
    """Multi-head attention over ``[B, N, C]`` via :data:`flashim.ops.sdpa`."""

    def __init__(self, num_heads: int, d_model: int) -> None:
        super().__init__()
        if d_model % num_heads != 0:
            raise ValueError(f"d_model ({d_model}) must be divisible by num_heads ({num_heads})")
        self.dim = d_model // num_heads
        self.num_heads = num_heads
        self.merge = nn.Linear(d_model, d_model)
        self.proj = nn.ModuleList(nn.Linear(d_model, d_model) for _ in range(3))
        self.sdpa_impl: str | None = "reference"

    def _heads(self, x: Tensor) -> Tensor:
        return x.view(x.shape[0], x.shape[1], self.num_heads, self.dim).transpose(1, 2)

    def forward(self, query: Tensor, key: Tensor, value: Tensor) -> Tensor:
        q = self._heads(self.proj[0](query))
        k = self._heads(self.proj[1](key))
        v = self._heads(self.proj[2](value))
        out = ops.sdpa.get(self.sdpa_impl)(q, k, v)
        return self.merge(out.transpose(1, 2).reshape(query.shape))

    def forward_pair(self, x0: Tensor, x1: Tensor, *, cross: bool) -> tuple[Tensor, Tensor]:
        """Self or cross attention on a pair; batches both images into ``[2, N, C]`` when their shapes match."""
        if x0.shape == x1.shape:
            x = torch.cat((x0, x1), dim=0)
            q = self._heads(self.proj[0](x))
            k = self._heads(self.proj[1](x))
            v = self._heads(self.proj[2](x))
            if cross:
                k, v = k.flip(0), v.flip(0)
            out = ops.sdpa.get(self.sdpa_impl)(q, k, v)
            merged = self.merge(out.transpose(1, 2).reshape(x.shape))
            m0, m1 = merged.chunk(2, dim=0)
            return m0, m1
        if cross:
            return self.forward(x0, x1, x1), self.forward(x1, x0, x0)
        return self.forward(x0, x0, x0), self.forward(x1, x1, x1)


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

    def forward_pair(self, x0: Tensor, x1: Tensor, *, cross: bool) -> tuple[Tensor, Tensor]:
        m0, m1 = self.attn.forward_pair(x0, x1, cross=cross)
        if x0.shape == x1.shape:
            x = torch.cat((x0, x1), dim=0)
            m = torch.cat((m0, m1), dim=0)
            updated = x + _apply_mlp(self.mlp, torch.cat((x, m), dim=-1))
            y0, y1 = updated.chunk(2, dim=0)
            return y0, y1
        return (
            x0 + _apply_mlp(self.mlp, torch.cat((x0, m0), dim=-1)),
            x1 + _apply_mlp(self.mlp, torch.cat((x1, m1), dim=-1)),
        )


class AttentionalGNN(nn.Module):
    """Alternating self- and cross-attention graph neural network."""

    def __init__(self, feature_dim: int, layer_names: Sequence[str], num_heads: int = 4) -> None:
        super().__init__()
        for name in layer_names:
            if name not in ("self", "cross"):
                raise ValueError(f"Unknown GNN layer type {name!r}; expected 'self' or 'cross'")
        self.layers = nn.ModuleList(AttentionalPropagation(feature_dim, num_heads) for _ in layer_names)
        self.names = list(layer_names)

    def forward(self, desc0: Tensor, desc1: Tensor) -> tuple[Tensor, Tensor]:
        for layer, name in zip(self.layers, self.names):
            desc0, desc1 = layer.forward_pair(desc0, desc1, cross=(name == "cross"))
        return desc0, desc1


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

    def encode_features(self, a: SparseFeatures, b: SparseFeatures) -> tuple[Tensor, Tensor]:
        """Projected descriptors ``(mdesc0, mdesc1)`` in FP32 ``[1, M, C]`` and ``[1, N, C]``."""
        kpts0 = normalize_keypoints(a.keypoints, a.image_size)
        kpts1 = normalize_keypoints(b.keypoints, b.image_size)
        desc0 = a.descriptors.float() + self.kenc(kpts0, a.scores.float())
        desc1 = b.descriptors.float() + self.kenc(kpts1, b.scores.float())
        desc0 = desc0.to(self.compute_dtype, memory_format=torch.contiguous_format)
        desc1 = desc1.to(self.compute_dtype, memory_format=torch.contiguous_format)
        desc0, desc1 = self.gnn(desc0, desc1)
        if desc0.shape == desc1.shape:
            mdesc0, mdesc1 = self.final_proj(torch.cat((desc0, desc1), dim=0)).chunk(2, dim=0)
        else:
            mdesc0, mdesc1 = self.final_proj(desc0), self.final_proj(desc1)
        return mdesc0.float(), mdesc1.float()

    def scores(
        self, a: SparseFeatures, b: SparseFeatures, *, sinkhorn_iterations: int | None = None
    ) -> Tensor:
        """Full ``[1, M + 1, N + 1]`` log-assignment matrix (with dustbin) in FP32."""
        mdesc0, mdesc1 = self.encode_features(a, b)
        sim = torch.matmul(mdesc0, mdesc1.transpose(-1, -2)) / (self.config.descriptor_dim**0.5)
        iters = self.sinkhorn_iterations if sinkhorn_iterations is None else sinkhorn_iterations
        return ops.log_optimal_transport(sim, self.bin_score.float(), iters)

    @torch.inference_mode()
    def match_features(
        self, a: SparseFeatures, b: SparseFeatures, threshold: float | None = None
    ) -> Matches:
        if a.keypoints.shape[0] != 1 or b.keypoints.shape[0] != 1:
            raise ValueError("SuperGlue matches one image pair at a time (batch size 1)")
        if a.keypoints.shape[1] == 0 or b.keypoints.shape[1] == 0:
            empty = a.keypoints.new_zeros(0, 2)
            return Matches(empty, empty, a.keypoints.new_zeros(0))
        scores = self.scores(a, b)
        thresh = self.match_threshold if threshold is None else threshold
        ia, ib, conf = filter_matches(scores[:, :-1, :-1], thresh)
        return Matches(a.keypoints[0][ia], b.keypoints[0][ib], conf, ia, ib)
