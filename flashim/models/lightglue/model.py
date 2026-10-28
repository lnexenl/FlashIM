"""LightGlue sparse matcher (native FlashIM port; no early stopping or point pruning)."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from flashim import ops
from flashim.models.lightglue import weights
from flashim.models.lightglue.config import LightGlueConfig
from flashim.platforms import default_device
from flashim.quant.methods.linear import StaticMXFP8Linear
from flashim.types import Matches, SparseFeatures
from flashim.utils import device_constant


def normalize_keypoints(keypoints: Tensor, image_size: Tensor) -> Tensor:
    """Center keypoints ``[B, K, 2]`` on ``image_size / 2`` and scale by half its largest side."""
    size = image_size.to(keypoints)
    return (keypoints - size[..., None, :] / 2) / (size.amax(-1) / 2)[..., None, None]


def rotate_half(x: Tensor) -> Tensor:
    return torch.stack((-x[..., 1::2], x[..., ::2]), dim=-1).flatten(-2)


def apply_rotary(t: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
    """Interleaved rotary embedding in FP32; ``t`` is ``[B, H, N, D]``, the tables ``[B, 1, N, D]``."""
    t32 = t.float()
    return (t32 * cos + rotate_half(t32) * sin).to(t.dtype)


class FourierPositionalEncoding(nn.Module):
    """Learnable Fourier features of the normalized keypoint position: ``[2 (cos, sin), B, 1, K, D]``."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.Wr = nn.Linear(2, dim // 2, bias=False)

    def forward(self, keypoints: Tensor) -> Tensor:
        projected = F.linear(keypoints.float(), self.Wr.weight.float())
        emb = torch.stack((projected.cos(), projected.sin()), 0).unsqueeze(-3)
        return emb.repeat_interleave(2, dim=-1)


def apply_batched(fn, xs: list[Tensor]) -> list[Tensor]:
    """``fn`` on every tensor of ``xs``; one call on the concatenation when the shapes agree."""
    if len({x.shape for x in xs}) == 1:
        return list(fn(torch.cat(xs)).chunk(len(xs)))
    return [fn(x) for x in xs]


def ffn_forward(ffn: nn.Sequential, x: Tensor) -> Tensor:
    """``ffn(x)`` (``Linear -> LayerNorm -> GELU -> Linear``); with MXFP8 linears, LayerNorm, GELU and the last linear's
    input quantization run as one kernel with the first linear's bias in it, instead of four passes. INT8 keeps the
    unfused path: there the fused statistics' one-ulp BF16 differences moved the matches of the RDD pair in
    tests/parity/test_lightglue_parity.py below its 90 % threshold (106 / 118 vs 107+), in either variance form tried."""
    first, norm, act, last = ffn
    fusable = isinstance(first, StaticMXFP8Linear) and isinstance(last, StaticMXFP8Linear) and first.bias is not None
    if not (fusable and isinstance(norm, nn.LayerNorm) and isinstance(act, nn.GELU)):
        return ffn(x)
    hidden = first.matmul(first.quantize(x))
    packed = last.quantize_layernorm(
        hidden, None, norm, input_bias=first.bias[: first.out_features], activation=act.approximate
    )
    return last.forward_prequantized(packed).to(x.dtype)


def make_ffn(dim: int) -> nn.Sequential:
    return nn.Sequential(nn.Linear(2 * dim, 2 * dim), nn.LayerNorm(2 * dim), nn.GELU(), nn.Linear(2 * dim, dim))


class SelfBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.Wqkv = nn.Linear(dim, 3 * dim)
        self.out_proj = nn.Linear(dim, dim)
        self.ffn = make_ffn(dim)
        self.sdpa_impl: str | None = "reference"

    def attend(self, x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
        batch, tokens, _ = x.shape
        q, k, v = self.Wqkv(x).view(batch, tokens, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        q, k = ops.rope_qk_interleaved(q, k, cos, sin)  # = apply_rotary on each: one kernel for both instead of ~10
        out = ops.sdpa.get(self.sdpa_impl)(q, k, v)
        return self.out_proj(out.transpose(1, 2).reshape(batch, tokens, -1))

    def forward(self, xs: list[Tensor], encodings: list[Tensor]) -> list[Tensor]:
        """Per-image tokens ``[1, K_i, C]`` with their rotary tables ``[2, 1, 1, K_i, D]``."""
        if len({x.shape for x in xs}) == 1:
            x = torch.cat(xs)
            enc = torch.cat(encodings, dim=1)
            x = x + ffn_forward(self.ffn, torch.cat((x, self.attend(x, enc[0], enc[1])), -1))
            return list(x.chunk(len(xs)))
        return [
            x + ffn_forward(self.ffn, torch.cat((x, self.attend(x, e[0], e[1])), -1)) for x, e in zip(xs, encodings)
        ]


class CrossBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.to_qk = nn.Linear(dim, dim)
        self.to_v = nn.Linear(dim, dim)
        self.to_out = nn.Linear(dim, dim)
        self.ffn = make_ffn(dim)
        self.sdpa_impl: str | None = "reference"

    def _heads(self, x: Tensor) -> Tensor:
        return x.view(x.shape[0], x.shape[1], self.num_heads, -1).transpose(1, 2)

    def forward(self, x0: Tensor, x1: Tensor) -> tuple[Tensor, Tensor]:
        """Each image attends to the other: ``m0 = attn(qk0, qk1, v1)``, ``m1 = attn(qk1, qk0, v0)``."""
        sdpa = ops.sdpa.get(self.sdpa_impl)
        if x0.shape == x1.shape:
            x = torch.cat((x0, x1))
            qk, v = self._heads(self.to_qk(x)), self._heads(self.to_v(x))
            m = sdpa(qk, qk.flip(0), v.flip(0))
            m0, m1 = m.transpose(1, 2).reshape(x.shape).chunk(2)
        else:
            qk0, qk1 = (self._heads(self.to_qk(x)) for x in (x0, x1))
            v0, v1 = (self._heads(self.to_v(x)) for x in (x0, x1))
            m0 = sdpa(qk0, qk1, v1).transpose(1, 2).reshape(x0.shape)
            m1 = sdpa(qk1, qk0, v0).transpose(1, 2).reshape(x1.shape)
        m0, m1 = apply_batched(self.to_out, [m0, m1])
        x0 = x0 + ffn_forward(self.ffn, torch.cat((x0, m0), -1))
        x1 = x1 + ffn_forward(self.ffn, torch.cat((x1, m1), -1))
        return x0, x1


class TransformerLayer(nn.Module):
    def __init__(self, dim: int, num_heads: int) -> None:
        super().__init__()
        self.self_attn = SelfBlock(dim, num_heads)
        self.cross_attn = CrossBlock(dim, num_heads)

    def forward(self, x0: Tensor, x1: Tensor, enc0: Tensor, enc1: Tensor) -> tuple[Tensor, Tensor]:
        x0, x1 = self.self_attn([x0, x1], [enc0, enc1])
        return self.cross_attn(x0, x1)


class TokenConfidence(nn.Module):
    """Per-keypoint probability that the descriptor is final (used for early stopping)."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.token = nn.Sequential(nn.Linear(dim, 1), nn.Sigmoid())

    def forward(self, x: Tensor) -> Tensor:
        return self.token(x).squeeze(-1)


class MatchAssignment(nn.Module):
    """Assignment head of one layer: similarity of the projected descriptors + per-point matchability."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.matchability = nn.Linear(dim, 1)
        self.final_proj = nn.Linear(dim, dim)

    def forward(self, x0: Tensor, x1: Tensor) -> Tensor:
        """Log assignment scores ``[B, M, N]`` (the dustbin row/column never yields a match and is omitted)."""
        dim = x0.shape[-1]
        m0, m1 = (self.final_proj(x).float() / dim**0.25 for x in (x0, x1))
        sim = m0 @ m1.transpose(-1, -2)
        z0, z1 = (self.matchability(x).float() for x in (x0, x1))
        certainty = F.logsigmoid(z0) + F.logsigmoid(z1).transpose(-1, -2)
        return F.log_softmax(sim, -1) + F.log_softmax(sim, -2) + certainty

    def matches(self, x0: Tensor, x1: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """Per point of image A (batch 1): the maximum of :meth:`forward`'s log assignment over image B's points,
        its index, and whether the pair is mutual -- without the ``[M, N]`` score matrix
        (``ops.dual_softmax_mutual``; the log matchabilities are its row / column biases)."""
        dim = x0.shape[-1]
        m0, m1 = (self.final_proj(x).float() / dim**0.25 for x in (x0, x1))
        sim = (m0 @ m1.transpose(-1, -2))[0]
        z0, z1 = (F.logsigmoid(self.matchability(x).float())[0, :, 0] for x in (x0, x1))
        return ops.dual_softmax_mutual(sim, z0, z1)


def filter_matches(scores: Tensor, threshold: float) -> tuple[Tensor, Tensor, Tensor]:
    """Mutual best matches of a log assignment matrix ``[1, M, N]`` whose probability exceeds ``threshold``:
    indices into image A, indices into image B, and the match probabilities."""
    best0, best1 = scores.max(2), scores.max(1)
    m0, m1 = best0.indices, best1.indices
    mutual = torch.arange(m0.shape[1], device=m0.device)[None] == m1.gather(1, m0)
    prob = best0.values.exp()
    valid = (mutual & (prob > threshold))[0]
    index_a = torch.nonzero(valid)[:, 0]
    return index_a, m0[0][index_a], prob[0][index_a]


class LightGlue(nn.Module):
    """Attention-based sparse matcher: 9 x (self-attention with rotary position, cross-attention) + assignment.

    Module paths follow the RDD checkpoint after :func:`weights.convert_state_dict`. ``match_features``
    takes the :class:`SparseFeatures` of two images and returns the mutual matches above the threshold.
    """

    def __init__(self, config: LightGlueConfig | None = None) -> None:
        super().__init__()
        self.config = cfg = config or LightGlueConfig()
        self.input_proj = nn.Linear(cfg.input_dim, cfg.dim) if cfg.input_dim != cfg.dim else nn.Identity()
        head_dim = cfg.dim // cfg.num_heads
        self.posenc = FourierPositionalEncoding(head_dim)
        self.transformers = nn.ModuleList(TransformerLayer(cfg.dim, cfg.num_heads) for _ in range(cfg.num_layers))
        self.log_assignment = nn.ModuleList(MatchAssignment(cfg.dim) for _ in range(cfg.num_layers))
        self.token_confidence = nn.ModuleList(TokenConfidence(cfg.dim) for _ in range(cfg.num_layers - 1))
        self.confidence_thresholds = [
            min(0.8 + 0.1 * math.exp(-4.0 * i / cfg.num_layers), 1.0) for i in range(cfg.num_layers)
        ]
        self.executed_layers = cfg.num_layers  # of the last match
        # Filled in by ``load_lightglue``.
        self.preset: str | None = None
        self.engine: str | None = None
        self.quant_report = None
        self.calibration_metadata: dict[str, str] | None = None
        self.compute_dtype = torch.float32

    @classmethod
    def from_pretrained(
        cls,
        config: LightGlueConfig | None = None,
        *,
        checkpoint: str | None = None,
        device: torch.device | str | None = None,
        state_dict: dict[str, Tensor] | None = None,
    ) -> LightGlue:
        device = torch.device(device or default_device())
        config = config or LightGlueConfig()
        with torch.device("meta"):
            model = cls(config)
        state = state_dict or weights.load_state_dict(config.num_heads, config.num_layers, checkpoint)
        model.load_state_dict(state, strict=True, assign=True)
        model.to(device)
        return model.eval()

    @property
    def device(self) -> torch.device:
        return self.posenc.Wr.weight.device

    def set_compute_dtype(self, dtype: torch.dtype) -> None:
        """Run the transformer layers and the assignment head's projections in ``dtype`` (positions stay FP32)."""
        for module in (self.input_proj, self.transformers, self.log_assignment, self.token_confidence):
            module.to(dtype)
        self.compute_dtype = dtype

    def normalized(self, keypoints: Tensor, image_size: tuple[int, int]) -> Tensor:
        height, width = image_size
        size = (height, width) if self.config.reference_size_order else (width, height)
        return normalize_keypoints(keypoints, device_constant(size, keypoints, torch.int64)[None])

    def encode(self, keypoints: Tensor, image_size: tuple[int, int]) -> Tensor:
        return self.posenc(self.normalized(keypoints, image_size))

    def should_stop(self, x0: Tensor, x1: Tensor, layer: int, depth_confidence: float) -> bool:
        """Host-synchronized exit test after ``layer``: enough keypoints are past the layer's confidence threshold."""
        confidence = torch.cat((self.token_confidence[layer](x0), self.token_confidence[layer](x1)), -1).float()
        ratio = 1.0 - (confidence < self.confidence_thresholds[layer]).sum() / confidence.numel()
        return bool(ratio > depth_confidence)

    def features(self, desc0: Tensor, desc1: Tensor, enc0: Tensor, enc1: Tensor) -> tuple[Tensor, Tensor, int]:
        """Point features after the last executed layer and its index (``executed_layers`` records how many ran)."""
        # contiguous: SuperPoint's descriptors are a transposed view, and `.to` keeps that layout for the whole residual
        # stream, so every quantized linear reading it fell back to the torch quantize (54 calls / match, 1.8-3.2 ms)
        x0, x1 = (
            self.input_proj(d.to(self.compute_dtype, memory_format=torch.contiguous_format)) for d in (desc0, desc1)
        )
        enc0, enc1 = enc0.to(self.compute_dtype), enc1.to(self.compute_dtype)
        depth_confidence = self.config.depth_confidence
        last = len(self.transformers) - 1
        for i, layer in enumerate(self.transformers):
            x0, x1 = layer(x0, x1, enc0, enc1)
            if depth_confidence > 0 and i < last and self.should_stop(x0, x1, i, depth_confidence):
                break
        self.executed_layers = i + 1
        return x0, x1, i

    def scores(self, desc0: Tensor, desc1: Tensor, enc0: Tensor, enc1: Tensor) -> Tensor:
        """Log assignment scores ``[1, M, N]`` after the last executed layer."""
        x0, x1, i = self.features(desc0, desc1, enc0, enc1)
        return self.log_assignment[i](x0, x1)

    @torch.inference_mode()
    def match_features(
        self, a: SparseFeatures, b: SparseFeatures, threshold: float | None = None
    ) -> Matches:
        if a.keypoints.shape[0] != 1 or b.keypoints.shape[0] != 1:
            raise ValueError("LightGlue matches one image pair at a time (batch size 1)")
        if a.keypoints.shape[1] == 0 or b.keypoints.shape[1] == 0:
            empty = a.keypoints.new_zeros(0, 2)
            return Matches(empty, empty, a.keypoints.new_zeros(0))
        x0, x1, i = self.features(
            a.descriptors, b.descriptors, self.encode(a.keypoints, a.image_size), self.encode(b.keypoints, b.image_size)
        )
        # same matches as ``filter_matches(self.scores(...))`` (identical lists on 200 MegaDepth pairs); the fused op
        # measured 4.2 to 4.8 ms less per 4096 x 4096 pair on sm_120 (SuperPoint 18.8 -> 14.6 ms, XFeat 8.8 -> 4.0 ms)
        log_conf, index, mutual = self.log_assignment[i].matches(x0, x1)
        conf = log_conf.exp()
        threshold = self.config.filter_threshold if threshold is None else threshold
        ia = torch.nonzero(mutual & (conf > threshold))[:, 0]
        ib = index[ia]
        return Matches(a.keypoints[0][ia], b.keypoints[0][ib], conf[ia], ia, ib)
