"""FlexGlue without a checkpoint: entmax readout, rotary, transformer/match-head units, decode_pair gating,
checkpoint key conversion and point normalization on CPU."""

import dataclasses

import torch

from flashim import ops
from flashim.models.flexglue import FlexGlue, FlexGlueConfig
from flashim.models.flexglue.model import (
    MatchHead,
    TransformerLayer,
    apply_rotary,
    cell_centers,
    denormalize_keypoints,
    normalize_keypoints,
    rotate_half,
)
from flashim.models.flexglue.weights import convert_state_dict
from flashim.ops.flexglue import _sorted_entmax15_threshold

SMALL = FlexGlueConfig(input_dim=32, descriptor_dim=64, search_dim=16, num_layers=2, num_heads=4)


def _entmax15_window(logits: torch.Tensor, grid: torch.Tensor, hw: tuple[int, int], radius: int = 1):
    """Straightforward independent 1.5-entmax readout: window around the argmax, p = (y - tau)_+^2."""
    h, w = hw
    idx = logits.argmax(-1, keepdim=True)
    vmax = logits.gather(-1, idx)
    y = (logits - vmax) / 2
    tau = _sorted_entmax15_threshold(y.sort(-1, descending=True).values)
    r = torch.arange(-radius, radius + 1, device=logits.device)
    ny = (idx // w)[..., None] + r[:, None]
    nx = (idx % w)[..., None] + r[None, :]
    inside = ((ny >= 0) & (ny < h) & (nx >= 0) & (nx < w)).flatten(-2)
    nidx = (ny.clamp(0, h - 1) * w + nx.clamp(0, w - 1)).flatten(-2)
    p = (y.gather(-1, nidx) - tau).clamp_min(0).square()
    position = (p[..., None] * grid[nidx]).sum(-2) / p.sum(-1, keepdim=True)
    return position, (p * inside).sum(-1)


def test_entmax_readout_matches_independent_computation():
    torch.manual_seed(0)
    h, w = 3, 4
    logits = torch.randn(2, 5, h * w)
    grid = cell_centers(h, w, scale=(w, h), device=logits.device)
    position, mass = ops.flexglue_entmax_readout(logits, grid, (h, w))
    exp_position, exp_mass = _entmax15_window(logits, grid, (h, w))
    assert torch.allclose(position, exp_position, atol=1e-6)
    assert torch.allclose(mass, exp_mass, atol=1e-6)
    # positions are convex combinations of the (2r+1)^2 window's cell centers -> inside the [-1, 1] map
    assert (position.abs() <= 1 + 1e-6).all()
    assert (mass >= 0).all() and (mass <= 1 + 1e-6).all()


def test_rotary_interleaved_pairs():
    assert torch.equal(rotate_half(torch.tensor([1.0, 2.0, 3.0, 4.0])), torch.tensor([-2.0, 1.0, -4.0, 3.0]))
    # sin-only encoding rotates each interleaved pair by +90 degrees
    encoding = torch.zeros(2, 1, 1, 1, 4)
    encoding[1] = 1.0
    t = torch.tensor([1.0, 2.0, 3.0, 4.0]).view(1, 1, 1, 4)
    assert torch.allclose(apply_rotary(encoding, t), torch.tensor([[-2.0, 1.0, -4.0, 3.0]]).view(1, 1, 1, 4))
    # a random (cos, sin) is a 2D rotation on each interleaved pair: pair norms are preserved
    torch.manual_seed(0)
    tokens = torch.randn(1, 5, 4, 8)
    theta = torch.randn(1, 5, 1, 4)  # one angle per pair, broadcast over the heads
    enc = torch.stack([theta.cos(), theta.sin()], 0).repeat_interleave(2, dim=-1)  # [2, B, N, 1, D]
    out = apply_rotary(enc, tokens)
    t1, t2 = tokens.unflatten(-1, (-1, 2)).unbind(-1)
    o1, o2 = out.unflatten(-1, (-1, 2)).unbind(-1)
    assert torch.allclose((o1.square() + o2.square()).sqrt(), (t1.square() + t2.square()).sqrt(), atol=1e-5)


def test_transformer_layer_shapes_and_no_nan():
    torch.manual_seed(0)
    layer = TransformerLayer(dim=64, num_heads=4).eval()
    x0, x1 = torch.randn(1, 7, 64), torch.randn(1, 5, 64)
    d = 64 // 4
    enc0, enc1 = torch.randn(2, 1, 1, 7, d), torch.randn(2, 1, 1, 5, d)
    with torch.no_grad():
        y0, y1 = layer(x0, x1, enc0, enc1)
    assert y0.shape == (1, 7, 64) and y1.shape == (1, 5, 64)
    assert not torch.isnan(y0).any() and not torch.isnan(y1).any()


def test_match_head_shapes_and_chunking():
    torch.manual_seed(0)
    cf_i = torch.randn(1, 64, 3, 4)
    cf_j = torch.randn(1, 64, 5, 6)
    points = torch.rand(1, 7, 2) * 2 - 1
    for cross in (True, False):
        head = MatchHead(dim=64, search_dim=16, query_chunk_size=None, cross=cross).eval()
        with torch.no_grad():
            p2d, conf = head(points, cf_i, cf_j)
        assert p2d.shape == (1, 7, 2) and conf.shape == (1, 7)
        assert not torch.isnan(p2d).any() and not torch.isnan(conf).any()
    full = MatchHead(dim=64, search_dim=16, query_chunk_size=None, cross=True).eval()
    chunked = MatchHead(dim=64, search_dim=16, query_chunk_size=3, cross=True).eval()
    chunked.load_state_dict(full.state_dict())
    with torch.no_grad():
        p2d_full, conf_full = full(points, cf_i, cf_j)
        p2d_chunk, conf_chunk = chunked(points, cf_i, cf_j)
    assert torch.allclose(p2d_chunk, p2d_full, atol=1e-6)
    assert torch.allclose(conf_chunk, conf_full, atol=1e-6)


def test_decode_pair_layer_index_and_early_stop_gate():
    torch.manual_seed(0)
    model = FlexGlue(SMALL).eval()
    for p in model.parameters():
        torch.nn.init.normal_(p, std=0.02)
    f0, f1 = torch.randn(1, SMALL.input_dim, 4, 4), torch.randn(1, SMALL.input_dim, 4, 4)
    with torch.no_grad():
        layer, score, cf0, cf1 = model.decode_pair(f0, f1)
    assert layer == SMALL.num_layers  # no early stopping: every layer runs
    assert cf0.shape == (1, SMALL.descriptor_dim, 4, 4) and cf1.shape == (1, SMALL.descriptor_dim, 4, 4)
    assert not torch.isnan(cf0).any() and not torch.isnan(score).any()
    with torch.no_grad():  # layer-1 covisibility so negative every point stops after layer 1
        model.heads[1].covisibility.bias.fill_(-50.0)
    model.config = dataclasses.replace(SMALL, early_stop_decay=0.5, min_covisibility=1.0)
    with torch.no_grad():
        layer, score, _, _ = model.decode_pair(f0, f1)
    assert 1 <= layer < SMALL.num_layers and score.item() < 1.0
    assert torch.allclose(score, model.covisibility_score(*_tokens_after(model, f0, f1, layer), layer))
    # gate enabled but not firing: the score is recomputed with the last layer's head, not the last checked one
    model.config = dataclasses.replace(SMALL, early_stop_decay=0.5, min_covisibility=1e-6)
    with torch.no_grad():
        model.heads[1].covisibility.bias.fill_(0.0)
        layer, score, _, _ = model.decode_pair(f0, f1)
        expected = model.covisibility_score(*_tokens_after(model, f0, f1, SMALL.num_layers), SMALL.num_layers)
    assert layer == SMALL.num_layers and torch.allclose(score, expected)


def _tokens_after(model, f0, f1, layers):
    x0, x1 = (model.input_proj(f.flatten(2).mT) for f in (f0, f1))
    enc0, enc1 = (model._posenc(f) for f in (f0, f1))
    with torch.no_grad():
        for block in model.transformers[:layers]:
            x0, x1 = block(x0, x1, enc0, enc1)
    return x0, x1


def test_config_validation_and_truncation():
    import pytest

    from flashim.models.flexglue.weights import truncate

    for bad in ({"early_stop_decay": 1.5, "min_covisibility": 0.1}, {"early_stop_decay": 0.9},
                {"query_chunk_size": 7}, {"num_layers": 10}, {"num_layers": 0}, {"describe_resize": 640},
                {"describe_resize": (500, 512)}):
        with pytest.raises(ValueError):
            dataclasses.replace(FlexGlueConfig(), **bad).validate()
    dataclasses.replace(FlexGlueConfig(), num_layers=5, describe_resize=(640, 480)).validate()
    full = FlexGlue(SMALL).state_dict()
    small = dataclasses.replace(SMALL, num_layers=1)
    FlexGlue(small).load_state_dict(truncate(full, 1), strict=True)


def test_convert_state_dict_renames():
    state = {
        "matcher.posenc.Wr.weight": torch.zeros(4, 2),
        "matcher.transformers.0.self_attn.Wqkv.weight": torch.zeros(192, 64),
        "matcher.heads.0.disambiguate.weight": torch.zeros(1, 64),
        "extractor.backbone.stages.0.0.dwconv.weight": torch.zeros(64, 1, 7, 7),
        "extractor.head.down.bias": torch.zeros(32),
    }
    out = convert_state_dict(state)
    assert "posenc_wr.weight" in out
    assert "extractor." not in out and "matcher." not in out and "disambiguate" not in out
    assert out["descriptor.backbone.stages.0.0.dwconv.weight"].shape == (64, 1, 7, 7)
    assert out["heads.0.covisibility.weight"].shape == (1, 64)
    assert out["transformers.0.self_attn.Wqkv.weight"].shape == (192, 64)


def test_normalize_keypoints_round_trip_and_cell_centers():
    size = torch.tensor([64.0, 48.0]).view(1, 2)
    keypoints = torch.rand(1, 6, 2) * size.view(1, 1, 2)
    normalized = normalize_keypoints(keypoints, size)
    assert (normalized.abs() <= 1).all()
    assert torch.allclose(denormalize_keypoints(normalized, size), keypoints, atol=1e-5)
    h, w = 3, 4
    centers = cell_centers(h, w, scale=(w, h), device=size.device)
    assert centers.shape == (h * w, 2)
    assert (centers >= -1).all() and (centers <= 1).all()


def test_attention_impl_is_routed_and_fp32_rejects_it():
    import pytest

    from flashim.config import RunConfig
    from flashim.models.flexglue.spec import _set_attention, _validate

    model = FlexGlue(SMALL)
    _set_attention(model, "sage")
    assert all(layer.self_attn.sdpa_impl == layer.cross_attn.sdpa_impl == "sage" for layer in model.transformers)
    with pytest.raises(ValueError):
        _validate(RunConfig.from_dict({"model": "flexglue", "optimize": {"attention": "sage"}}))
