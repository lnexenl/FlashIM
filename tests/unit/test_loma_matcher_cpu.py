"""LoMa's matcher without a checkpoint: variants, coordinates, assignment and checkpoint layout on CPU."""

import dataclasses

import pytest
import torch

import flashim
from flashim.models.lightglue.weights import convert_state_dict
from flashim.models.loma import weights
from flashim.models.loma.matcher import VARIANTS, DualSoftmaxAssignment, LoMaMatcher, normalize_keypoints
from flashim.quant.rotation import hadamard
from flashim.types import SparseFeatures

SMALL = dataclasses.replace(VARIANTS["b128"], input_dim=16, dim=32, num_heads=2, num_layers=2)


def test_variants_follow_upstream_configs():
    shapes = {
        v: (c.input_dim, c.dim, c.num_heads, c.num_layers, c.filter_threshold) for v, c in VARIANTS.items()
    }
    assert shapes == {"b128": (128, 256, 4, 9, 0.1), "b": (256, 256, 4, 9, 0.1), "l": (256, 512, 8, 9, 0.1),
                      "g": (256, 1024, 16, 9, 0.1), "r": (256, 256, 4, 9, 0.1)}
    assert all(c.dim // c.num_heads == 64 for c in VARIANTS.values())  # posenc.Wr (32, 2) in every release


def test_normalized_coordinates_span_the_image_edges():
    # FlashIM pixel centers -> DaD's [-1, 1]: the first / last pixel centers sit half a pixel inside the edges
    keypoints = torch.tensor([[[0.0, 0.0], [99.0, 49.0], [49.5, 24.5]]])
    out = normalize_keypoints(keypoints, (50, 100))
    torch.testing.assert_close(out, torch.tensor([[[-0.99, -0.98], [0.99, 0.98], [0.0, 0.0]]]))
    upstream_px = 100 * (out[..., 0] + 1) / 2  # loma.to_pixel_coords: pixel-edge convention
    torch.testing.assert_close(upstream_px - 0.5, keypoints[..., 0])


def test_no_token_confidence_and_dual_softmax_heads():
    model = LoMaMatcher(SMALL)
    keys = set(model.state_dict())
    assert not any(k.startswith("token_confidence") for k in keys)
    assert {"log_assignment.1.matchability.weight", "input_proj.weight", "posenc.Wr.weight"} <= keys
    assert all(isinstance(m, DualSoftmaxAssignment) for m in model.log_assignment)
    with pytest.raises(ValueError, match="depth_confidence"):
        LoMaMatcher(dataclasses.replace(SMALL, depth_confidence=0.95))


def test_assignment_is_the_product_of_softmaxes_without_matchability():
    torch.manual_seed(0)
    head = DualSoftmaxAssignment(32)
    torch.nn.init.constant_(head.matchability.bias, -5.0)  # would change LightGlue's scores, not LoMa's
    x0, x1 = torch.randn(1, 6, 32), torch.randn(1, 8, 32)
    sim = (head.final_proj(x0) / 32**0.25) @ (head.final_proj(x1) / 32**0.25).transpose(-1, -2)
    with torch.inference_mode():
        torch.testing.assert_close(head(x0, x1).exp(), sim.softmax(-1) * sim.softmax(-2))
        log_conf, index, mutual = head.matches(x0, x1)
        scores = head(x0, x1)[0]
    torch.testing.assert_close(log_conf, scores.max(1).values)
    assert torch.equal(index, scores.argmax(1))
    assert torch.equal(mutual, scores.argmax(0)[index] == torch.arange(6))


def _upstream_checkpoint(model: LoMaMatcher) -> dict[str, torch.Tensor]:
    """A release-shaped state dict: matcher keys (interleaved Wqkv), an extra layer and extractor keys."""
    heads = model.config.num_heads
    state = {}
    for key, value in model.state_dict().items():
        if ".Wqkv." in key:  # back to upstream's [head, dim, (q, k, v)] rows
            value = value.reshape(3, heads, -1, *value.shape[1:]).movedim(0, 2).reshape(value.shape)
        state[key] = value
    extra = len(model.transformers)
    state[f"transformers.{extra}.self_attn.Wqkv.weight"] = torch.zeros(1)
    state["_detector.encoder.layers.0.weight"] = torch.zeros(1)
    state["_descriptor.decoder.layers.1.out_conv.bias"] = torch.zeros(1)
    return state


def test_release_layout_loads_strictly_and_round_trips():
    torch.manual_seed(0)
    source = LoMaMatcher(SMALL)
    parts = weights.split_state_dict(_upstream_checkpoint(source), n_layers=SMALL.num_layers)
    assert set(parts.detector) == {"encoder.layers.0.weight"} and parts.dinov2 is None
    loaded = LoMaMatcher.from_pretrained(
        SMALL, state_dict=convert_state_dict(parts.matcher, SMALL.num_heads, SMALL.num_layers), device="cpu"
    )
    for key, value in source.state_dict().items():
        assert torch.equal(loaded.state_dict()[key], value), key


def test_match_features_mutual_above_threshold():
    torch.manual_seed(0)
    model = LoMaMatcher(SMALL).eval()
    size = (48, 64)
    feats = [SparseFeatures(torch.rand(1, n, 2) * torch.tensor([63.0, 47.0]), torch.zeros(1, n),
                            torch.randn(1, n, 16), size) for n in (20, 30)]
    with torch.inference_mode():
        matches = model.match_features(*feats, threshold=0.0)
        enc = [model.encode(f.keypoints, size) for f in feats]
        probs = model.scores(feats[0].descriptors, feats[1].descriptors, *enc).exp()[0]
    best = probs.argmax(1)
    expected = [i for i in range(20) if probs.argmax(0)[best[i]] == i]
    assert matches.indices_A.tolist() == expected
    assert matches.indices_B.tolist() == best[expected].tolist()
    torch.testing.assert_close(matches.confidence, probs.max(1).values[expected])
    torch.testing.assert_close(matches.keypoints_A, feats[0].keypoints[0][expected])


def test_registered_with_every_preset_and_a_table_per_variant():
    from flashim.build import PRECISIONS, get_spec

    spec = get_spec("loma-matcher")
    assert spec.role == "matcher" and set(spec.presets) == set(PRECISIONS)
    for variant in VARIANTS:
        for quant in ("int8", "fp8"):
            assert spec.calibrations[f"{variant}/{quant}"].is_file(), (variant, quant)
        assert flashim.resolve("loma-matcher", "int8", config={"variant": variant}).precision.quant == "int8"


@pytest.mark.parametrize("input_dim", [16, 32])  # an input projection / nn.Identity (B, R)
def test_rotated_residual_stream_keeps_the_function(input_dim):
    import copy

    from flashim.models.lightglue.quant import rotate_residual_stream

    torch.manual_seed(0)
    model = LoMaMatcher(dataclasses.replace(SMALL, input_dim=input_dim)).eval()
    for param in model.parameters():
        param.data.normal_(0, 0.3)
    rotated = copy.deepcopy(model)
    # input_proj + 2 layers x (Wqkv, out_proj, to_qk, to_v, to_out, 2 ffn.0, 2 ffn.3) + 2 x (final_proj, matchability)
    assert rotate_residual_stream(rotated) == 1 + 2 * 9 + 2 * 2
    assert isinstance(rotated.input_proj, torch.nn.Linear)
    d0, d1 = torch.randn(1, 50, input_dim), torch.randn(1, 40, input_dim)
    e0, e1 = model.posenc(torch.rand(1, 50, 2)), model.posenc(torch.rand(1, 40, 2))
    with torch.inference_mode():
        stream = model.features(d0, d1, e0, e1)[:2]
        rotated_stream = rotated.features(d0, d1, e0, e1)[:2]
        scores, rotated_scores = model.scores(d0, d1, e0, e1), rotated.scores(d0, d1, e0, e1)
    for x, r in zip(stream, rotated_stream):
        assert not torch.allclose(x, r, atol=1e-2)  # the stream itself lives in another basis
        torch.testing.assert_close(r @ hadamard(32), x, atol=1e-4, rtol=1e-4)
    torch.testing.assert_close(rotated_scores, scores, atol=2e-4, rtol=1e-4)


def test_int8_preset_rotates_the_residual_stream():
    for variant in VARIANTS:
        config = flashim.resolve("loma-matcher", "int8", config={"variant": variant})
        assert config.optimize.passes == ("rotate_residual_stream",)
    assert flashim.resolve("loma-matcher", "fp8").optimize.passes == ()
