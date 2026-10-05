"""RunConfig layering / serialization and the generic builder (no checkpoints needed)."""

import json

import pytest
from torch import nn

import flashim
from flashim.build import EngineSpec, ModelSpec, Pass, build, get_spec, register_spec, resolve
from flashim.config import RunConfig, merge, parse_overrides


def test_merge_replaces_lists_and_appends_with_plus():
    base = {"optimize": {"passes": ["a"], "strict": False}}
    assert merge(base, {"optimize": {"passes": ["b"]}})["optimize"] == {"passes": ["b"], "strict": False}
    assert merge(base, {"optimize": {"passes+": ["b"]}})["optimize"]["passes"] == ["a", "b"]
    assert base == {"optimize": {"passes": ["a"], "strict": False}}  # inputs untouched


def test_parse_overrides():
    overlay = parse_overrides(["engine.kind=compile", "optimize.passes+=x", "inference.top_k=2048", "optimize.strict=true"])
    assert overlay == {
        "engine": {"kind": "compile"},
        "optimize": {"passes+": "x", "strict": True},
        "inference": {"top_k": 2048},
    }
    with pytest.raises(ValueError):
        parse_overrides(["nonsense"])


def test_roundtrip_digest_and_unknown_keys():
    config = resolve("romav2", "fp8-fast")
    again = RunConfig.from_dict(json.loads(json.dumps(config.to_dict())))
    assert again == config and again.digest() == config.digest()
    assert resolve("romav2", "fp8").digest() != config.digest()
    with pytest.raises(ValueError, match="Unknown config key"):
        RunConfig.from_dict({"model": "romav2", "optimise": {}})
    with pytest.raises(ValueError, match="unknown pass"):
        resolve("romav2", "fp8", overrides=["optimize.passes+=nope"])
    with pytest.raises(ValueError, match="compute_dtype"):
        resolve("romav2", "fp8", overrides=["precision.compute_dtype=fp64"])


def test_presets_resolve_and_are_not_mutated():
    for name in flashim.available_models():
        spec = get_spec(name)
        before = json.dumps(spec.presets, sort_keys=True, default=str)
        for preset in spec.presets:
            resolve(name, preset)
        assert json.dumps(spec.presets, sort_keys=True, default=str) == before, name


def test_romav2_presets_match_the_legacy_flags():
    from flashim.models.romav2.quant import PRESETS

    fast = resolve("romav2", "fp8-fast")
    assert fast.precision.quant == "fp8-fast" and PRESETS["fp8-fast"].batched_directions
    assert fast.optimize.passes == (
        "fold_refiners", "fused_bf16_s1", "fused_refiner_io", "fused_mx_blocks", "batched_directions",
    )
    assert resolve("romav2", "bf16").optimize.passes == ()
    with pytest.raises(ValueError, match="needs a quantized preset"):
        resolve("romav2", "bf16", overrides=["optimize.attention=sage"])


def test_pipeline_components_and_propagation(tmp_path):
    path = tmp_path / "run.toml"
    path.write_text('device = "cpu"\n[engine]\nkind = "compile"\n[components.extractor]\npreset = "bf16"\n')
    config = resolve("raco-aliked-lightglue", "int8", config=path)
    extractor, matcher = config.components["extractor"], config.components["matcher"]
    assert (matcher.preset, matcher.precision.quant, matcher.variant) == ("int8", "int8", "raco-aliked")
    raco, aliked = extractor.components["raco"], extractor.components["aliked"]
    assert "fused_ranker" in raco.optimize.passes and aliked.precision.compute_dtype == "bf16"
    assert {c.engine.kind for c in (matcher, raco, aliked)} == {"compile"}
    assert {c.device for c in (matcher, raco, aliked)} == {"cpu"}


# -- builder stages on a fake model ------------------------------------------------------------------------


class _Toy(nn.Module):
    def __init__(self):
        super().__init__()
        self.lin = nn.Linear(4, 4)
        self.log: list[str] = []


def _toy_spec():
    def mark(label):
        return lambda model, ctx: model.log.append(label) or label

    return ModelSpec(
        name="toy",
        role="dense",
        description="test model",
        load=lambda config: _Toy(),
        presets={"base": {"optimize": {"passes": ["late", "early", "gpu_only"]}}},
        default_preset="base",
        passes={
            "early": Pass(mark("early"), stage="pre_quant"),
            "late": Pass(mark("late")),
            "gpu_only": Pass(mark("gpu_only"), requires=lambda p: p.device_type == "gpu-that-does-not-exist"),
        },
        engines={"eager": EngineSpec()},
        set_precision=lambda model, config: model.log.append(f"precision:{config.precision.compute_dtype}"),
        apply_inference=lambda model, inference: model.log.append(f"inference:{sorted(inference)}"),
    )


def test_build_runs_stages_in_order_and_reports():
    register_spec("toy", _toy_spec())
    model = build("toy", overrides=["inference.top_k=5"])
    assert model.log == ["early", "precision:fp32", "late", "inference:['top_k']"]
    report = model.build_report
    assert set(report.passes) == {"early", "late"} and "gpu_only" in report.skipped
    assert report.config_digest == model.run_config.digest() and model.preset == "base"
    with pytest.raises(RuntimeError, match="gpu_only"):
        build("toy", overrides=["optimize.strict=true"])


@pytest.mark.parametrize("name", ["aliked", "raco"])
def test_keyword_api_matches_config_api(name):
    """The shims translate keyword arguments into the same resolved config as the config API."""
    from flashim.models.aliked.spec import legacy_overrides as aliked_overrides

    if name == "aliked":
        a = resolve("aliked", "bf16", config=aliked_overrides(engine="compile", channels_last=True))
        b = resolve("aliked", "bf16", overrides=["engine.kind=compile", "optimize.passes+=channels_last"])
        assert a.optimize.passes == b.optimize.passes == ("fold_batchnorm", "channels_last")
        assert a.engine == b.engine
    else:
        from flashim.models.raco.build import raco_aliked_overlay

        a = resolve("raco-aliked", "bf16", config=raco_aliked_overlay("bf16", fused_ranker=False))
        assert a.components["raco"].optimize.passes == ("fold_batchnorm",)
