"""RunConfig layering / serialization and the generic builder (no checkpoints needed)."""

import dataclasses
import json
import sys

import pytest
from torch import nn

import flashim
from flashim.build import EngineSpec, ModelSpec, Pass, build, get_spec, register_spec, resolve
from flashim.config import RunConfig, merge, parse_overrides
from flashim.platforms import Feature, Platform, Vendor


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


# -- FP8 presets on a GPU without FP8 ----------------------------------------------------------------------


def _gpu(*features: Feature) -> Platform:
    return Platform("cuda", 0, "test", (8, 7), frozenset(features), Vendor.NVIDIA, "sm_87")


NO_FP8, WITH_FP8 = _gpu(Feature.MMA_SYNC), _gpu(Feature.MMA_SYNC, Feature.FP8_E4M3)


def _uses_fp8(config: RunConfig) -> bool:
    quant = get_spec(config.model).quant_configs.get(config.precision.quant)
    own = quant is not None and any(scheme.weights.type != "int" for scheme in quant.config_groups.values())
    return own or any(_uses_fp8(component) for component in config.components.values())


def test_fp8_presets_fall_back_only_on_a_gpu_without_fp8():
    """Every preset of every model: untouched on the CPU and on a GPU with FP8; on a GPU without it the spec's fallback
    is resolved instead and nothing FP8 is left, components included (so an FP8 preset without a fallback fails here)."""
    for name in flashim.available_models():
        spec = get_spec(name)
        for preset in spec.presets:
            asked = resolve(name, preset)
            assert resolve(name, preset, platform=WITH_FP8) == asked == resolve(name, preset, platform=Platform("cpu"))
            built = resolve(name, preset, platform=NO_FP8)
            assert built.preset == spec.fp8_fallbacks.get(preset, preset), (name, preset)
            assert not _uses_fp8(built), (name, preset)


def test_fp8_fallback_of_pipelines_and_explicit_requests():
    # a pipeline falls back as a whole, so the matcher gets the INT8 preset's own (variant) calibration table;
    # naming the matcher's preset as well, as the keyword loaders do, changes nothing
    int8 = resolve("aliked-lightglue", "int8")
    assert resolve("aliked-lightglue", "fp8", platform=NO_FP8) == int8
    assert resolve("aliked-lightglue", "fp8", {"components": {"matcher": {"preset": "fp8"}}}, platform=NO_FP8) == int8
    # RDD has no INT8 preset: BF16, whatever the pipeline's (= the matcher's) preset is
    config = resolve("rdd-lightglue", "bf16", platform=NO_FP8)
    assert (config.preset, config.components["extractor"].preset) == ("bf16", "bf16")
    assert resolve("romav2", platform=NO_FP8).preset == "int8-full" and resolve("romav2").preset == "fp8"
    # explicit requests keep FP8 (emulated): strict, or a calibration table passed for an FP8 preset, anywhere in the
    # config and for all of it (a pipeline's preset and its matcher's table go together)
    assert resolve("romav2", "fp8", overrides=["optimize.strict=true"], platform=NO_FP8).preset == "fp8"
    assert resolve("lightglue", "fp8", overrides=["precision.calibration=t.safetensors"], platform=NO_FP8).preset == "fp8"
    for user in ({"components": {"matcher": {"precision": {"calibration": "t.safetensors"}}}}, {"optimize": {"strict": True}}):
        assert resolve("rdd-lightglue", "fp8", user, platform=NO_FP8) == resolve("rdd-lightglue", "fp8", user)


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


def test_build_falls_back_for_the_target_platform_and_reports_it(monkeypatch):
    presets = {"fp8": {"inference": {"format": "fp8"}}, "int8": {"inference": {"format": "int8"}}}
    spec = dataclasses.replace(_toy_spec(), name="toy-fp8", presets=presets, default_preset="fp8", fp8_fallbacks={"fp8": "int8"})
    register_spec("toy-fp8", spec)
    builder = sys.modules["flashim.build"]  # ``flashim.build`` itself is the function
    monkeypatch.setattr(builder, "platform_for", lambda device: NO_FP8)
    model = build("toy-fp8")
    assert (model.preset, model.run_config.inference) == ("int8", {"format": "int8"})
    assert model.build_report.fallbacks == {"toy-fp8": "fp8 -> int8"}
    monkeypatch.setattr(builder, "platform_for", lambda device: WITH_FP8)
    model = build("toy-fp8")
    assert model.preset == "fp8" and model.build_report.fallbacks == {}


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
