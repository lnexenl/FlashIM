"""RunConfig layering / serialization and the generic builder (no checkpoints needed)."""

import dataclasses
import json
import sys
from collections.abc import Mapping

import pytest
import torch
from torch import nn

import flashim
from flashim.build import (
    NEAREST,
    PRECISIONS,
    EngineSpec,
    ModelSpec,
    Pass,
    get_spec,
    load,
    register_spec,
    resolve,
)
from flashim.config import RunConfig, merge, parse_overrides
from flashim.models.base import ImageMatcher
from flashim.platforms import Feature, Platform, Vendor
from flashim.types import Matches


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
    config = resolve("romav2", "fp8")
    again = RunConfig.from_dict(json.loads(json.dumps(config.to_dict())))
    assert again == config and again.digest() == config.digest()
    assert resolve("romav2", "int8").digest() != config.digest()
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


def test_presets_are_the_four_precisions():
    for name in flashim.available_models():
        assert set(get_spec(name).presets) <= set(PRECISIONS), name
    with pytest.raises(ValueError, match="presets must be among"):
        dataclasses.replace(_toy_spec(), presets={"int8-full": {}}, default_preset="int8-full")
    with pytest.raises(ValueError, match="unknown preset 'fp8-p8'"):
        resolve("romav2", "fp8-p8")


def test_romav2_presets_match_the_legacy_flags():
    from flashim.models.romav2.quant import PRESETS
    from flashim.models.romav2.spec import ORACLE_OVERLAYS

    fast = resolve("romav2", "fp8")
    assert fast.precision.quant == "fp8" and PRESETS["fp8"].batched_directions
    assert fast.optimize.passes == (
        "fold_refiners", "fused_bf16_s1", "fused_refiner_io", "fused_mx_blocks", "batched_directions",
    )
    assert resolve("romav2", "bf16").optimize.passes == ()
    with pytest.raises(ValueError, match="needs a quantized preset"):
        resolve("romav2", "bf16", overrides=["optimize.attention=sage"])
    oracle = resolve("romav2", "bf16", config=ORACLE_OVERLAYS["fp8-s42"])  # the parity tests' configs
    assert (oracle.precision.quant, oracle.optimize.passes) == ("fp8-s42", ("fold_refiners",))


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


def test_pipeline_preset_is_every_components_nearest():
    fallbacks = {}
    config = resolve("aliked-lightglue", "int8", fallbacks=fallbacks)
    assert (config.components["extractor"].preset, config.components["matcher"].preset) == ("bf16", "int8")
    assert fallbacks == {"extractor": "int8 -> bf16 (aliked has no int8)"}
    fallbacks = {}
    config = resolve("raco-superpoint", "int8", fallbacks=fallbacks)
    assert (config.components["raco"].preset, config.components["descriptor"].preset) == ("bf16", "int8")
    assert list(fallbacks) == ["raco"]
    assert resolve("romav2", "fp32").preset == "bf16" and resolve("sift", "fp8").preset == "fp32"
    with pytest.raises(ValueError, match="no 'int8' preset"):
        resolve("aliked-lightglue", "int8", overrides=["optimize.strict=true"])  # strict: components included
    strict = resolve("rdd-lightglue", "fp8", overrides=["optimize.strict=true"])
    assert all(c.optimize.strict for c in strict.components.values())


# -- FP8 presets on a GPU without FP8 ----------------------------------------------------------------------


def _gpu(*features: Feature) -> Platform:
    return Platform("cuda", 0, "test", (8, 7), frozenset(features), Vendor.NVIDIA, "sm_87")


NO_FP8, WITH_FP8 = _gpu(Feature.MMA_SYNC), _gpu(Feature.MMA_SYNC, Feature.FP8_E4M3)


def _uses_fp8(config: RunConfig) -> bool:
    quant = get_spec(config.model).quant_configs.get(config.precision.quant)
    own = quant is not None and any(scheme.weights.type != "int" for scheme in quant.config_groups.values())
    return own or any(_uses_fp8(component) for component in config.components.values())


def test_fp8_presets_fall_back_only_on_a_gpu_without_fp8():
    """Every preset of every model: untouched on the CPU and on a GPU with FP8; on a GPU without it ``fp8`` is replaced
    by the nearest other preset and nothing FP8 is left, components included."""
    for name in flashim.available_models():
        spec = get_spec(name)
        for preset in spec.presets:
            asked = resolve(name, preset)
            assert resolve(name, preset, platform=WITH_FP8) == asked == resolve(name, preset, platform=Platform("cpu"))
            built = resolve(name, preset, platform=NO_FP8)
            nearest = next(p for p in NEAREST["fp8"] if p in spec.presets) if preset == "fp8" else preset
            assert built.preset == nearest, (name, preset)
            assert not _uses_fp8(built), (name, preset)


def test_fp8_fallback_of_pipelines_and_explicit_requests():
    # a pipeline falls back as a whole, so the matcher gets the INT8 preset's own (variant) calibration table;
    # naming the matcher's preset as well changes nothing
    int8 = resolve("aliked-lightglue", "int8")
    assert resolve("aliked-lightglue", "fp8", platform=NO_FP8) == int8
    assert resolve("aliked-lightglue", "fp8", {"components": {"matcher": {"preset": "fp8"}}}, platform=NO_FP8) == int8
    # RDD has no INT8 preset: the pipeline's int8 builds RDD bf16
    config = resolve("rdd-lightglue", "fp8", platform=NO_FP8)
    assert (config.preset, config.components["extractor"].preset, config.components["matcher"].preset) == (
        "int8", "bf16", "int8",
    )
    assert resolve("romav2", platform=NO_FP8).preset == "int8" and resolve("romav2").preset == "fp8"
    # explicit requests keep FP8 (emulated): strict, or a calibration table passed for an FP8 preset, anywhere in the
    # config and for all of it (a pipeline's preset and its matcher's table go together)
    assert resolve("romav2", "fp8", overrides=["optimize.strict=true"], platform=NO_FP8).preset == "fp8"
    assert resolve("lightglue", "fp8", overrides=["precision.calibration=t.safetensors"], platform=NO_FP8).preset == "fp8"
    for user in ({"components": {"matcher": {"precision": {"calibration": "t.safetensors"}}}}, {"optimize": {"strict": True}}):
        assert resolve("rdd-lightglue", "fp8", user, platform=NO_FP8) == resolve("rdd-lightglue", "fp8", user)



def test_lightglue_variants_default_to_their_own_calibration_table(monkeypatch):
    """A variant's table is the matcher spec's default, not a path in the pipeline preset: the resolved config (and
    its digest) does not depend on where flashim is installed, and a standalone ``lightglue`` gets it too."""
    from flashim import build
    from flashim.quant.calibration import CalibrationTable, Int8CalibrationTable

    for name in ("aliked-lightglue", "raco-aliked-lightglue", "rdd-lightglue"):
        for preset in ("int8", "fp8"):
            assert resolve(name, preset).components["matcher"].precision.calibration is None
    loaded = []
    for cls in (CalibrationTable, Int8CalibrationTable):
        monkeypatch.setattr(cls, "load", classmethod(lambda cls, path: loaded.append(path.stem)))
    spec = get_spec("lightglue")
    for variant, preset in (("aliked", "int8"), ("raco-aliked", "fp8"), ("rdd", "fp8"), (None, "int8")):
        build._load_calibration(spec, resolve("lightglue", preset, {"variant": variant}), int8=preset == "int8")
    assert loaded == [
        "lightglue_aliked_int8_mega", "lightglue_raco-aliked_fp8_mega", "lightglue_fp8_mega", "lightglue_int8_mega",
    ]

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
        presets={"fp32": {"optimize": {"passes": ["late", "early", "gpu_only"]}}},
        default_preset="fp32",
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
    model = load("toy", overrides=["inference.top_k=5"])
    assert model.log == ["early", "precision:fp32", "late", "inference:['top_k']"]
    report = model.build_report
    assert set(report.passes) == {"early", "late"} and "gpu_only" in report.skipped
    assert report.config_digest == model.run_config.digest() and model.preset == "fp32"
    with pytest.raises(RuntimeError, match="gpu_only"):
        load("toy", strict=True)
    model = load("toy", "int8", device="cpu", engine="eager", overrides={"optimize.passes": ["early"]})
    assert model.preset == "fp32" and model.build_report.fallbacks == {"toy": "int8 -> fp32 (toy has no int8)"}
    assert (model.run_config.device, model.run_config.optimize.passes) == ("cpu", ("early",))


def test_build_falls_back_for_the_target_platform_and_reports_it(monkeypatch):
    presets = {"fp8": {"inference": {"format": "fp8"}}, "int8": {"inference": {"format": "int8"}}}
    spec = dataclasses.replace(_toy_spec(), name="toy-fp8", presets=presets, default_preset="fp8")
    register_spec("toy-fp8", spec)
    builder = sys.modules["flashim.build"]
    monkeypatch.setattr(builder, "platform_for", lambda device: NO_FP8)
    model = load("toy-fp8")
    assert (model.preset, model.run_config.inference) == ("int8", {"format": "int8"})
    assert model.build_report.fallbacks == {"toy-fp8": f"fp8 -> int8 (no FP8 on {NO_FP8.describe()})"}
    monkeypatch.setattr(builder, "platform_for", lambda device: WITH_FP8)
    model = load("toy-fp8")
    assert model.preset == "fp8" and model.build_report.fallbacks == {}


class _ToyMatcher(ImageMatcher, _Toy):
    match_defaults: Mapping[str, int] = {"max_matches": 7}

    def _match(self, img_a, img_b, *, resize, max_matches, threshold):
        self.log.append((resize, max_matches, threshold))
        return Matches(torch.zeros(0, 2), torch.zeros(0, 2), torch.zeros(0))


def test_match_defaults_come_from_the_inference_section():
    register_spec("toy-matcher", dataclasses.replace(_toy_spec(), name="toy-matcher", load=lambda config: _ToyMatcher()))
    model = load("toy-matcher", overrides=["inference.resize=640", "inference.threshold=0.5", "inference.top_k=1"])
    assert model.match_defaults == {"max_matches": 7, "resize": 640, "threshold": 0.5}
    model.match(None, None)
    model.match(None, None, resize=320, max_matches=3)
    assert model.log[-2:] == [(640, 7, 0.5), (320, 3, 0.5)]
    for bad in (0, -1, 2.5, True):
        with pytest.raises(ValueError, match="max_matches"):
            model.match(None, None, max_matches=bad)
