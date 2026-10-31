"""LoMa's pipelines resolve without weights: one per variant (``loma`` = ``loma-b``), every preset applied to both
components with the same variant."""

import pytest

import flashim
from flashim.build import PRECISIONS, get_spec
from flashim.models.loma import spec as loma_spec


@pytest.mark.parametrize("name", ["loma", "loma-b128", "loma-b", "loma-l", "loma-g", "loma-r"])
@pytest.mark.parametrize("preset", PRECISIONS)
def test_pipeline_components_follow_the_preset(name, preset):
    variant = "b" if name == "loma" else name.removeprefix("loma-")
    config = flashim.resolve(name, preset)
    extractor, matcher = config.components["extractor"], config.components["matcher"]
    assert (extractor.model, matcher.model) == ("loma-extractor", "loma-matcher")
    assert extractor.variant == matcher.variant == variant
    quant = preset if preset in ("int8", "fp8") else None
    assert extractor.precision.quant == matcher.precision.quant == quant


def test_loma_is_loma_b():
    loma, loma_b = get_spec("loma"), get_spec("loma-b")
    assert loma.presets == loma_b.presets and loma.default_preset == loma_b.default_preset == loma_spec.DEFAULT_PRESET
    assert loma.role == "pipeline"
