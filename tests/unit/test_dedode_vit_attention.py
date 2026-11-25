"""``vit_sage_attention`` pass of dedode-g: layer selection and platform gating (no GPU needed)."""

import pytest

from flashim.models.dedode import spec


def test_parse_layers():
    assert spec.parse_layers("2-23") == list(range(2, 24))
    assert spec.parse_layers("0,3,5") == [0, 3, 5]
    assert spec.parse_layers("!0-1") == list(range(2, 24))
    assert spec.parse_layers("all") == list(range(24))
    assert spec.parse_layers("") == []
    assert spec.parse_layers("4-6,10") == [4, 5, 6, 10]
    with pytest.raises(ValueError):
        spec.parse_layers("20-24")


def test_only_vit_models_have_the_pass():
    assert "vit_sage_attention" in spec.SPEC_G.passes
    assert "vit_sage_attention" not in spec.SPEC.passes
    from flashim.models.dad.spec import SPEC as DAD
    from flashim.models.loma.extractor_spec import SPEC as LOMA_EXT

    assert "vit_sage_attention" in DAD.passes
    assert "vit_sage_attention" in LOMA_EXT.passes


def test_platform_gate():
    import importlib.util
    from types import SimpleNamespace

    gate = spec.SPEC_G.passes["vit_sage_attention"].requires
    has_sage = importlib.util.find_spec("sageattention") is not None
    assert gate(SimpleNamespace(capability=(8, 6))) is False
    assert gate(SimpleNamespace(capability=(8, 9))) is has_sage
    assert gate(SimpleNamespace(capability=(9, 0))) is has_sage
    assert gate(SimpleNamespace(capability=(12, 0))) is has_sage
    assert gate(SimpleNamespace(capability=None)) is False
