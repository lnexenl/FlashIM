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


def test_only_dedode_g_has_the_pass():
    assert "vit_sage_attention" in spec.SPEC_G.passes
    assert "vit_sage_attention" not in spec.SPEC.passes
    from flashim.models.dad.spec import SPEC as DAD

    assert "vit_sage_attention" not in DAD.passes  # dad shares the ViT but keeps its BF16 attention


def test_platform_gate():
    from types import SimpleNamespace

    gate = spec.SPEC_G.passes["vit_sage_attention"].requires
    assert gate(SimpleNamespace(capability=(8, 6))) is False
    assert gate(SimpleNamespace(capability=(12, 0))) is False
    assert gate(SimpleNamespace(capability=None)) is False
