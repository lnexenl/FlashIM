import pytest

import flashim
from scripts.model_license_table import DOCUMENT, TABLE_HEADING, render


def test_model_license_table_is_current():
    table = DOCUMENT.read_text(encoding="utf-8").split(TABLE_HEADING)[1]
    assert table == render(), "Run PYTHONPATH=. python scripts/model_license_table.py"


@pytest.mark.parametrize(("name", "licenses", "commercial_ok"), [
    ("tiny-roma", ("MIT",), True),
    ("ufm", ("CC-BY-NC-SA-4.0",), False),
    ("raco", ("Apache-2.0",), True),
    ("xfeat", ("Apache-2.0",), True),
    ("xfeat-lightglue", ("Apache-2.0",), True),
    ("dedode-g", ("Apache-2.0", "MIT"), True),
    ("loma-b128", ("MIT",), True),
    ("loma-g", ("Apache-2.0", "MIT"), True),
    ("rdd-lightglue", ("NOASSERTION",), False),
    ("aliked-lightglue", ("Apache-2.0", "BSD-3-Clause"), True),
    ("minima-roma", ("Apache-2.0",), True),
    ("orb", (), True),
    ("omniglue", ("Apache-2.0", "MIT"), True),
    ("ripepp-lightglue", ("LicenseRef-Fraunhofer-Academic-2.0",), False),
    ("raco-superpoint-lightglue", ("Apache-2.0", "LicenseRef-MagicLeap-SuperPoint"), False),
])
def test_default_weights_licenses(name, licenses, commercial_ok):
    info = flashim.model_info(name)
    assert info.weights_licenses == licenses
    assert info.commercial_ok is commercial_ok


def test_romav2_includes_dinov3_code_and_weights():
    info = flashim.model_info("romav2")
    assert "LicenseRef-DINOv3" in info.code_licenses and "LicenseRef-DINOv3" in info.weights_licenses
    assert not info.commercial_ok


def test_unknown_code_license_is_conservative():
    assert not flashim.ModelInfo(("unrecognized",), ("MIT",)).commercial_ok


def test_unregistered_model():
    with pytest.raises(ValueError, match="Unknown model"):
        flashim.model_info("missing-model")
