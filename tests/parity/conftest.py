import pytest


@pytest.fixture(autouse=True, scope="module")
def _pil_decoder():
    """Image paths decode with PIL here, as the upstream oracles do (the default GPU decoder is 0.0008 levels away).
    Module-scoped so module-scoped fixtures (e.g. LoMa's fp32 reference matches) decode the same way as the tests, and
    undone after each module (a session fixture would leak into the tests run after tests/parity)."""
    from flashim.models.romav2 import model

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(model, "IMAGE_DECODER", "pil")
        yield
