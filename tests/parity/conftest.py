import pytest


@pytest.fixture(autouse=True)
def _pil_decoder(monkeypatch):
    """Image paths decode with PIL here, as the upstream oracles do (the default GPU decoder is 0.0008 levels away)."""
    from flashim.models.romav2 import model

    monkeypatch.setattr(model, "IMAGE_DECODER", "pil")
