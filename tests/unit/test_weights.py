import hashlib
import importlib
import re

import pytest
import torch

from flashim.models.weights_util import Checkpoint, find_weights

WEIGHT_MODULES = ("alike", "aliked", "dad", "dedode", "disk", "lightglue", "loftr", "loma", "raco", "rdd", "ripe",
                  "romav2", "superpoint", "xfeat")


@pytest.fixture
def hub(tmp_path, monkeypatch):
    monkeypatch.setattr(torch.hub, "get_dir", lambda: str(tmp_path / "hub"))
    return tmp_path / "hub" / "checkpoints"


@pytest.fixture
def release(tmp_path):
    source = tmp_path / "upstream.pth"
    source.write_bytes(b"weights")
    return Checkpoint("model.pth", source.as_uri(), hashlib.sha256(b"weights").hexdigest())


def test_downloads_once_into_the_hub_folder(hub, release):
    assert find_weights(release) == hub / "model.pth"
    assert (hub / "model.pth").read_bytes() == b"weights"
    (hub / "model.pth").write_bytes(b"local copy")  # present files are used as they are
    assert find_weights(release).read_bytes() == b"local copy"


def test_hash_mismatch_leaves_nothing(hub, release):
    bad = Checkpoint(release.filename, release.url, "0" * 64)
    with pytest.raises(FileNotFoundError, match="downloading"):
        find_weights(bad)
    assert not hub.exists() or not any(hub.iterdir())


def test_explicit_path_and_env_take_precedence(hub, release, tmp_path, monkeypatch):
    mine = tmp_path / "mine.pth"
    mine.write_bytes(b"mine")
    assert find_weights(release, path=mine) == mine
    monkeypatch.setenv("FLASHIM_TEST_CHECKPOINT", str(mine))
    assert find_weights(release, env="FLASHIM_TEST_CHECKPOINT") == mine
    with pytest.raises(FileNotFoundError, match="checkpoint="):
        find_weights(release, path=tmp_path / "missing.pth")  # never silently replaced by a download
    monkeypatch.setenv("FLASHIM_TEST_CHECKPOINT", str(tmp_path / "missing.pth"))
    with pytest.raises(FileNotFoundError, match="FLASHIM_TEST_CHECKPOINT"):
        find_weights(release, env="FLASHIM_TEST_CHECKPOINT")
    assert not hub.exists()


def _checkpoints(value):
    if isinstance(value, Checkpoint):
        yield value
    elif isinstance(value, dict):
        for item in value.values():
            yield from _checkpoints(item)


def test_every_model_pins_its_release():
    found = {}
    for name in WEIGHT_MODULES:
        module = importlib.import_module(f"flashim.models.{name}.weights")
        for value in vars(module).values():
            for checkpoint in _checkpoints(value):
                assert checkpoint.url.startswith("https://"), checkpoint
                assert re.fullmatch(r"[0-9a-f]{64}", checkpoint.sha256), checkpoint
                assert found.setdefault(checkpoint.filename, checkpoint.sha256) == checkpoint.sha256, checkpoint
    assert len(found) == 34
