import hashlib
import importlib
import re
from pathlib import Path

import pytest
import torch

import flashim
from flashim.models.weights_util import Checkpoint, find_weights

WEIGHT_MODULES = ("alike", "aliked", "dad", "dedode", "disk", "dust3r", "eloftr", "flexglue", "lightglue", "loftr",
                  "loma", "mast3r", "omniglue", "raco", "rdd", "ripe", "ripepp", "roma", "romav2", "superglue",
                  "superpoint", "tiny_roma", "ufm", "xfeat", "xoftr")


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


def test_license_warning_precedes_download_once_per_model(hub, monkeypatch, caplog):
    from flashim import log
    from flashim.models.superglue.weights import CHECKPOINTS
    from flashim.models.superpoint.weights import CHECKPOINTS as superpoint

    monkeypatch.setattr(log, "_warned", set())
    warnings_before_download = []

    def download(url, target, hash_prefix):
        warnings_before_download.append([record.getMessage() for record in caplog.records
                                         if "non-commercial" in record.getMessage()])
        Path(target).touch()

    monkeypatch.setattr(torch.hub, "download_url_to_file", download)
    hub.mkdir(parents=True)
    cached = hub / CHECKPOINTS["indoor"].filename
    cached.touch()
    assert find_weights(CHECKPOINTS["indoor"]) == cached
    assert find_weights(superpoint["v1"], path=cached) == cached
    assert not caplog.records
    cached.unlink()
    for checkpoint in (superpoint["sp_v6"], CHECKPOINTS["indoor"], CHECKPOINTS["outdoor"], superpoint["v1"]):
        assert find_weights(checkpoint).is_file()
    assert [len(messages) for messages in warnings_before_download] == [0, 1, 1, 2]
    first_model, second_model = warnings_before_download[-1]
    assert "superglue" in first_model and "LicenseRef-MagicLeap-SuperGlue" in first_model
    assert "superpoint" in second_model and "LicenseRef-MagicLeap-SuperPoint" in second_model


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


def test_romav2_load_uses_checkpoint_override(tmp_path, monkeypatch):
    checkpoint = tmp_path / "custom.pt"
    torch.save({"custom_weight": torch.ones(1)}, checkpoint)
    monkeypatch.setenv("FLASHIM_ROMAV2_CHECKPOINT", str(tmp_path / "unused.pt"))
    with pytest.raises(RuntimeError, match=r'Unexpected key\(s\) in state_dict: "custom_weight"'):
        flashim.load("romav2", device="cpu", overrides={"checkpoint": str(checkpoint)})


def _checkpoints(value):
    if isinstance(value, Checkpoint):
        yield value
    elif isinstance(value, dict):
        for item in value.values():
            yield from _checkpoints(item)


@pytest.mark.parametrize("family", ["dust3r", "mast3r", "ripe", "ripepp", "superglue", "superpoint", "ufm"])
def test_noncommercial_checkpoints_match_model_info(family):
    from flashim import model_info

    module = importlib.import_module(f"flashim.models.{family}.weights")
    checkpoints = {checkpoint for value in vars(module).values() for checkpoint in _checkpoints(value)}
    for checkpoint in checkpoints:
        if checkpoint.filename == "sp_v6.tgz":
            assert checkpoint.noncommercial is None
        else:
            assert checkpoint.noncommercial.model == family
            assert checkpoint.noncommercial.license in model_info(family).weights_licenses


def test_every_model_pins_its_release():
    found = {}
    for name in WEIGHT_MODULES:
        module = importlib.import_module(f"flashim.models.{name}.weights")
        for value in vars(module).values():
            for checkpoint in _checkpoints(value):
                assert checkpoint.url.startswith("https://"), checkpoint
                assert re.fullmatch(r"[0-9a-f]{64}", checkpoint.sha256), checkpoint
                assert found.setdefault(checkpoint.filename, checkpoint.sha256) == checkpoint.sha256, checkpoint
    assert len(found) == 54
