"""Every registered model has a row in the latency table (``benchmarks/speed``), or is an alias / component of one.

The rule (AGENTS.md, "Verifying changes"): a change that can change a model's latency re-measures its rows and updates
docs/benchmarks.md, and a new model adds its rows. This test makes the second half checkable: a model that is neither
in ``speed_table.py``'s lists nor in ``NOT_A_ROW`` fails, and every listed model must be scheduled by a speed script.
"""

from __future__ import annotations

import importlib.util
import re
from pathlib import Path

import flashim

SPEED = Path(__file__).resolve().parents[2] / "benchmarks" / "speed"

# Registered names that are not rows of their own, with the row that stands for them.
NOT_A_ROW = {
    "lightglue": "the matcher of every *-lightglue pipeline (Sparse matchers table)",
    "loma": "alias of loma-b",
    "loma-extractor": "component of the loma-* pipelines",
    "loma-matcher": "component of the loma-* pipelines",
    "omniglue-extractor": "component of omniglue",
    "omniglue-matcher": "component of omniglue",
    "roma-indoor": "same architecture and speed as roma (other weights)",
    "ripepp-aachen": "same architecture and speed as ripepp (other weights)",
    "ripepp-scared": "same architecture and speed as ripepp (other weights)",
    "superglue": "the matcher inside superpoint-superglue",
    "raco": "the detector of raco-aliked",
}
NOT_A_ROW_PREFIXES = ("raco-",)  # raco-<extractor>[-lightglue]: RaCo keypoints with another descriptor (raco-aliked is a row)


def _speed_table():
    spec = importlib.util.spec_from_file_location("speed_table", SPEED / "speed_table.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _rows() -> set[str]:
    table = _speed_table()
    rows = set(table.EXTRACTORS) | set(table.PAIR_LEVEL) | set(table.PIPELINES) | {"romav2", "flexglue"}
    return rows


def test_every_registered_model_has_a_speed_row():
    rows = _rows()
    missing = [
        name for name in flashim.available_models()
        if name not in rows and name not in NOT_A_ROW
        and not (name.startswith(NOT_A_ROW_PREFIXES) and name != "raco-aliked" and name != "raco-aliked-lightglue")
    ]
    assert not missing, (
        f"models without a latency row: {missing}. Add them to benchmarks/speed/speed_table_models.sh (or "
        "speed_table.sh) and to the lists in benchmarks/speed/speed_table.py, run them, and update docs/benchmarks.md; "
        "or list them in NOT_A_ROW with the row that stands for them."
    )


def test_every_listed_row_is_scheduled_by_a_speed_script():
    scripts = (SPEED / "speed_table.sh").read_text() + (SPEED / "speed_table_models.sh").read_text()
    tokens = set(re.findall(r"[A-Za-z0-9_+-]+", scripts))
    table = _speed_table()
    for name in sorted(_rows() - {"romav2"}):  # romav2 has its own function in speed_table.sh
        base = name.removesuffix("-lightglue") if name in table.LIGHTGLUE else name
        assert base in tokens, f"{name} is in speed_table.py but no speed script runs it"


def test_not_a_row_names_exist():
    registered = set(flashim.available_models())
    stale = [name for name in NOT_A_ROW if name not in registered]
    assert not stale, f"NOT_A_ROW lists models that are not registered anymore: {stale}"
