"""Every Triton entry point has a real launch; device-only helpers compile inside their callers."""

import ast
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
MANIFEST = ROOT / "ci/triton.jsonl"


def test_triton_specialization_coverage():
    records = MANIFEST.read_text().splitlines()
    assert records and records == sorted(set(records))
    entries = [json.loads(line) for line in records]
    assert all(any(arches for _, arches in row["routes"]) for row in entries)
    recorded = {row["kernel"] for row in entries}
    dependencies = {}
    launches = set()
    for path in (ROOT / "flashim").rglob("*.py"):
        module = ".".join(path.relative_to(ROOT).with_suffix("").parts)
        tree = ast.parse(path.read_text())
        imports = {
            alias.asname or alias.name: f"{node.module}:{alias.name}"
            for node in ast.walk(tree)
            if isinstance(node, ast.ImportFrom)
            for alias in node.names
        }
        for node in ast.walk(tree):
            if isinstance(node, ast.Subscript) and isinstance(node.value, ast.Name):
                launches.add(imports.get(node.value.id, f"{module}:{node.value.id}"))
            if isinstance(node, ast.FunctionDef) and any(
                ast.unparse(decorator).split("(")[0] in ("triton.jit", "triton.autotune")
                for decorator in node.decorator_list
            ):
                dependencies[f"{module}:{node.name}"] = {
                    imports.get(ref.id, f"{module}:{ref.id}")
                    for ref in ast.walk(node)
                    if isinstance(ref, ast.Name) and isinstance(ref.ctx, ast.Load)
                }
    assert recorded <= dependencies.keys()
    assert launches & dependencies.keys() <= recorded
    covered = set(recorded)
    pending = list(recorded)
    while pending:
        helpers = (dependencies[pending.pop()] & dependencies.keys()) - covered
        covered.update(helpers)
        pending.extend(helpers)
    assert dependencies.keys() == covered


def test_recorded_routes_respect_architecture_and_call_arguments():
    pytest.importorskip("triton")
    from ci.compile_triton import eligible, simulated_platform
    from flashim.platforms import Feature

    entries = [json.loads(line) for line in MANIFEST.read_text().splitlines()]
    ampere, ada, blackwell = (simulated_platform(arch) for arch in (80, 89, 120))
    fused = [row for row in entries if row["kernel"].endswith(":_mxfp8_conv2d_scaled_kernel")]
    assert fused and not any(eligible(row, ada) for row in fused)
    assert all(eligible(row, blackwell) == blackwell.supports(Feature.MXFP8_SCALED_MM) for row in fused)
    producer = [row for row in entries if row["kernel"].endswith(":_leaky_relu_quantize_kernel")]
    assert {row["constants"]["QUANT_MODE"] for row in producer} == {1, 2}
    for row in producer:
        assert eligible(row, ampere) == (row["constants"]["QUANT_MODE"] == 1)
        assert eligible(row, ada)


def test_replay_cache_survives_worker_and_launch_order_changes(tmp_path):
    pytest.importorskip("triton")
    entries = [json.loads(line) for line in MANIFEST.read_text().splitlines()]
    rows = [
        next(row for row in entries if row["kernel"] == f"flashim.backends.triton.quantize:{name}")
        for name in ("_layernorm_quantize_kernel", "_quantize_static_kernel")
    ]
    manifest = tmp_path / "launches.jsonl"
    cache = tmp_path / "triton-cache"
    env = {**os.environ, "TRITON_CACHE_DIR": str(cache), "CUDA_VISIBLE_DEVICES": "", "PYTHONPATH": str(ROOT)}

    def replay(workers):
        manifest.write_text("".join(json.dumps(row) + "\n" for row in rows))
        subprocess.run(
            [
                sys.executable,
                ROOT / "ci/compile_triton.py",
                "--manifest",
                manifest,
                "--arch",
                "120",
                "--workers",
                str(workers),
            ],
            check=True,
            env=env,
        )
        return {path: path.stat().st_mtime_ns for path in cache.rglob("*.cubin")}

    cold = replay(1)
    assert len(cold) == 2
    rows.reverse()
    assert replay(2) == cold
    rows[0]["options"]["num_warps"] = 8 if rows[0]["options"]["num_warps"] == 4 else 4
    assert len(replay(1).keys() - cold.keys()) == 1
