"""Repository rules that used to be prose in AGENTS.md, checked statically (CPU, no model is built).

* import layering (``platforms < quant.formats < ops < backends``, ``layers < quant.passes < models``),
* every ``models/<m>/weights.py`` is covered by ``tests/unit/test_weights.py``,
* the code stays importable on Python 3.10 (JetPack 6),
* TileLang kernel files do not use ``from __future__ import annotations``,
* every path AGENTS.md and the skills mention exists.

A failure message says what to change; if a rule is wrong for a good reason, change it here and in AGENTS.md together.
"""

from __future__ import annotations

import ast
import re
from importlib.util import resolve_name
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
PACKAGE = ROOT / "flashim"


def _module_name(path: Path) -> str:
    parts = list(path.relative_to(ROOT).with_suffix("").parts)
    if parts[-1] == "__init__":
        parts.pop()
    return ".".join(parts)


def _sources() -> list[tuple[str, Path, ast.Module]]:
    return [(_module_name(p), p, ast.parse(p.read_text())) for p in sorted(PACKAGE.rglob("*.py"))]


def _imports(tree: ast.Module, path: Path) -> list[tuple[int, str]]:
    """``(line, absolute dotted name)`` of every import in the file, function-level ones included."""
    found = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            found += [(node.lineno, alias.name) for alias in node.names]
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            if node.level:
                package = ".".join(path.parent.relative_to(ROOT).parts)
                module = resolve_name("." * node.level + module, package)
            found.append((node.lineno, module))
            found += [(node.lineno, f"{module}.{alias.name}") for alias in node.names]
    return found


def _area(module: str) -> str | None:
    parts = module.split(".")
    if parts[0] != "flashim" or len(parts) < 2:
        return None
    if parts[1] == "quant" and len(parts) > 2 and parts[2] in ("formats", "passes", "methods", "tensor"):
        return f"quant.{parts[2]}"
    return parts[1]


# area -> areas its modules must not import. Pairs not listed are unconstrained (``build`` / ``registry`` / ``engines``
# sit above the models and are not constrained either).
FORBIDDEN = {
    "platforms": {"ops", "backends", "layers", "models", "quant.formats", "quant.methods", "quant.passes", "quant"},
    "quant.formats": {"ops", "backends", "layers", "models", "quant.methods", "quant.passes", "quant"},
    "quant.tensor": {"ops", "backends", "layers", "models", "quant.methods", "quant.passes", "quant"},
    "ops": {"layers", "models", "quant.methods", "quant.passes", "quant"},
    "backends": {"layers", "models", "quant.passes", "build", "registry"},
    "layers": {"backends", "models", "quant.passes"},
    "quant.passes": {"backends", "models"},
    "models": {"backends", "build", "model_info", "registry"},
}
# The registry loads the backends lazily, that is its job: the only modules of ``ops`` that may import ``backends``.
OPS_MAY_LOAD_BACKENDS = {"flashim.ops", "flashim.ops._registry"}


def test_import_layering():
    violations = []
    for module, path, tree in _sources():
        area = _area(module)
        forbidden = FORBIDDEN.get(area, set())
        for line, name in _imports(tree, path):
            target = _area(name)
            if target is None or target == area or target not in forbidden:
                continue
            if area == "ops" and target == "backends" and module in OPS_MAY_LOAD_BACKENDS:
                continue
            if area == "models" and target == "build" and (path.stem == "spec" or path.stem.endswith("_spec")):
                continue
            violations.append(f"{path.relative_to(ROOT)}:{line} imports {name} ({area} must not import {target})")
    assert not violations, (
        "Dependencies point one way: platforms < quant.formats/tensor < ops < backends; layers < quant.passes < models "
        "(models and layers never import a backend: they call ops.*). Move the shared helper down to flashim/utils.py "
        "or into ops/, do not import upward:\n" + "\n".join(sorted(violations))
    )


def test_every_weights_module_is_in_the_weights_test():
    from tests.unit.test_weights import WEIGHT_MODULES

    on_disk = {p.parent.name for p in (PACKAGE / "models").glob("*/weights.py")}
    assert on_disk == set(WEIGHT_MODULES), (
        f"flashim/models/<m>/weights.py and WEIGHT_MODULES in tests/unit/test_weights.py differ: "
        f"missing from the test {sorted(on_disk - set(WEIGHT_MODULES))}, stale in the test "
        f"{sorted(set(WEIGHT_MODULES) - on_disk)}"
    )


# Names that only exist from Python 3.11 on; JetPack 6 ships 3.10, so they must be guarded (TYPE_CHECKING / try-except).
PY311 = {"tomllib", "Self", "StrEnum", "LiteralString", "Never", "assert_never", "ExceptionGroup", "getLevelNamesMapping"}


def _guarded_lines(tree: ast.Module) -> set[int]:
    guarded = set()
    for node in ast.walk(tree):
        is_try = isinstance(node, ast.Try)
        is_type_checking = isinstance(node, ast.If) and "TYPE_CHECKING" in ast.unparse(node.test)
        for stmt in node.body if (is_try or is_type_checking) else ():
            guarded.update(range(stmt.lineno, (stmt.end_lineno or stmt.lineno) + 1))
    return guarded


def test_python_310_compatible():
    try_star = getattr(ast, "TryStar", ())  # ast.TryStar is 3.11+: on 3.10 the parser rejects except* before this runs
    problems = []
    for _, path, tree in _sources():
        guarded = _guarded_lines(tree)
        for node in ast.walk(tree):
            if isinstance(node, try_star):
                problems.append(f"{path.relative_to(ROOT)}:{node.lineno} uses except* (3.11+)")
            names = []
            if isinstance(node, ast.Import):
                names = [alias.name.split(".")[0] for alias in node.names]
            elif isinstance(node, ast.ImportFrom) and node.level == 0:
                names = [alias.name for alias in node.names] + [(node.module or "").split(".")[0]]
            elif isinstance(node, ast.Attribute):
                names = [node.attr]
            hit = PY311.intersection(names)
            if hit and node.lineno not in guarded:
                problems.append(f"{path.relative_to(ROOT)}:{node.lineno} uses {sorted(hit)} (3.11+) unguarded")
    assert not problems, (
        "Keep flashim importable on Python 3.10 (Jetson / JetPack 6): import 3.11+ names under `if TYPE_CHECKING:` or "
        "in a try/except with the backport (see flashim/config.py load_file):\n" + "\n".join(problems)
    )


def test_tilelang_files_do_not_use_future_annotations():
    offenders = []
    for _, path, tree in _sources():
        if "backends/tilelang" not in path.as_posix():
            continue
        imports_tilelang = any(name.split(".")[0] == "tilelang" for _, name in _imports(tree, path))
        has_future = any(
            isinstance(node, ast.ImportFrom) and node.module == "__future__"
            and any(alias.name == "annotations" for alias in node.names)
            for node in tree.body
        )
        if imports_tilelang and has_future:
            offenders.append(str(path.relative_to(ROOT)))
    assert not offenders, (
        "TileLang evaluates the T.Tensor annotations: remove `from __future__ import annotations` from "
        + ", ".join(offenders)
    )


# -- documentation ---------------------------------------------------------------------------------------------

DOCS = ["AGENTS.md", *sorted(str(p.relative_to(ROOT)) for p in ROOT.glob(".agents/skills/*/SKILL.md")
                                                         if p.parent.name not in {"kernelwiki", "ncu-report-skill"})]
UNTRACKED = {"data", "results", ".venv", "~", "kernelwiki"}  # gitignored or machine-local (third-party skill)
FILE_SUFFIXES = (".py", ".md", ".sh", ".toml", ".json", ".txt", ".yaml", ".yml", ".cu")
TOKEN = re.compile(r"`([^`\s]+)`")


def _path_tokens(text: str):
    for token in TOKEN.findall(text):
        token = token.rstrip(",.;:)")
        if any(ch in token for ch in "<>*{}$=()[]|") or token.startswith(("/", "-", ".venv", "http")):
            continue
        if "::" in token:
            token = token.split("::")[0]
        if "/" in token or token.endswith(FILE_SUFFIXES):
            if not re.fullmatch(r"[\w./+-]+", token) or token.startswith(("~", "$")):
                continue
            yield token


def _exists(token: str, all_paths: set[str]) -> bool:
    clean = token.rstrip("/")
    if UNTRACKED & set(clean.split("/")):
        return True
    for base in (ROOT, PACKAGE):  # docs write `ops/x.py` for flashim/ops/x.py
        if (base / clean).exists():
            return True
    # relative to a skill / doc, or a bare name (`csrc/`, `references/x.md`, `build.py`): any path that ends with it
    return any(p == clean or p.endswith("/" + clean) for p in all_paths)


@pytest.mark.parametrize("doc", DOCS)
def test_documented_paths_exist(doc):
    path = ROOT / doc
    if not path.exists():
        pytest.skip(f"{doc} not present")
    skip_dirs = {".venv", ".git", "results", "data", "__pycache__", ".ruff_cache", ".pytest_cache"}
    all_paths = {
        p.relative_to(ROOT).as_posix()
        for p in ROOT.rglob("*")
        if not (skip_dirs & set(p.relative_to(ROOT).parts))
    }
    missing = sorted({t for t in _path_tokens(path.read_text()) if not _exists(t, all_paths)})
    assert not missing, f"{doc} mentions paths that do not exist (renamed or removed?): {missing}"
