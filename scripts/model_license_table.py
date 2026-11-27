"""Regenerate docs/licenses.md with `PYTHONPATH=. python scripts/model_license_table.py`."""

from pathlib import Path

from flashim import model_info
from flashim.registry import SPECS

DOCUMENT = Path(__file__).resolve().parents[1] / "docs" / "licenses.md"
TABLE_HEADING = "## Registered models\n\n"


def render() -> str:
    rows = ["| Model | Model / upstream code | Weights | `commercial_ok` |", "| --- | --- | --- | --- |"]
    for name in sorted(SPECS):
        info = model_info(name)
        code = ", ".join(info.code_licenses)
        weights = ", ".join(info.weights_licenses) or "No weights"
        rows.append(f"| `{name}` | {code} | {weights} | {str(info.commercial_ok).lower()} |")
    return "\n".join(rows) + "\n"


if __name__ == "__main__":
    introduction, _ = DOCUMENT.read_text(encoding="utf-8").split(TABLE_HEADING)
    DOCUMENT.write_text(introduction + TABLE_HEADING + render(), encoding="utf-8")
