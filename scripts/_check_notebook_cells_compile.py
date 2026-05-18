"""Compile-check every code cell in the runpod ablation notebook.

Strips Jupyter magic syntax (lines starting with ! or %, with backslash
line continuations) before compiling — those aren't valid Python; they
get translated by IPython at execution time.

Run: python scripts/_check_notebook_cells_compile.py
"""

from __future__ import annotations

import json
from pathlib import Path

NOTEBOOK = Path("notebooks/sft_phase1_runpod_ablation_2000.ipynb")


def strip_magics(src: str) -> str:
    out_lines: list[str] = []
    in_continuation = False
    for line in src.split("\n"):
        if in_continuation:
            ends_continued = line.rstrip().endswith("\\")
            if not ends_continued:
                in_continuation = False
            continue
        stripped = line.lstrip()
        if stripped.startswith("!") or stripped.startswith("%"):
            if line.rstrip().endswith("\\"):
                in_continuation = True
            continue
        out_lines.append(line)
    return "\n".join(out_lines)


def main() -> None:
    nb = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    errors: list[tuple[int, str]] = []
    checked = 0
    for i, cell in enumerate(nb["cells"]):
        if cell["cell_type"] != "code":
            continue
        checked += 1
        cleaned = strip_magics("".join(cell["source"]))
        try:
            compile(cleaned, f"<cell-{i}>", "exec")
        except SyntaxError as exc:
            errors.append((i, str(exc)))

    if errors:
        for i, msg in errors:
            print(f"CELL {i}: {msg}")
        raise SystemExit(1)

    print(f"All {checked} code cells compile cleanly.")


if __name__ == "__main__":
    main()
