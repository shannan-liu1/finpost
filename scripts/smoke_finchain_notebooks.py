"""Static smoke checks for FinChain GPU notebooks."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

NOTEBOOKS = {
    "sft": Path("notebooks/01_sft_ablation.ipynb"),
    "dpo": Path("notebooks/02_dpo.ipynb"),
    "grpo": Path("notebooks/03_grpo_rlvr.ipynb"),
    "opd_gkd": Path("notebooks/04_opd_gkd.ipynb"),
}

SELECTED_SFT = "results/checkpoints/qwen25-1p5b-finchain-v2-trl-sft-2gpu/selected"

COMMON_REQUIRED_FRAGMENTS = [
    "data/finchain_v2_template_disjoint",
    '"batch_size_finchain":',
    "--finchain-split validation --n 870",
    "--finchain-split test --n 870",
    "--enable-chaineval",
    '"selection_split": "validation"',
    '"final_eval_split": "test"',
    '"test_touched_before_selection": False',
    "step_recall",
    "step_precision",
    "step_f1",
    "json.loads",
    "accuracy_summary.json",
]

METHOD_REQUIRED_FRAGMENTS = {
    "sft": [
        "configs/finchain/sft/qwen25_1_5b_sft.yaml",
        "scripts/train_finchain_trl_sft.py",
        "snapshot_download",
        "shannan-liu1/qwen25-1p5b-finchain-v2-sft-selected",
    ],
    "dpo": [
        SELECTED_SFT,
        "configs/finchain/dpo/finchain_qwen25_1_5b.yaml",
        "scripts/build_dpo_pairs.py",
        "scripts/train_finchain_trl_dpo.py",
        "DPOConfig OK",
    ],
    "opd_gkd": [
        SELECTED_SFT,
        "configs/finchain/gkd/finchain_qwen25_1_5b",
        "finpost.training.gkd_train",
        "Qwen/Qwen2.5-7B-Instruct",
        "GKDConfig OK",
    ],
    "grpo": [
        SELECTED_SFT,
        "configs/finchain/grpo/qwen25_1_5b_grpo.yaml",
        "scripts/train_finchain_trl_grpo.py",
        "GRPOConfig OK",
    ],
}


def notebook_source(path: Path) -> tuple[dict[str, Any], str]:
    notebook = json.loads(path.read_text(encoding="utf-8"))
    text = "\n".join("".join(cell.get("source", [])) for cell in notebook["cells"])
    return notebook, text


def strip_notebook_magics(text: str) -> str:
    lines: list[str] = []
    skipping_continuation = False
    for line in text.splitlines():
        if skipping_continuation:
            skipping_continuation = line.rstrip().endswith("\\")
            continue
        if line.lstrip().startswith(("!", "%")):
            skipping_continuation = line.rstrip().endswith("\\")
            continue
        lines.append(line)
    return "\n".join(lines)


def compile_code_cells(path: Path, notebook: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    for index, cell in enumerate(notebook["cells"]):
        if cell.get("cell_type") != "code":
            continue
        cleaned = strip_notebook_magics("".join(cell.get("source", [])))
        try:
            compile(cleaned, f"{path}:cell-{index}", "exec")
        except SyntaxError as exc:
            errors.append(f"cell {index}: {exc}")
    return errors


def smoke_notebook(name: str, path: Path) -> dict[str, Any]:
    notebook, text = notebook_source(path)
    compile_errors = compile_code_cells(path, notebook)
    required_fragments = [
        *COMMON_REQUIRED_FRAGMENTS,
        *METHOD_REQUIRED_FRAGMENTS.get(name, []),
    ]
    missing = [fragment for fragment in required_fragments if fragment not in text]
    selection_pos = text.find('"selection_split": "validation"')
    test_eval_pos = text.find("--finchain-split test --n 870")
    return {
        "path": str(path),
        "compiles": not compile_errors,
        "compile_errors": compile_errors,
        "required_fragments_present": not missing,
        "missing_fragments": missing,
        "test_split_before_selection_guard": (
            selection_pos < test_eval_pos if test_eval_pos >= 0 else False
        ),
    }


def smoke_all_notebooks() -> dict[str, dict[str, Any]]:
    return {name: smoke_notebook(name, path) for name, path in NOTEBOOKS.items()}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    results = smoke_all_notebooks()
    report = {
        "ok": all(
            item["compiles"]
            and item["required_fragments_present"]
            and item["test_split_before_selection_guard"]
            for item in results.values()
        ),
        "notebooks": results,
    }
    text = json.dumps(report, indent=2, sort_keys=True)
    print(text)
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text + "\n", encoding="utf-8")
    if not report["ok"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
