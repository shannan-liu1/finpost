from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def test_finchain_verifier_accepts_matching_final_answer() -> None:
    from finpost.evals.finchain_metrics import grade_finchain_generation

    grade = grade_finchain_generation(
        "Reasoning omitted. Final answer: 42.00",
        gold_answer="42",
    )

    assert grade.parse_success
    assert grade.final_answer_correct
    assert grade.reason == "correct"


def test_finchain_verifier_rejects_unparseable_output() -> None:
    from finpost.evals.finchain_metrics import grade_finchain_generation

    grade = grade_finchain_generation("I cannot determine it.", gold_answer="42")

    assert not grade.parse_success
    assert not grade.final_answer_correct
    assert grade.reason == "parse_failure"
