"""Behavior tests for the lightweight FinChain verifier."""

from __future__ import annotations


def test_finchain_grade_accepts_numeric_equivalent_answer() -> None:
    """Currency and thousands separators should not break final-answer scoring."""
    from finpost.evals.finchain_metrics import grade_finchain_generation

    grade = grade_finchain_generation("Work\nFinal Answer: $1,210.0", gold_answer="1210.00")

    assert grade.parse_success is True
    assert grade.final_answer_correct is True
    assert grade.reason == "correct"


def test_finchain_grade_rejects_corrupted_final_answer() -> None:
    """Wrong numeric answers should fail with a useful reason code."""
    from finpost.evals.finchain_metrics import grade_finchain_generation

    grade = grade_finchain_generation("Work\nFinal Answer: 999", gold_answer="1210")

    assert grade.parse_success is True
    assert grade.final_answer_correct is False
    assert grade.reason == "answer_mismatch"


def test_finchain_grade_reports_parse_failures_separately() -> None:
    """Parse failures are distinct from wrong parsed answers."""
    from finpost.evals.finchain_metrics import grade_finchain_generation

    grade = grade_finchain_generation("No final marker here", gold_answer="1210")

    assert grade.parse_success is False
    assert grade.final_answer_correct is False
    assert grade.reason == "parse_failure"
