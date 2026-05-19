"""Lightweight FinChain verifier utilities.

The official FinChain project introduces ChainEval for step-level trace
alignment. This module intentionally starts with the deterministic substrate
needed by local training loops: parseability and final-answer correctness.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

from finpost.data.finchain import score_finchain_answer, try_parse_finchain_final_answer


@dataclass(frozen=True)
class FinChainGrade:
    """Result of grading one FinChain model generation."""

    parsed_answer: str | None
    parse_success: bool
    final_answer_correct: bool
    reason: str


def grade_finchain_generation(generation: str, *, gold_answer: str) -> FinChainGrade:
    """Grade one model generation against a FinChain final answer."""
    parsed = try_parse_finchain_final_answer(generation)
    if parsed is None:
        return FinChainGrade(
            parsed_answer=None,
            parse_success=False,
            final_answer_correct=False,
            reason="parse_failure",
        )

    correct = score_finchain_answer(parsed, gold_answer)
    return FinChainGrade(
        parsed_answer=parsed,
        parse_success=True,
        final_answer_correct=correct,
        reason="correct" if correct else "answer_mismatch",
    )


def summarize_finchain_grades(grades: Iterable[FinChainGrade]) -> dict[str, float]:
    """Aggregate parse and final-answer accuracy for a FinChain eval run."""
    rows = list(grades)
    if not rows:
        return {
            "n": 0.0,
            "parse_success_rate": 0.0,
            "final_answer_accuracy": 0.0,
        }
    return {
        "n": float(len(rows)),
        "parse_success_rate": sum(g.parse_success for g in rows) / len(rows),
        "final_answer_accuracy": sum(g.final_answer_correct for g in rows) / len(rows),
    }
