"""Tests for the GSM8K loader.

Network-free tests covering the parser only. The end-to-end load is
verified through the CLI (PRD 0002 section 1.3 / acceptance criteria
1 and 2). Parser correctness is the part most likely to harbor subtle
bugs and benefits most from unit-level coverage.
"""

from __future__ import annotations

import pytest

from finpost.data.gsm8k import parse_gsm8k_final_answer


def test_parses_simple_integer_answer() -> None:
    """The most common case: chain of thought followed by '#### N'."""
    answer = "Janet sells 9 eggs at $2 each, so 9 * 2 = 18.\n#### 18"
    assert parse_gsm8k_final_answer(answer) == "18"


def test_strips_thousands_separator_commas() -> None:
    """Source uses '1,200' style; the canonical grader expects '1200'."""
    answer = "She earns 100 dollars per hour over 12 hours.\n#### 1,200"
    assert parse_gsm8k_final_answer(answer) == "1200"


def test_strips_trailing_punctuation() -> None:
    """Some examples end the sentinel line with a period or comma."""
    answer = "The total is forty-two.\n#### 42."
    assert parse_gsm8k_final_answer(answer) == "42"


def test_handles_negative_answer() -> None:
    """Negative answers exist; the leading minus must survive."""
    answer = "He owes 5 dollars and pays 3, leaving him at -2.\n#### -2"
    assert parse_gsm8k_final_answer(answer) == "-2"


def test_takes_last_sentinel_when_multiple() -> None:
    """If '####' appears earlier in the text, only the last one counts."""
    answer = "Discussion includes the marker #### but answer follows.\n#### 7"
    assert parse_gsm8k_final_answer(answer) == "7"


def test_missing_sentinel_raises() -> None:
    """Data-integrity error: any record without '####' is unparseable."""
    with pytest.raises(ValueError, match="missing '####' sentinel"):
        parse_gsm8k_final_answer("Just text, no sentinel.")


def test_empty_after_sentinel_raises() -> None:
    """Data-integrity error: '####' followed by nothing is unparseable."""
    with pytest.raises(ValueError, match="empty content after"):
        parse_gsm8k_final_answer("Some reasoning.\n####")


def test_only_punctuation_after_sentinel_raises() -> None:
    """Data-integrity error: '####' followed only by stripped chars."""
    with pytest.raises(ValueError, match="empty after cleaning"):
        parse_gsm8k_final_answer("Some reasoning.\n#### ,.")
