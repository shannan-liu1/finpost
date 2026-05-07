"""Tests for the MATH loader.

Network-free tests covering the three pure functions in
``finpost.data.math_dataset``: the boxed-answer parser, the level
parser, and the category normalizer. End-to-end load is verified via
the CLI.
"""

from __future__ import annotations

import pytest

from finpost.data.math_dataset import (
    normalize_math_category,
    parse_math_difficulty,
    parse_math_final_answer,
)


# -----------------------------------------------------------------------------
# parse_math_final_answer — \boxed{...} extraction with brace counting
# -----------------------------------------------------------------------------


def test_parses_simple_boxed_answer() -> None:
    """Simple integer answer wrapped in \\boxed."""
    solution = r"The answer is $\boxed{42}$."
    assert parse_math_final_answer(solution) == "42"


def test_parses_nested_braces_in_boxed() -> None:
    """The motivating case for brace-counting: \\frac contains nested braces."""
    solution = r"Therefore the answer is $\boxed{\frac{1}{2}}$."
    assert parse_math_final_answer(solution) == r"\frac{1}{2}"


def test_parses_deeply_nested_braces() -> None:
    """Three levels of nesting (a fraction with a subscript)."""
    solution = r"$\boxed{\frac{x_{1}}{y_{2}}}$"
    assert parse_math_final_answer(solution) == r"\frac{x_{1}}{y_{2}}"


def test_takes_last_boxed_when_multiple() -> None:
    """If \\boxed appears more than once, the last one is the final answer."""
    solution = r"First we get $\boxed{3}$, then we adjust to $\boxed{4}$."
    assert parse_math_final_answer(solution) == "4"


def test_parses_boxed_with_latex_expression() -> None:
    """Answer can be an arbitrary LaTeX expression, not just a number."""
    solution = r"So $\boxed{2\sqrt{3}}$."
    assert parse_math_final_answer(solution) == r"2\sqrt{3}"


def test_missing_boxed_raises() -> None:
    """No \\boxed marker is a data-integrity error."""
    with pytest.raises(ValueError, match="no .*answer extractable"):
        parse_math_final_answer("The answer is 42.")


def test_unbalanced_braces_raises() -> None:
    """A \\boxed{ without matching close brace AND no no-brace form is unparseable."""
    with pytest.raises(ValueError, match="no .*answer extractable"):
        parse_math_final_answer(r"$\boxed{\frac{1}{2}$")


def test_parses_fbox_fallback() -> None:
    """\\fbox{...} is the alternative wrapper used in some solutions."""
    solution = r"The answer is $\fbox{42}$."
    assert parse_math_final_answer(solution) == "42"


def test_parses_no_brace_boxed_single_char() -> None:
    """The motivating real example: '... our answer is $\\boxed 2$.'"""
    solution = r"... we find x = 2. So our answer is $\boxed 2$."
    assert parse_math_final_answer(solution) == "2"


def test_parses_no_brace_boxed_multi_char() -> None:
    """No-brace form should extend to the next math-mode delimiter."""
    solution = r"The total = $\boxed 42$."
    assert parse_math_final_answer(solution) == "42"


def test_no_brace_takes_last_when_brace_form_also_present_earlier() -> None:
    """If a brace form precedes a no-brace form, the no-brace one wins as the last \\boxed."""
    solution = r"Intermediate $\boxed{1}$, then refined to $\boxed 2$."
    assert parse_math_final_answer(solution) == "2"


def test_brace_form_takes_last_when_no_brace_form_precedes() -> None:
    """If a no-brace form precedes a brace form, the brace one wins."""
    solution = r"First $\boxed 1$, then refined to $\boxed{2}$."
    assert parse_math_final_answer(solution) == "2"


# -----------------------------------------------------------------------------
# parse_math_difficulty — strict 'Level N' parsing
# -----------------------------------------------------------------------------


def test_parses_each_valid_level() -> None:
    for n in (1, 2, 3, 4, 5):
        assert parse_math_difficulty(f"Level {n}") == n


def test_rejects_lowercase_prefix() -> None:
    """Strict on the prefix — drift in the upstream format should fail loudly."""
    with pytest.raises(ValueError, match="Unexpected MATH level format"):
        parse_math_difficulty("level 3")


def test_rejects_missing_prefix() -> None:
    with pytest.raises(ValueError, match="Unexpected MATH level format"):
        parse_math_difficulty("3")


def test_rejects_non_integer_non_question_level() -> None:
    """Non-integer values that aren't the known '?' marker still raise."""
    with pytest.raises(ValueError, match="neither an integer"):
        parse_math_difficulty("Level easy")


def test_question_mark_returns_none() -> None:
    """'Level ?' is the dataset's known unknown-marker; return None, do not raise."""
    assert parse_math_difficulty("Level ?") is None


def test_rejects_out_of_range_level() -> None:
    with pytest.raises(ValueError, match="out of expected range"):
        parse_math_difficulty("Level 6")


# -----------------------------------------------------------------------------
# normalize_math_category — snake_case slug from presentation string
# -----------------------------------------------------------------------------


def test_normalizes_simple_single_word() -> None:
    assert normalize_math_category("Algebra") == "algebra"


def test_normalizes_two_word_category() -> None:
    assert normalize_math_category("Number Theory") == "number_theory"


def test_normalizes_ampersand_category() -> None:
    """The 'Counting & Probability' case is the reason this function exists."""
    assert normalize_math_category("Counting & Probability") == "counting_and_probability"


def test_normalizes_intermediate_algebra() -> None:
    assert normalize_math_category("Intermediate Algebra") == "intermediate_algebra"
