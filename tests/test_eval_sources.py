"""Tests for the eval source registry.

Tests the EvalSource dataclass contract, answer extractors (GSM8K and MATH
with positive and negative cases), and the REGISTRY lookup table.
"""

from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

from finpost.evals.sources import REGISTRY


# =============================================================================
# REGISTRY lookup and basic contract
# =============================================================================


def test_registry_contains_gsm8k_and_math() -> None:
    """The REGISTRY dict has exactly the two Phase 1 sources."""
    assert sorted(REGISTRY.keys()) == ["gsm8k", "math"]


def test_eval_source_is_frozen() -> None:
    """Registry entries must be immutable so callers cannot mutate the registry at runtime."""
    gsm8k = REGISTRY["gsm8k"]
    with pytest.raises(FrozenInstanceError):
        gsm8k.name = "modified"  # type: ignore[misc]


# =============================================================================
# GSM8K answer extractor
# =============================================================================


def test_gsm8k_extracts_simple_answer() -> None:
    """Extract the final numeric answer from a GSM8K-format response."""
    gsm8k = REGISTRY["gsm8k"]
    result = gsm8k.extract_answer("blah blah\n#### 42")
    assert result == "42"


def test_gsm8k_extracts_last_marker() -> None:
    """When '####' appears multiple times, use the last one."""
    gsm8k = REGISTRY["gsm8k"]
    result = gsm8k.extract_answer("marker #### 1\nmore text\n#### 7")
    assert result == "7"


def test_gsm8k_strips_whitespace() -> None:
    """Surrounding whitespace is normalized."""
    gsm8k = REGISTRY["gsm8k"]
    result = gsm8k.extract_answer("#### \t 42 \n")
    assert result == "42"


def test_gsm8k_strips_leading_dollar_sign() -> None:
    """Currency markers are removed."""
    gsm8k = REGISTRY["gsm8k"]
    result = gsm8k.extract_answer("#### $100")
    assert result == "100"


def test_gsm8k_strips_trailing_period() -> None:
    """Trailing punctuation is removed."""
    gsm8k = REGISTRY["gsm8k"]
    result = gsm8k.extract_answer("#### 42.")
    assert result == "42"


def test_gsm8k_strips_commas() -> None:
    """Thousands separators are removed."""
    gsm8k = REGISTRY["gsm8k"]
    result = gsm8k.extract_answer("#### 1,200")
    assert result == "1200"


def test_gsm8k_composite_normalization() -> None:
    """The spec example: $1,234. should normalize to 1234 (strip $, then trailing ., then commas)."""
    result = REGISTRY["gsm8k"].extract_answer("answer is\n#### $1,234.")
    assert result == "1234"


def test_gsm8k_handles_negative() -> None:
    """Negative answers preserve their sign."""
    gsm8k = REGISTRY["gsm8k"]
    result = gsm8k.extract_answer("#### -5")
    assert result == "-5"


def test_gsm8k_returns_none_on_no_marker() -> None:
    """Return None if no '####' marker exists."""
    gsm8k = REGISTRY["gsm8k"]
    result = gsm8k.extract_answer("no marker here")
    assert result is None


def test_gsm8k_returns_none_on_empty_after_marker() -> None:
    """Return None if '####' is followed only by whitespace."""
    gsm8k = REGISTRY["gsm8k"]
    result = gsm8k.extract_answer("#### \n")
    assert result is None


def test_gsm8k_returns_none_on_marker_with_only_punctuation() -> None:
    """Return None if nothing remains after stripping punctuation/whitespace."""
    gsm8k = REGISTRY["gsm8k"]
    result = gsm8k.extract_answer("#### ,.")
    assert result is None


# =============================================================================
# MATH answer extractor
# =============================================================================


def test_math_extracts_simple_boxed_answer() -> None:
    """Extract the answer from a simple \\boxed{N} form."""
    math = REGISTRY["math"]
    result = math.extract_answer(r"answer is \boxed{42}")
    assert result == "42"


def test_math_extracts_nested_braces() -> None:
    """Handle nested braces in LaTeX commands like \\frac{1}{2}."""
    math = REGISTRY["math"]
    result = math.extract_answer(r"answer is \boxed{\frac{1}{2}}")
    assert result == r"\frac{1}{2}"


def test_math_extracts_deeply_nested_braces() -> None:
    """Handle multiple levels of nesting (e.g., \\frac with subscripts)."""
    math = REGISTRY["math"]
    result = math.extract_answer(r"$\boxed{\frac{x_{1}}{y_{2}}}$")
    assert result == r"\frac{x_{1}}{y_{2}}"


def test_math_extracts_last_boxed_when_multiple() -> None:
    """When \\boxed appears multiple times, use the last one."""
    math = REGISTRY["math"]
    result = math.extract_answer(
        r"First we get $\boxed{3}$, then we adjust to $\boxed{4}$."
    )
    assert result == "4"


def test_math_strips_whitespace_inside_boxed() -> None:
    """Leading/trailing whitespace inside the braces is stripped."""
    math = REGISTRY["math"]
    result = math.extract_answer(r"\boxed{  42  }")
    assert result == "42"


def test_math_strips_wrapping_dollar_signs() -> None:
    """Dollar signs wrapping the answer are stripped if present."""
    math = REGISTRY["math"]
    result = math.extract_answer(r"\boxed{$42$}")
    assert result == "42"


def test_math_returns_none_on_no_boxed() -> None:
    """Return None if no \\boxed marker exists."""
    math = REGISTRY["math"]
    result = math.extract_answer("no boxed here")
    assert result is None


def test_math_returns_none_on_unmatched_braces() -> None:
    """Return None if \\boxed braces are unbalanced."""
    math = REGISTRY["math"]
    result = math.extract_answer(r"\boxed{unclosed")
    assert result is None


def test_math_returns_none_when_boxed_has_no_opening_brace() -> None:
    """\\boxed appearing without a following { should return None, not crash."""
    result = REGISTRY["math"].extract_answer(r"the answer is \boxed and then more text")
    assert result is None


def test_math_extracts_with_literal_braces_inside() -> None:
    """Literal { and } inside \\boxed{} should be preserved in the extracted answer."""
    result = REGISTRY["math"].extract_answer(r"\boxed{\{a, b\}}")
    assert result == r"\{a, b\}"


def test_math_returns_none_on_empty_boxed() -> None:
    """Return None if the boxed content is empty after stripping whitespace."""
    math = REGISTRY["math"]
    result = math.extract_answer(r"\boxed{   }")
    assert result is None


# =============================================================================
# Score function
# =============================================================================


def test_gsm8k_score_exact_match() -> None:
    """Exact string match returns True."""
    gsm8k = REGISTRY["gsm8k"]
    assert gsm8k.score("42", "42") is True


def test_gsm8k_score_mismatch() -> None:
    """String mismatch returns False."""
    gsm8k = REGISTRY["gsm8k"]
    assert gsm8k.score("42", "43") is False


def test_gsm8k_score_none_is_always_wrong() -> None:
    """Predicted None (parse failure) is always incorrect."""
    gsm8k = REGISTRY["gsm8k"]
    assert gsm8k.score(None, "42") is False


def test_math_score_exact_match() -> None:
    """Exact string match returns True."""
    math = REGISTRY["math"]
    assert math.score(r"\frac{1}{2}", r"\frac{1}{2}") is True


def test_math_score_mismatch() -> None:
    """String mismatch returns False."""
    math = REGISTRY["math"]
    assert math.score(r"\frac{1}{2}", r"\frac{2}{1}") is False


def test_math_score_none_is_always_wrong() -> None:
    """Predicted None (parse failure) is always incorrect."""
    math = REGISTRY["math"]
    assert math.score(None, "42") is False


# =============================================================================
# MATH LaTeX normalization via _strip_string (Bug 1)
# =============================================================================


def test_math_score_dfrac_equals_frac() -> None:
    r"""'\dfrac{1}{2}' and '\frac{1}{2}' should score True after normalization."""
    assert REGISTRY["math"].score("\\frac{1}{2}", "\\dfrac{1}{2}") is True


def test_math_score_tfrac_equals_frac() -> None:
    r"""'\tfrac{1}{2}' and '\frac{1}{2}' should score True after normalization."""
    assert REGISTRY["math"].score("\\frac{1}{2}", "\\tfrac{1}{2}") is True


def test_math_score_left_right_removed() -> None:
    r"""'\left(1, 2\right)' and '(1, 2)' should score True — size hints stripped."""
    assert REGISTRY["math"].score("(1, 2)", "\\left(1, 2\\right)") is True


def test_math_score_numeric_via_float_fallback() -> None:
    """'42' vs '42.0' scores True via numeric fallback (not _strip_string)."""
    assert REGISTRY["math"].score("42", "42.0") is True


def test_math_score_different_fractions_still_false() -> None:
    r"""'\frac{1}{2}' vs '\frac{2}{4}': _strip_string won't reduce these — False."""
    assert REGISTRY["math"].score("\\frac{1}{2}", "\\frac{2}{4}") is False


# =============================================================================
# Unicode minus sign normalization (Bug 4)
# =============================================================================


def test_gsm8k_extract_unicode_minus() -> None:
    """U+2212 (Unicode minus) in '#### −42' should extract as ASCII '-42'."""
    result = REGISTRY["gsm8k"].extract_answer("#### −42")
    assert result == "-42"


def test_math_extract_unicode_minus() -> None:
    r"""U+2212 in '\boxed{−42}' should extract as ASCII '-42'."""
    result = REGISTRY["math"].extract_answer("\\boxed{−42}")
    assert result == "-42"


# =============================================================================
# GSM8K numeric-equality fallback (Bug 2)
# =============================================================================


def test_gsm8k_score_decimal_zero_equals_integer() -> None:
    """Model emitting '42.0' when gold is '42' should score True."""
    assert REGISTRY["gsm8k"].score("42.0", "42") is True


def test_gsm8k_score_scientific_notation() -> None:
    """'4.2e3' and '4200' are numerically equal — should score True."""
    assert REGISTRY["gsm8k"].score("4.2e3", "4200") is True


def test_gsm8k_score_numeric_mismatch_still_false() -> None:
    """'42.5' vs '42' are numerically different — still False."""
    assert REGISTRY["gsm8k"].score("42.5", "42") is False


def test_gsm8k_score_none_numeric_fallback() -> None:
    """None predicted is always False, even with numeric fallback present."""
    assert REGISTRY["gsm8k"].score(None, "42") is False


def test_gsm8k_score_non_numeric_strings() -> None:
    """Non-parseable strings fall through to string equality — still False."""
    assert REGISTRY["gsm8k"].score("forty-two", "42") is False


def test_math_score_decimal_zero_equals_integer() -> None:
    """MATH: '42.0' vs '42' should score True via numeric fallback."""
    assert REGISTRY["math"].score("42.0", "42") is True


def test_math_score_scientific_notation() -> None:
    """MATH: '4.2e3' vs '4200' numerically equal — should score True."""
    assert REGISTRY["math"].score("4.2e3", "4200") is True


def test_math_score_numeric_mismatch_still_false() -> None:
    """MATH: '42.5' vs '42' are numerically different — still False."""
    assert REGISTRY["math"].score("42.5", "42") is False


def test_math_score_none_numeric_fallback() -> None:
    """MATH: None predicted is always False."""
    assert REGISTRY["math"].score(None, "42") is False


def test_math_score_non_numeric_strings() -> None:
    """MATH: Non-parseable strings fall through to string equality — still False."""
    assert REGISTRY["math"].score("forty-two", "42") is False


# =============================================================================
# default_max_new_tokens
# =============================================================================


def test_gsm8k_default_max_new_tokens() -> None:
    """GSM8K has a default generation budget of 256 tokens."""
    gsm8k = REGISTRY["gsm8k"]
    assert gsm8k.default_max_new_tokens == 256


def test_math_default_max_new_tokens() -> None:
    """MATH has a default generation budget of 768 tokens."""
    math = REGISTRY["math"]
    assert math.default_max_new_tokens == 768
