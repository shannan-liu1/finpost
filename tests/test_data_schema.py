"""Tests for the Phase 1 data schema.

These tests document the contract of ``Example`` by exercising it.
Each test corresponds to one promise the schema makes:

1. A minimal valid GSM8K example constructs.
2. A minimal valid MATH example with optional fields constructs.
3. Unknown ``source`` values are rejected (the Literal enforces the enum).
4. Empty required strings are rejected (min_length=1 catches them).
5. ``difficulty`` outside 1..5 is rejected (ge/le bounds).
6. Unknown field names at construction are rejected (extra='forbid').
7. Instances are immutable after construction (frozen=True).
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from finpost.data.schema import Example


def test_minimal_valid_gsm8k_example_constructs() -> None:
    ex = Example(
        id="gsm8k-train-0",
        source="gsm8k",
        prompt="A baker has 3 cakes.",
        response="So the baker has 3 cakes.\n#### 3",
        final_answer="3",
    )
    assert ex.source == "gsm8k"
    assert ex.difficulty is None
    assert ex.category is None


def test_minimal_valid_math_example_with_optional_fields() -> None:
    ex = Example(
        id="math-train-0",
        source="math",
        prompt="What is 2+2?",
        response="2+2=4. So the answer is $\\boxed{4}$.",
        final_answer="4",
        difficulty=1,
        category="prealgebra",
    )
    assert ex.difficulty == 1
    assert ex.category == "prealgebra"


def test_unknown_source_rejected() -> None:
    with pytest.raises(ValidationError):
        Example(
            id="x",
            source="other",  # type: ignore[arg-type]
            prompt="x",
            response="x",
            final_answer="x",
        )


def test_empty_required_string_rejected() -> None:
    with pytest.raises(ValidationError):
        Example(
            id="x",
            source="gsm8k",
            prompt="",
            response="x",
            final_answer="x",
        )


def test_difficulty_out_of_range_rejected() -> None:
    with pytest.raises(ValidationError):
        Example(
            id="x",
            source="math",
            prompt="x",
            response="x",
            final_answer="x",
            difficulty=10,
        )


def test_unknown_field_name_rejected() -> None:
    """A typo in a field name should raise, not be silently dropped."""
    with pytest.raises(ValidationError):
        Example(
            id="x",
            source="gsm8k",
            prompt="x",
            response="x",
            final_answer="x",
            promptt="x",  # typo  # type: ignore[call-arg]
        )


def test_instance_is_immutable() -> None:
    """frozen=True: any mutation attempt raises after construction."""
    ex = Example(
        id="x",
        source="gsm8k",
        prompt="x",
        response="x",
        final_answer="x",
    )
    with pytest.raises(ValidationError):
        ex.prompt = "y"  # type: ignore[misc]
