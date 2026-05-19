"""Common record schema for Phase 1 training data.

Both the GSM8K and MATH loaders normalize to the ``Example`` shape so
the trainer never needs to know which dataset a record came from.
Anything dataset-specific (difficulty level, subject category) lives
in optional fields.

Validation is enforced by Pydantic at construction time. If a loader
tries to build an ``Example`` with bad data — empty string in a
required field, unknown source, out-of-range difficulty — Pydantic
raises ``ValidationError`` and the bad record never enters the
pipeline. The loader is responsible for catching and reporting; the
schema is responsible for refusing.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

# Tiny type alias for the ``source`` field. Used in the model below
# and re-exportable for callers that want to refer to the allowed
# values without restating the literal.
Source = Literal["gsm8k", "math", "finchain"]


class Example(BaseModel):
    """A single Phase 1 training example.

    Both GSM8K and MATH examples normalize to this shape. Fields that
    apply only to one dataset (``difficulty`` and ``category``, both
    from MATH) are optional and default to ``None`` for GSM8K.
    """

    # frozen=True       instances are immutable after construction.
    #                   Stops accidental mutation as records flow
    #                   through the pipeline. Catch bugs early.
    # extra="forbid"    unknown field names at construction raise an
    #                   error rather than being silently dropped.
    #                   Catches typos like ``promtp=...`` immediately.
    model_config = ConfigDict(frozen=True, extra="forbid")

    id: str = Field(
        ...,
        min_length=1,
        description="Unique identifier within the source dataset.",
    )
    source: Source = Field(
        ...,
        description="Which dataset this example came from.",
    )
    prompt: str = Field(
        ...,
        min_length=1,
        description="The question text shown to the model.",
    )
    response: str = Field(
        ...,
        min_length=1,
        description="Gold response: chain of thought + sentinel + final answer.",
    )
    final_answer: str = Field(
        ...,
        min_length=1,
        description="Parsed final answer used by the grader.",
    )
    difficulty: int | None = Field(
        default=None,
        ge=1,
        le=5,
        description="MATH difficulty level 1-5; None for GSM8K.",
    )
    category: str | None = Field(
        default=None,
        description="MATH subject category (e.g. 'algebra'); None for GSM8K.",
    )
    domain: str | None = Field(
        default=None,
        description="FinChain financial domain; None for non-FinChain sources.",
    )
    topic: str | None = Field(
        default=None,
        description="FinChain topic; None for non-FinChain sources.",
    )
    subtopic: str | None = Field(
        default=None,
        description="FinChain subtopic/template label; None when unavailable.",
    )
