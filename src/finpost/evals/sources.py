"""Source registry for deterministic FinChain exact-answer evaluation."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from finpost.data.finchain_dataset import (
    load_finchain,
    score_finchain_answer,
    try_parse_finchain_final_answer,
)
from finpost.data.schema import Example


@dataclass(frozen=True)
class EvalSource:
    """Contract for an exact-answer evaluation source."""

    name: str
    load_examples: Callable[[], list[Example]]
    extract_answer: Callable[[str], str | None]
    score: Callable[[str | None, str], bool]
    default_max_new_tokens: int


REGISTRY: dict[str, EvalSource] = {
    "finchain": EvalSource(
        name="finchain",
        load_examples=lambda: load_finchain("test"),
        extract_answer=try_parse_finchain_final_answer,
        score=score_finchain_answer,
        default_max_new_tokens=768,
    ),
}
