"""Domain-agnostic source registry for Phase 1 evaluation.

Defines the ``EvalSource`` dataclass contract and registers the two
initial Phase 1 benchmarks (GSM8K and MATH). Each entry encapsulates
dataset-specific logic: loading examples, parsing model generations to
extract final answers, and scoring against gold standard answers.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from finpost.data.gsm8k import load_gsm8k
from finpost.data.math_dataset import load_math
from finpost.data.schema import Example


@dataclass(frozen=True)
class EvalSource:
    """Contract for a Phase 1 evaluation source.

    Attributes
    ----------
    name
        The source identifier (e.g., ``"gsm8k"``, ``"math"``).
    load_examples
        A callable that, when invoked, returns the test split as a list
        of ``Example`` instances. Wrapped as a thunk to defer dataset
        downloads until explicitly called.
    extract_answer
        A callable that parses a model generation (a string) and returns
        the extracted final answer as a normalized string, or ``None``
        if parsing fails (e.g., no answer marker found).
    score
        A callable that takes (predicted_answer, gold_answer) and
        returns ``True`` if they match under the source's grading rule,
        ``False`` otherwise. Predicted ``None`` is always incorrect.
    default_max_new_tokens
        A suggested generation budget (max tokens) for this source.
        Used by the CLI as a default if not overridden by the user.

    The dataclass is frozen to prevent accidental mutation of registry
    entries at runtime.
    """

    name: str
    load_examples: Callable[[], list[Example]]
    extract_answer: Callable[[str], str | None]
    score: Callable[[str | None, str], bool]
    default_max_new_tokens: int


# =============================================================================
# GSM8K answer extractor and score function
# =============================================================================


def extract_gsm8k_answer(generation: str) -> str | None:
    """Extract the final numeric answer from a GSM8K-format generation.

    GSM8K's gold convention is a final line ``#### <number>``. The
    extractor finds the *last* ``####`` marker (to handle the rare case
    where chain-of-thought might contain a ``####`` artifact) and
    returns the answer immediately following it, normalized by:

    - Stripping leading and trailing whitespace.
    - Stripping leading dollar signs (currency notation).
    - Stripping trailing periods and commas (punctuation).

    Returns ``None`` if no ``####`` marker is found or if nothing
    remains after normalization.

    Parameters
    ----------
    generation
        The model's full text generation.

    Returns
    -------
    The extracted numeric answer (string, not float — some answers are
    decimals or negative), or ``None`` on parse failure.
    """
    if "####" not in generation:
        return None

    # rsplit with maxsplit=1 takes everything after the LAST '####'.
    after_marker = generation.rsplit("####", 1)[1].strip()

    if not after_marker:
        return None

    # Take the first whitespace-delimited token (the number itself).
    first_token = after_marker.split()[0]

    # Normalize: remove leading $, trailing . and ,, and all commas
    # (thousands separators).
    cleaned = first_token.lstrip("$").rstrip(",.").replace(",", "")

    if not cleaned:
        return None

    return cleaned


def score_gsm8k(predicted: str | None, gold: str) -> bool:
    """Score a GSM8K answer via exact string match after normalization.

    Both strings are compared as-is (no additional normalization here;
    the extractor is responsible for normalization on the model side).

    Parameters
    ----------
    predicted
        The extracted answer from the model, or ``None`` if extraction
        failed.
    gold
        The gold answer (already normalized by the data loader).

    Returns
    -------
    ``True`` if predicted and gold match exactly, ``False`` otherwise.
    Predicted ``None`` is always ``False``.
    """
    if predicted is None:
        return False
    return predicted == gold


# =============================================================================
# MATH answer extractor and score function
# =============================================================================


def extract_math_answer(generation: str) -> str | None:
    """Extract the final answer from a MATH-format generation.

    MATH's gold convention is ``\\boxed{<answer>}``. The extractor
    finds the *last* ``\\boxed{...}`` in the generation, respecting
    balanced braces (important for nested LaTeX like ``\\frac{1}{2}``),
    and returns the inner content normalized by:

    - Stripping leading and trailing whitespace.
    - Stripping outer ``$`` if present.

    Returns ``None`` if no ``\\boxed{...}`` is found or if the braces
    are unbalanced or if nothing remains after normalization.

    Parameters
    ----------
    generation
        The model's full text generation.

    Returns
    -------
    The extracted answer (string), or ``None`` on parse failure.
    """
    if "\\boxed" not in generation:
        return None

    # Find the LAST occurrence of \boxed.
    last_idx = generation.rfind("\\boxed")

    # Expect the opening brace immediately after \boxed.
    if last_idx + len("\\boxed") >= len(generation):
        return None
    if generation[last_idx + len("\\boxed")] != "{":
        return None

    # Count braces to find the matching closing brace.
    brace_start = last_idx + len("\\boxed")
    brace_count = 0
    closing_idx = None

    for i in range(brace_start, len(generation)):
        if generation[i] == "{":
            brace_count += 1
        elif generation[i] == "}":
            brace_count -= 1
            if brace_count == 0:
                closing_idx = i
                break

    # If braces are unbalanced, return None.
    if closing_idx is None:
        return None

    # Extract the content between the braces (not including the braces).
    inner = generation[brace_start + 1 : closing_idx]

    # Normalize: strip whitespace and outer $ if present.
    inner = inner.strip()
    if inner.startswith("$") and inner.endswith("$"):
        inner = inner[1:-1]
    inner = inner.strip()

    if not inner:
        return None

    return inner


def score_math(predicted: str | None, gold: str) -> bool:
    """Score a MATH answer via exact string match.

    Both strings are compared as-is (no additional normalization).

    Parameters
    ----------
    predicted
        The extracted answer from the model, or ``None`` if extraction
        failed.
    gold
        The gold answer (already normalized by the data loader).

    Returns
    -------
    ``True`` if predicted and gold match exactly, ``False`` otherwise.
    Predicted ``None`` is always ``False``.
    """
    if predicted is None:
        return False
    return predicted == gold


# =============================================================================
# Registry
# =============================================================================


# The REGISTRY dict maps source names to their EvalSource definitions.
# Each entry's ``load_examples`` is wrapped as a lambda (thunk) so the
# registry import itself does not trigger dataset downloads.
REGISTRY: dict[str, EvalSource] = {
    "gsm8k": EvalSource(
        name="gsm8k",
        load_examples=lambda: load_gsm8k("test"),
        extract_answer=extract_gsm8k_answer,
        score=score_gsm8k,
        default_max_new_tokens=256,
    ),
    "math": EvalSource(
        name="math",
        load_examples=lambda: load_math("test"),
        extract_answer=extract_math_answer,
        score=score_math,
        default_max_new_tokens=768,
    ),
}
