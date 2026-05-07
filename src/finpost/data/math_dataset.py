"""MATH loader.

Loads the MATH dataset (Hendrycks et al., 2021) from the parquet-only
mirror ``DigitalLearningGmbH/MATH-lighteval``. That mirror is MIT
licensed, attributed to the original Hendrycks authors, and ships as
parquet only — no Python loader script. See ``SECURITY.md``.

Module name is ``math_dataset`` rather than ``math`` to avoid collision
with Python's stdlib ``math`` module.

MATH gold solutions follow a flexible format: a worked solution in
LaTeX whose final answer is wrapped in either ``\\boxed{...}`` (the
common form) or ``\\boxed N`` (the LaTeX no-brace form for a single
token). Some solutions use ``\\fbox{...}`` instead. The extraction
logic in this module handles all three.

Each example also carries:

- ``difficulty`` — an integer 1..5, parsed from the ``"Level N"`` field.
- ``category`` — normalized subject (e.g. ``"counting_and_probability"``).
"""

from __future__ import annotations

from pydantic import ValidationError

from finpost.data.schema import Example
from finpost.safety import safe_load_dataset

_DATASET_ID = "DigitalLearningGmbH/MATH-lighteval"
_CONFIG_NAME = "default"


# =============================================================================
# Vendored from hendrycks/math (MIT licensed). Original authors retain
# copyright. We include verbatim source for the canonical functions plus
# small documented extensions for known gaps.
#
#   Source: https://github.com/hendrycks/math
#     File: modeling/dataset/util.py        — last_boxed_only_string
#     File: modeling/eval_math_gpt.py       — remove_boxed
#   License: MIT
#   Vendored: 2026-05-05
#
# Extensions to the originals (clearly marked in each function):
#
#   1. _remove_boxed also strips the \fbox{ wrapper. The canonical
#      last_boxed_only_string falls back to \fbox{...} when no \boxed
#      is found, but the canonical remove_boxed only handles \boxed{ —
#      a known gap (EleutherAI/lm-evaluation-harness#3116).
#
#   2. _parse_no_brace_boxed handles the LaTeX form `\boxed N` (with
#      no curly braces) where the macro's argument runs to the next
#      math-mode delimiter ($). Several MATH solutions use this form;
#      the canonical parser returns None for them.
# =============================================================================


def _last_boxed_only_string(string: str) -> str | None:
    """Return the last ``\\boxed{...}`` or ``\\fbox{...}`` substring,
    including the wrapper, with balanced braces. Returns ``None`` if no
    such substring exists.

    Vendored verbatim from hendrycks/math/modeling/dataset/util.py.
    """
    idx = string.rfind("\\boxed")
    if idx < 0:
        idx = string.rfind("\\fbox")
        if idx < 0:
            return None

    i = idx
    right_brace_idx = None
    num_left_braces_open = 0
    while i < len(string):
        if string[i] == "{":
            num_left_braces_open += 1
        if string[i] == "}":
            num_left_braces_open -= 1
            if num_left_braces_open == 0:
                right_brace_idx = i
                break
        i += 1

    if right_brace_idx is None:
        retval = None
    else:
        retval = string[idx:right_brace_idx + 1]

    return retval


def _remove_boxed(s: str) -> str | None:
    """Strip the outer ``\\boxed{...}`` or ``\\fbox{...}`` wrapper.

    Returns the inner content, or ``None`` if the wrapping doesn't
    match either expected form.

    Adapted from hendrycks/math/modeling/eval_math_gpt.py with one
    extension: the canonical version only handles ``\\boxed{`` and
    raises an ``AssertionError`` on mismatch. We additionally accept
    the ``\\fbox{`` wrapper (closing the upstream gap noted above) and
    return None on mismatch instead of asserting.
    """
    for left in ("\\boxed{", "\\fbox{"):
        if s.startswith(left) and s.endswith("}"):
            return s[len(left):-1]
    return None


def _parse_no_brace_boxed(string: str) -> str | None:
    """Handle the LaTeX form ``\\boxed N`` (no braces).

    LaTeX strictly takes only the next single token after ``\\boxed``,
    but some MATH solutions write the form expecting the answer to
    extend through the closing math-mode delimiter ``$``. We follow the
    latter convention to match observed dataset usage:

      ``... our answer is $\\boxed 2$.``  →  ``"2"``
      ``... = $\\boxed 42$.``             →  ``"42"``

    Returns ``None`` if no ``\\boxed `` (with trailing whitespace) appears.

    This is an extension to the canonical Hendrycks parser, not vendored
    code. Documented here so the divergence is explicit.
    """
    sentinel = "\\boxed "
    if sentinel not in string:
        return None
    # Take everything after the LAST '\boxed ', then trim at the next '$'.
    after_boxed = string.rsplit(sentinel, 1)[1]
    content = after_boxed.split("$", 1)[0].strip()
    return content if content else None


# =============================================================================
# End of vendored / vendored-with-extension code. Project code below.
# =============================================================================


def parse_math_final_answer(solution_text: str) -> str:
    """Extract the final answer from a MATH solution.

    Tries the canonical Hendrycks parser first (handles ``\\boxed{...}``,
    ``\\fbox{...}``, with proper brace nesting). If that returns None,
    falls back to the no-brace form ``\\boxed N``.

    Raises
    ------
    ValueError
        If no extraction succeeds in any form. The error message
        includes the full solution for debugging.
    """
    boxed_string = _last_boxed_only_string(solution_text)
    if boxed_string is not None:
        content = _remove_boxed(boxed_string)
        if content is not None:
            return content

    no_brace_content = _parse_no_brace_boxed(solution_text)
    if no_brace_content is not None:
        return no_brace_content

    raise ValueError(
        f"MATH solution: no \\boxed answer extractable in any known form: {solution_text!r}"
    )


def parse_math_difficulty(level_text: str) -> int | None:
    """Parse the MATH ``level`` field to ``int`` or ``None`` (unknown).

    The dataset uses two well-known formats:

    - ``"Level N"`` for N in 1..5 — known difficulty, returned as int.
    - ``"Level ?"`` — the dataset's marker for unknown difficulty,
      returned as ``None``. The problem itself is still valid; we just
      have no rating for it.

    Any OTHER format (typos, missing prefix, integers outside 1..5)
    raises ``ValueError``. We're permissive about the known unknowns
    and strict about everything else, so genuine format drift fails
    loudly.
    """
    prefix = "Level "
    if not level_text.startswith(prefix):
        raise ValueError(
            f"Unexpected MATH level format (expected 'Level N' or 'Level ?'): "
            f"{level_text!r}"
        )

    number_part = level_text[len(prefix):].strip()

    # The known unknown marker. Treat as missing-but-valid.
    if number_part == "?":
        return None

    try:
        n = int(number_part)
    except ValueError as exc:
        raise ValueError(
            f"MATH level value is neither an integer in 1..5 nor the known "
            f"'?' unknown-marker: {level_text!r}"
        ) from exc

    if not 1 <= n <= 5:
        raise ValueError(
            f"MATH level out of expected range 1..5: got {n} from {level_text!r}"
        )
    return n


def normalize_math_category(type_text: str) -> str:
    """Normalize the MATH ``type`` field to a snake_case slug.

    Examples:
      'Algebra'                  -> 'algebra'
      'Counting & Probability'   -> 'counting_and_probability'
      'Intermediate Algebra'     -> 'intermediate_algebra'

    The dataset's ``type`` values are presentation strings; downstream
    code wants stable slugs for filtering and comparison. Doing the
    normalization once here is cheaper than every consumer remembering
    to lowercase.
    """
    return (
        type_text.lower()
        .replace(" & ", "_and_")
        .replace(" ", "_")
    )


def load_math(split: str = "train") -> list[Example]:
    """Load a MATH split and normalize to ``Example`` records.

    Parameters
    ----------
    split
        Either ``"train"`` (~7,500 records) or ``"test"`` (~5,000 records).

    Returns
    -------
    A list of ``Example`` instances. Each carries ``difficulty``
    (1..5) and ``category`` (snake_case slug) populated from the
    source row.
    """
    if split not in ("train", "test"):
        raise ValueError(f"split must be 'train' or 'test', got {split!r}")

    raw = safe_load_dataset(
        _DATASET_ID,
        _CONFIG_NAME,
        split=split,
    )

    # Per-record skip-and-report. The MATH dataset has a small number
    # of records that fail extraction; rather than fail the entire
    # load on the first one, we count and continue. The summary printed
    # below makes the loss visible to the caller.
    #
    # Known cases as of the 2026-05-05 scan (MATH train, 7500 records):
    #   - empty_answer x2: rows where the gold solution writes
    #     `\boxed{}` (literal empty braces) for a count question
    #     whose mathematically correct answer is "0". Author typo for
    #     `\boxed{0}`. Skipped because we lack an answer-equivalence
    #     normalizer to handle the multiple ways a model might
    #     correctly emit "no such things exist" (\emptyset, \text{none},
    #     "0", etc.). Revisit when the grader normalizer lands.
    #   - test split has zero failures.
    #
    # If this number ever spikes meaningfully (>1% of records), do not
    # silently extend the skip — investigate before accepting.
    examples: list[Example] = []
    skipped: dict[str, int] = {
        "parse_failure": 0,
        "empty_answer": 0,
        "difficulty_failure": 0,
        "validation_failure": 0,
    }

    for idx, row in enumerate(raw):
        try:
            final_answer = parse_math_final_answer(row["solution"])
        except ValueError:
            skipped["parse_failure"] += 1
            continue

        if not final_answer.strip():
            skipped["empty_answer"] += 1
            continue

        try:
            difficulty = parse_math_difficulty(row["level"])
        except ValueError:
            skipped["difficulty_failure"] += 1
            continue

        try:
            examples.append(
                Example(
                    id=f"math-{split}-{idx}",
                    source="math",
                    prompt=row["problem"],
                    response=row["solution"],
                    final_answer=final_answer,
                    difficulty=difficulty,
                    category=normalize_math_category(row["type"]),
                )
            )
        except ValidationError:
            skipped["validation_failure"] += 1

    total_skipped = sum(skipped.values())
    if total_skipped > 0:
        print(
            f"[load_math] Skipped {total_skipped}/{len(raw)} records "
            f"({total_skipped / len(raw) * 100:.2f}%): {dict(skipped)}"
        )

    return examples
