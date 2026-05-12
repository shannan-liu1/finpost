"""Domain-agnostic source registry for Phase 1 evaluation.

Defines the ``EvalSource`` dataclass contract and registers the two
initial Phase 1 benchmarks (GSM8K and MATH). Each entry encapsulates
dataset-specific logic: loading examples, parsing model generations to
extract final answers, and scoring against gold standard answers.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from typing import Callable

from finpost.data.gsm8k import load_gsm8k
from finpost.data.math_dataset import load_math, try_parse_math_final_answer
from finpost.data.schema import Example

# =============================================================================
# Decimal-based numeric equality helper
# =============================================================================


def _decimal_equal(predicted: str, gold: str) -> bool:
    """Return True iff both strings parse as finite Decimals that compare equal.

    Handles "42.0" == "42", "4.2e3" == "4200", and similar non-string-equal
    numeric cases. Rejects nan, inf, -inf, and non-numeric strings (the
    caller's string-equality check has already failed by the time we get here).

    Decimal is used instead of float to avoid precision artifacts and to
    explicitly reject non-finite values (float("nan") == float("nan") is
    False, but float("inf") == float("inf") is True — surprising in a
    scoring context where inf is not a valid numeric answer).
    """
    try:
        a = Decimal(predicted)
        b = Decimal(gold)
    except (InvalidOperation, ValueError, TypeError):
        return False
    if not (a.is_finite() and b.is_finite()):
        return False
    return a == b


# =============================================================================
# Module-level counter for _strip_string normalization failures (RB1).
#
# score_math increments this counter whenever _strip_string raises inside the
# try/except block.  run_eval in eval_exact resets it at the start of each
# invocation and prints a summary at the end so the operator knows if a
# suspiciously low MATH accuracy is partly explained by normalization errors.
#
# A module-level int is intentional here: eval_exact is a single-process CLI,
# there is no concurrency, and this avoids threading the counter through every
# function signature in the generation/scoring stack.
# =============================================================================

_strip_string_failure_count: int = 0


# =============================================================================
# Vendored Hendrycks MATH LaTeX normalization helpers
#
# Adapted from Hendrycks et al. (2021) MATH dataset evaluation:
# https://github.com/hendrycks/math/blob/main/modeling/math_equivalence.py
# Vendored here under MIT license. The original function handles LaTeX
# normalization required for fair string comparison: \dfrac/\tfrac/\frac,
# \left(\right), spacing macros, and trailing-zero handling.
#
# All four helpers (_fix_fracs, _fix_a_slash_b, _fix_sqrt,
# _remove_right_units) are required by _strip_string; vendored verbatim.
# =============================================================================


def _fix_fracs(string: str) -> str:
    substrs = string.split("\\frac")
    new_str = substrs[0]
    if len(substrs) > 1:
        substrs = substrs[1:]
        for substr in substrs:
            new_str += "\\frac"
            if substr[0] == "{":
                new_str += substr
            else:
                try:
                    assert len(substr) >= 2
                except Exception:
                    return string
                a = substr[0]
                b = substr[1]
                if b != "{":
                    if len(substr) > 2:
                        post_substr = substr[2:]
                        new_str += "{" + a + "}{" + b + "}" + post_substr
                    else:
                        new_str += "{" + a + "}{" + b + "}"
                else:
                    if len(substr) > 2:
                        post_substr = substr[2:]
                        new_str += "{" + a + "}" + b + post_substr
                    else:
                        new_str += "{" + a + "}" + b
    string = new_str
    return string


def _fix_a_slash_b(string: str) -> str:
    if len(string.split("/")) != 2:
        return string
    a = string.split("/")[0]
    b = string.split("/")[1]
    try:
        a = int(a)
        b = int(b)
        assert string == "{}/{}".format(a, b)
        new_string = "\\frac{" + str(a) + "}{" + str(b) + "}"
        return new_string
    except Exception:
        return string


def _remove_right_units(string: str) -> str:
    if "\\text{ " in string:
        splits = string.split("\\text{ ")
        assert len(splits) == 2
        return splits[0]
    else:
        return string


def _fix_sqrt(string: str) -> str:
    if "\\sqrt" not in string:
        return string
    splits = string.split("\\sqrt")
    new_string = splits[0]
    for split in splits[1:]:
        if split[0] != "{":
            a = split[0]
            new_substr = "\\sqrt{" + a + "}" + split[1:]
        else:
            new_substr = "\\sqrt" + split
        new_string += new_substr
    return new_string


def _strip_string(string: str) -> str:
    """Normalize a LaTeX math string for comparison.

    Applies the canonical Hendrycks MATH grader normalizations:
    removes newlines and spacing macros, normalizes fraction variants
    (\\dfrac, \\tfrac) to \\frac, strips \\left/\\right size hints,
    degree symbols, currency markers, and trailing-zero handling on
    decimals. Called on both predicted and gold before comparison in
    ``score_math``.
    """
    string = string.replace("\n", "")
    string = string.replace("\\!", "")
    string = string.replace("\\\\", "\\")
    string = string.replace("tfrac", "frac")
    string = string.replace("dfrac", "frac")
    string = string.replace("\\left", "")
    string = string.replace("\\right", "")
    string = string.replace("^{\\circ}", "")
    string = string.replace("^\\circ", "")
    string = string.replace("\\$", "")
    string = _remove_right_units(string)
    string = string.replace("\\%", "")
    string = string.replace("\\%", "")
    string = string.replace(" .", " 0.")
    string = string.replace("{.", "{0.")
    if len(string) == 0:
        return string
    if string[0] == ".":
        string = "0" + string
    if len(string.split("=")) == 2:
        if len(string.split("=")[0]) <= 2:
            string = string.split("=")[1]
    string = _fix_sqrt(string)
    string = string.replace(" ", "")
    string = _fix_fracs(string)
    if string == "0.5":
        string = "\\frac{1}{2}"
    string = _fix_a_slash_b(string)
    return string


# =============================================================================
# End of vendored Hendrycks normalization code.
# =============================================================================


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

    - Normalizing Unicode minus (U+2212) to ASCII hyphen. Models
      occasionally emit U+2212 when copying from LaTeX or other typeset
      math contexts; GSM8K gold answers always use ASCII ``-``.
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
    # Normalize Unicode minus (U+2212) to ASCII hyphen, which is what
    # GSM8K gold answers use.
    generation = generation.replace("−", "-")

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
    """Score a GSM8K answer via exact string match, with a numeric fallback.

    The extractor is responsible for normalization on the model side.
    After exact string comparison, a float-equality fallback handles the
    common case where the model emits ``"42.0"`` when gold is ``"42"``,
    or uses scientific notation like ``"4.2e3"`` for ``"4200"``. Both
    sides must parse as float for the fallback to apply; if either side
    fails to parse, the fallback returns ``False`` (non-numeric strings
    are not numerically equal to a number).

    Parameters
    ----------
    predicted
        The extracted answer from the model, or ``None`` if extraction
        failed.
    gold
        The gold answer (already normalized by the data loader).

    Returns
    -------
    ``True`` if predicted and gold match exactly or numerically,
    ``False`` otherwise. Predicted ``None`` is always ``False``.
    """
    if predicted is None:
        return False
    if predicted == gold:
        return True
    # Numeric fallback: handles "42.0" vs "42", "4.2e3" vs "4200", etc.
    # _decimal_equal rejects nan/inf and non-numeric strings explicitly.
    return _decimal_equal(predicted, gold)


# =============================================================================
# MATH answer extractor and score function
# =============================================================================


def extract_math_answer(generation: str) -> str | None:
    """Extract the final answer from a MATH-format generation.

    Delegates the actual brace/no-brace parsing to
    ``try_parse_math_final_answer`` so eval-time extraction stays
    symmetric with the data-loader: ``\\boxed{...}``, ``\\fbox{...}``,
    and the no-brace LaTeX form ``\\boxed N`` are all accepted. Prior
    to this unification the eval extractor only accepted
    ``\\boxed{...}`` with a literal opening brace, which silently
    marked otherwise-correct ``\\fbox`` and ``\\boxed N`` outputs as
    parse-fail.

    Post-normalizations applied here (and not in the data loader,
    which trusts its gold solutions to be already-clean LaTeX):

    - Unicode minus (U+2212) normalized to ASCII hyphen. Models that
      copy from typeset math contexts sometimes emit U+2212.
    - Leading and trailing whitespace stripped.
    - Outer ``$...$`` math-mode wrapping stripped if present.

    Returns ``None`` if no marker is found, if braces are unbalanced,
    or if nothing remains after normalization.

    Parameters
    ----------
    generation
        The model's full text generation.

    Returns
    -------
    The extracted answer (string), or ``None`` on parse failure.
    """
    # Normalize Unicode minus (U+2212) to ASCII hyphen, which is what
    # MATH gold answers use.
    generation = generation.replace("−", "-")

    inner = try_parse_math_final_answer(generation)
    if inner is None:
        return None

    # Normalize: strip whitespace and outer $ if present.
    inner = inner.strip()
    if inner.startswith("$") and inner.endswith("$"):
        inner = inner[1:-1]
    inner = inner.strip()

    if not inner:
        return None

    return inner


def score_math(predicted: str | None, gold: str) -> bool:
    """Score a MATH answer using Hendrycks LaTeX normalization and numeric fallback.

    Applies three layers in order:

    1. Exact string match (fast path, no normalization cost).
    2. Hendrycks ``_strip_string`` normalization on both sides — handles
       ``\\dfrac``/``\\tfrac`` → ``\\frac``, ``\\left``/``\\right``
       removal, spacing macros, and other canonical LaTeX transforms.
       Wrapped in ``try/except`` because ``_strip_string`` can raise on
       pathological input (e.g., unbalanced LaTeX).
    3. Float-equality fallback — handles purely numeric answers where
       representation differs (e.g., ``"42.0"`` vs ``"42"``). LaTeX
       strings will fail ``float()`` parsing and fall through to
       ``False``.

    Parameters
    ----------
    predicted
        The extracted answer from the model, or ``None`` if extraction
        failed.
    gold
        The gold answer (already normalized by the data loader).

    Returns
    -------
    ``True`` if predicted and gold match under any of the three layers,
    ``False`` otherwise. Predicted ``None`` is always ``False``.
    """
    global _strip_string_failure_count

    if predicted is None:
        return False
    if predicted == gold:
        return True
    # Hendrycks LaTeX normalization layer.
    try:
        if _strip_string(predicted) == _strip_string(gold):
            return True
    except Exception:
        # _strip_string can raise on pathological input — e.g., the bare
        # assert in _remove_right_units (string with two \text{ markers) or
        # an IndexError in _fix_sqrt (string ending in \sqrt).  Increment the
        # module-level counter so the caller (run_eval) can report how many
        # examples fell through to the direct equality / numeric fallback
        # rather than the proper normalization path.
        _strip_string_failure_count += 1
    # Numeric fallback: handles "42.0" vs "42", "4.2e3" vs "4200", etc.
    # LaTeX strings (e.g., \frac{1}{2}) will fail Decimal parsing and fall
    # through to False, which is correct. _decimal_equal also rejects nan/inf.
    return _decimal_equal(predicted, gold)


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
