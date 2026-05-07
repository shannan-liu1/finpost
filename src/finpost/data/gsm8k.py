"""GSM8K loader.

Loads the GSM8K (Grade School Math 8K) dataset from Hugging Face,
parses each gold answer to extract the final number, and returns a
list of normalized ``Example`` records.

The dataset is loaded via ``safe_load_dataset`` (which passes
``trust_remote_code=False``). With remote code disallowed, the
``datasets`` library auto-discovers and loads the parquet files
shipped alongside the loader script — so the upstream Python loader
is never executed even though we point at the canonical ``main``
branch. See ``SECURITY.md`` for the policy.

(We tried explicitly pinning to ``refs/convert/parquet`` first; that
branch's auto-converted configs are flattened to a single ``default``
name, which loses the ``main`` vs ``socratic`` distinction this
project needs. Trusting the upstream-name + parquet-fallback path is
both correct and idiomatic.)

GSM8K's gold answer format is well-defined: each ``answer`` field is a
chain of thought followed by a sentinel line ``#### N`` where ``N`` is
the final numeric answer. The parser below extracts ``N`` and strips
any commas the source uses for thousands separators (the canonical
grader compares against the comma-less form).
"""

from __future__ import annotations

from finpost.data.schema import Example
from finpost.safety import safe_load_dataset

# The Hugging Face dataset identifier and configuration. Hardcoded
# rather than parameterized: this loader is specifically for GSM8K's
# main configuration. If a different config is needed (e.g. the
# 'socratic' variant), write a separate function.
_DATASET_ID = "openai/gsm8k"
_CONFIG_NAME = "main"


def parse_gsm8k_final_answer(answer_text: str) -> str:
    """Extract the final answer from a GSM8K-formatted response.

    Convention: GSM8K answers end with a line ``#### N``. Everything
    after the *last* ``####`` is the final answer. Commas are stripped
    (the dataset writes 1,200 but the canonical grader compares 1200).

    Parameters
    ----------
    answer_text
        The full ``answer`` field from a GSM8K row.

    Returns
    -------
    The parsed final answer as a string (string, not int — some answers
    are decimals; downstream graders normalize as needed).

    Raises
    ------
    ValueError
        If the ``####`` sentinel is missing or the final answer is
        empty after parsing. These are data-integrity errors and we
        prefer to fail loudly over returning a silent ``None``.
    """
    if "####" not in answer_text:
        raise ValueError(
            f"GSM8K answer missing '####' sentinel: {answer_text!r}"
        )

    # rsplit with maxsplit=1 takes everything after the LAST '####'.
    # Important if the chain of thought ever happened to contain '####'
    # itself for some reason (rare but possible).
    after_sentinel = answer_text.rsplit("####", 1)[1].strip()

    if not after_sentinel:
        raise ValueError(
            f"GSM8K answer has empty content after '####': {answer_text!r}"
        )

    # Take the first whitespace-delimited token; strip thousands-separator
    # commas; strip any trailing sentence punctuation.
    first_token = after_sentinel.split()[0]
    cleaned = first_token.replace(",", "").rstrip(".,")

    if not cleaned:
        raise ValueError(
            f"GSM8K final answer empty after cleaning: {answer_text!r}"
        )

    return cleaned


def load_gsm8k(split: str = "train") -> list[Example]:
    """Load a GSM8K split and normalize to ``Example`` records.

    Parameters
    ----------
    split
        Either ``"train"`` (~7,473 records) or ``"test"`` (~1,319 records).

    Returns
    -------
    A list of ``Example`` instances. Construction-time validation runs
    on each — if any record has missing or empty fields, Pydantic
    raises ``ValidationError`` and the whole load fails. We prefer
    early failure over silently dropping bad rows.
    """
    if split not in ("train", "test"):
        raise ValueError(f"split must be 'train' or 'test', got {split!r}")

    raw = safe_load_dataset(
        _DATASET_ID,
        _CONFIG_NAME,
        split=split,
    )

    examples: list[Example] = []
    for idx, row in enumerate(raw):
        examples.append(
            Example(
                id=f"gsm8k-{split}-{idx}",
                source="gsm8k",
                prompt=row["question"],
                response=row["answer"],
                final_answer=parse_gsm8k_final_answer(row["answer"]),
            )
        )
    return examples
