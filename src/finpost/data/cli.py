"""Command-line interface for inspecting Phase 1 datasets.

Loads a dataset (GSM8K or MATH), tokenizes the records with the Gemma
3 1B tokenizer, prints summary length statistics and a sample example.
Useful for sanity-checking that the loaders work end-to-end and for
deciding context-length budgets before training.

Usage:
    python -m finpost.data.cli --dataset gsm8k --split train
    python -m finpost.data.cli --dataset math --split test --limit 200
"""

from __future__ import annotations

import argparse
import statistics
from collections import Counter
from dataclasses import dataclass

from transformers import AutoTokenizer

from finpost.data.gsm8k import load_gsm8k
from finpost.data.math_dataset import load_math
from finpost.data.schema import Example

# Default tokenizer — the canonical Phase 1 training base. Overridable
# via --tokenizer at the command line.
_DEFAULT_TOKENIZER = "google/gemma-3-1b-it"

# How many examples to tokenize at once. Tokenization is fast; the
# only reason to batch at all is to amortize Python overhead. 64 is a
# round number that fits comfortably in any reasonable memory budget.
_TOKENIZE_BATCH_SIZE = 64


@dataclass(frozen=True)
class LengthStats:
    """Summary statistics for a list of token counts.

    All fields are integers except ``mean``. ``p50`` is the median,
    ``p95`` the 95th percentile, ``max`` the longest in the set.
    """

    count: int
    mean: float
    p50: int
    p95: int
    max: int

    def format(self, label: str) -> str:
        """Render as a human-readable block under the given label."""
        return (
            f"{label} (n={self.count})\n"
            f"  mean: {self.mean:6.1f}\n"
            f"  p50:  {self.p50:6d}\n"
            f"  p95:  {self.p95:6d}\n"
            f"  max:  {self.max:6d}"
        )


def compute_length_stats(token_counts: list[int]) -> LengthStats:
    """Compute count / mean / p50 / p95 / max from a list of integers.

    The p50 and p95 are computed with ``sorted_list[index]`` rather
    than scipy/numpy percentiles. For our sample sizes (thousands)
    that's accurate enough; for percentile-based gating decisions in
    production you'd want a proper interpolated percentile.
    """
    if not token_counts:
        raise ValueError("Cannot compute length stats over an empty list")

    sorted_counts = sorted(token_counts)
    n = len(sorted_counts)
    return LengthStats(
        count=n,
        mean=statistics.mean(token_counts),
        # n // 2 is the index of the median for odd-length lists; for
        # even-length lists we take the upper of the two middle
        # values rather than the average — close enough for stats
        # purposes and avoids fractional medians on integer counts.
        p50=sorted_counts[n // 2],
        # int() floors; fine — small underestimate at the boundary.
        p95=sorted_counts[int(n * 0.95)],
        max=sorted_counts[-1],
    )


def tokenize_lengths(texts: list[str], tokenizer) -> list[int]:
    """Token-count each text using the given tokenizer.

    Batched for speed. ``add_special_tokens=False`` so the count
    reflects the user-visible content of each example, not the
    BOS / EOS / chat-template overhead the trainer adds later.
    """
    counts: list[int] = []
    for batch_start in range(0, len(texts), _TOKENIZE_BATCH_SIZE):
        batch = texts[batch_start : batch_start + _TOKENIZE_BATCH_SIZE]
        # The tokenizer accepts a list and returns a dict whose
        # 'input_ids' is a list of lists (one per input string).
        encoded = tokenizer(batch, add_special_tokens=False, return_tensors=None)
        for ids in encoded["input_ids"]:
            counts.append(len(ids))
    return counts


def _load_dataset(name: str, split: str) -> list[Example]:
    """Dispatch to the right loader based on --dataset name."""
    if name == "gsm8k":
        return load_gsm8k(split)
    if name == "math":
        return load_math(split)
    raise ValueError(f"Unknown dataset: {name!r}")


def _print_math_distributions(examples: list[Example]) -> None:
    """Print category and difficulty breakdowns. Only meaningful for MATH."""
    cats = Counter(ex.category for ex in examples)
    diffs = Counter(ex.difficulty for ex in examples)
    print("\nCategory distribution:")
    for k in sorted(cats):
        print(f"  {k:30s} {cats[k]:5d}")
    print("\nDifficulty distribution:")
    # difficulty can be None for the dataset's "Level ?" unknown-marker.
    # sort the int keys first; print None separately at the bottom so we
    # don't have to compare None < int.
    int_keys = sorted(k for k in diffs if k is not None)
    for k in int_keys:
        print(f"  Level {k}: {diffs[k]:5d}")
    none_count = diffs.get(None, 0)
    if none_count:
        print(f"  Level ? (unknown): {none_count:5d}")


def _print_examples(examples: list[Example], n: int) -> None:
    """Pretty-print the first n examples for visual inspection."""
    print(f"\nFirst {n} example(s):")
    for i, ex in enumerate(examples[:n]):
        print(f"\n--- Example {i} (id={ex.id}) ---")
        print(f"Prompt: {ex.prompt}")
        print(f"Response: {ex.response}")
        print(f"Final answer: {ex.final_answer!r}")
        if ex.difficulty is not None:
            print(f"Difficulty: {ex.difficulty}")
        if ex.category is not None:
            print(f"Category: {ex.category}")


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="finpost.data.cli",
        description=(
            "Inspect a Phase 1 dataset: counts, length distribution, "
            "and a sample example."
        ),
    )
    parser.add_argument(
        "--dataset",
        choices=("gsm8k", "math"),
        required=True,
        help="Which dataset to load.",
    )
    parser.add_argument(
        "--split",
        choices=("train", "test"),
        default="train",
        help="Which split to load (default: train).",
    )
    parser.add_argument(
        "--tokenizer",
        default=_DEFAULT_TOKENIZER,
        help=f"Tokenizer to use for length stats (default: {_DEFAULT_TOKENIZER}).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional cap on number of examples (useful for fast iteration).",
    )
    parser.add_argument(
        "--num-examples",
        type=int,
        default=1,
        help="How many full examples to print at the end (default: 1).",
    )
    args = parser.parse_args()

    print(f"Loading {args.dataset} ({args.split})...")
    examples = _load_dataset(args.dataset, args.split)

    if args.limit is not None:
        examples = examples[: args.limit]
        print(f"Truncated to {len(examples)} examples (--limit {args.limit})")

    print(f"Loaded {len(examples)} examples.")

    print(f"\nLoading tokenizer {args.tokenizer}...")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)

    print("\nTokenizing for length stats...")
    prompt_lengths = tokenize_lengths([ex.prompt for ex in examples], tokenizer)
    response_lengths = tokenize_lengths([ex.response for ex in examples], tokenizer)
    combined_lengths = [p + r for p, r in zip(prompt_lengths, response_lengths)]

    print()
    print(compute_length_stats(prompt_lengths).format("Prompt tokens"))
    print()
    print(compute_length_stats(response_lengths).format("Response tokens"))
    print()
    print(compute_length_stats(combined_lengths).format("Prompt + Response tokens"))

    if args.dataset == "math":
        _print_math_distributions(examples)

    _print_examples(examples, args.num_examples)


if __name__ == "__main__":
    main()
