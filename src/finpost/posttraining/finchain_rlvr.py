"""FinChain helpers for industry RLVR trainers.

The repo keeps the DPO/OPD/GRPO math small and testable, but notebook runs often
need a slightly different surface: prompt-only rows for TRL/Axolotl-style
trainers and reward functions that can be passed directly to GRPO or online-DPO
APIs. This module is that adapter layer.
"""

from __future__ import annotations

import json
import random
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from finpost.data.schema import Example
from finpost.evals.finchain_metrics import grade_finchain_generation
from finpost.training.dataset import serialize_prompt


def completion_to_text(completion: Any) -> str:
    """Normalize trainer-specific completion objects into plain text.

    TRL can hand reward functions either raw strings or conversational message
    lists. Keeping this conversion here lets the notebook reward function stay
    one line while still documenting the real interface mismatch.
    """
    if isinstance(completion, str):
        return completion

    if isinstance(completion, dict):
        content = completion.get("content")
        return str(content if content is not None else completion)

    if isinstance(completion, Sequence):
        parts: list[str] = []
        for item in completion:
            if isinstance(item, dict) and item.get("content") is not None:
                parts.append(str(item["content"]))
            else:
                parts.append(str(item))
        return "\n".join(parts)

    return str(completion)


def _as_batch(values: Any, *, n: int, field_name: str) -> list[str]:
    if values is None:
        raise ValueError(f"{field_name} is required to score FinChain rewards")
    if isinstance(values, str):
        return [values] * n
    if isinstance(values, Sequence):
        if len(values) != n:
            raise ValueError(
                f"{field_name} length {len(values)} does not match completions length {n}"
            )
        return [str(value) for value in values]
    return [str(values)] * n


def finchain_binary_rewards(
    *args: Any,
    completions: list[Any] | None = None,
    gold_answer: Any = None,
    final_answer: Any = None,
    answer: Any = None,
    **_: Any,
) -> list[float]:
    """Return `1.0` for correct FinChain final answers and `0.0` otherwise.

    TRL trainer variants do not all call reward functions with the same
    positional shape. GRPO-style hooks commonly pass completions directly;
    OnlineDPO-style hooks may pass `(prompts, completions)`. This adapter
    accepts both and ignores prompts because FinChain scoring only needs the
    completion text plus the gold answer column.

    The keyword aliases match the column names used by this repo
    (`gold_answer`), the common normalized schema (`final_answer`), and some
    public datasets (`answer`). This makes the function usable with TRL reward
    hooks without wrapping it in every notebook.
    """
    if completions is None:
        if len(args) == 1:
            completions = args[0]
        elif len(args) >= 2:
            completions = args[1]
        else:
            raise ValueError("completions are required to score FinChain rewards")

    gold_values = gold_answer if gold_answer is not None else final_answer
    if gold_values is None:
        gold_values = answer
    gold_batch = _as_batch(gold_values, n=len(completions), field_name="gold_answer")

    rewards: list[float] = []
    for completion, gold in zip(completions, gold_batch, strict=True):
        grade = grade_finchain_generation(completion_to_text(completion), gold_answer=gold)
        rewards.append(1.0 if grade.final_answer_correct else 0.0)
    return rewards


def build_finchain_prompt_rows(
    examples: Sequence[Example],
    *,
    format_prompt: bool = True,
) -> list[dict[str, Any]]:
    """Convert FinChain examples into prompt rows for TRL-style trainers."""
    rows: list[dict[str, Any]] = []
    for example in examples:
        rows.append(
            {
                "prompt": serialize_prompt(example.prompt) if format_prompt else example.prompt,
                "raw_prompt": example.prompt,
                "gold_answer": example.final_answer,
                "prompt_id": example.id,
                "source": example.source,
                "difficulty": example.difficulty,
                "topic": example.topic,
                "subtopic": example.subtopic,
            }
        )
    return rows


def deterministic_sample(
    examples: Sequence[Example],
    *,
    n: int | None,
    seed: int,
) -> list[Example]:
    """Return a deterministic prompt subset without mutating the caller's list."""
    rows = list(examples)
    if n is None:
        return rows
    if n <= 0:
        raise ValueError("n must be positive when provided")
    if n > len(rows):
        raise ValueError(f"requested {n} examples, but only {len(rows)} are available")
    rng = random.Random(seed)
    rng.shuffle(rows)
    return rows[:n]


def write_jsonl(path: str | Path, rows: Sequence[dict[str, Any]]) -> None:
    """Write JSONL with ASCII-safe encoding for portable RunPod artifacts."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fp:
        for row in rows:
            fp.write(json.dumps(row, ensure_ascii=True) + "\n")
