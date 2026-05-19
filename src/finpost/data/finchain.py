"""FinChain local-data loader.

FinChain's public project ships executable Python templates. This loader
does not execute those templates. It reads local JSONL exports that were
generated or audited elsewhere and normalizes them into the repo's common
``Example`` schema.
"""

from __future__ import annotations

import json
import os
import re
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any

from finpost.data.schema import Example

_DEFAULT_DATA_DIR = Path("data/finchain")
_FINAL_MARKER_RE = re.compile(
    r"(?:final\s+answer|answer)\s*[:=]\s*(?P<answer>.+)$",
    flags=re.IGNORECASE | re.MULTILINE,
)
_NUMERIC_RE = re.compile(r"-?(?:\$)?\d[\d,]*(?:\.\d+)?(?:e[+-]?\d+)?%?", re.IGNORECASE)


def normalize_finchain_answer(value: str) -> str:
    """Normalize a FinChain final answer for exact/numeric comparison."""
    cleaned = str(value).strip()
    cleaned = cleaned.replace("−", "-")
    cleaned = cleaned.lstrip("$").rstrip(".,;")
    cleaned = cleaned.replace(",", "")
    if cleaned.endswith("%"):
        cleaned = cleaned[:-1]
    return cleaned.strip()


def parse_finchain_final_answer(text: str) -> str:
    """Extract a final answer from a FinChain-style solution or generation."""
    marker_matches = list(_FINAL_MARKER_RE.finditer(text))
    if marker_matches:
        candidate = marker_matches[-1].group("answer").strip().splitlines()[0]
        token = candidate.split()[0] if candidate.split() else ""
        normalized = normalize_finchain_answer(token)
        if normalized:
            return normalized

    numeric_matches = _NUMERIC_RE.findall(text)
    if numeric_matches:
        return normalize_finchain_answer(numeric_matches[-1])

    raise ValueError(f"FinChain text has no extractable final answer: {text!r}")


def try_parse_finchain_final_answer(text: str) -> str | None:
    """Extract a final answer, returning None when model output is unparsable."""
    try:
        return parse_finchain_final_answer(text)
    except ValueError:
        return None


def score_finchain_answer(predicted: str | None, gold: str) -> bool:
    """Score FinChain final answers with formatting-tolerant numeric equality."""
    if predicted is None:
        return False

    predicted_norm = normalize_finchain_answer(predicted)
    gold_norm = normalize_finchain_answer(gold)
    if predicted_norm == gold_norm:
        return True

    try:
        pred_decimal = Decimal(predicted_norm)
        gold_decimal = Decimal(gold_norm)
    except (InvalidOperation, ValueError):
        return False
    if not (pred_decimal.is_finite() and gold_decimal.is_finite()):
        return False
    return pred_decimal == gold_decimal


def _first_present(row: dict[str, Any], keys: tuple[str, ...]) -> Any:
    for key in keys:
        value = row.get(key)
        if value is not None and str(value).strip():
            return value
    return None


def _parse_difficulty(value: Any) -> int | None:
    if value is None:
        return None
    if isinstance(value, int):
        return value if 1 <= value <= 5 else None

    lowered = str(value).strip().lower()
    mapping = {
        "easy": 1,
        "basic": 1,
        "intermediate": 3,
        "medium": 3,
        "advanced": 5,
        "hard": 5,
    }
    if lowered in mapping:
        return mapping[lowered]
    if lowered.startswith("level "):
        lowered = lowered.removeprefix("level ").strip()
    try:
        parsed = int(lowered)
    except ValueError:
        return None
    return parsed if 1 <= parsed <= 5 else None


def _normalize_row(row: dict[str, Any], *, idx: int, split: str) -> Example:
    prompt = _first_present(row, ("problem", "question", "prompt"))
    response = _first_present(row, ("solution", "reasoning", "response", "trace"))
    if prompt is None:
        raise ValueError(f"FinChain row {idx} missing problem/question/prompt")
    if response is None:
        raise ValueError(f"FinChain row {idx} missing solution/reasoning/response/trace")

    raw_answer = _first_present(row, ("answer", "final_answer", "gold_answer"))
    final_answer = (
        normalize_finchain_answer(str(raw_answer))
        if raw_answer is not None
        else parse_finchain_final_answer(str(response))
    )
    if not final_answer:
        raise ValueError(f"FinChain row {idx} has empty final answer")

    return Example(
        id=str(row.get("id") or f"finchain-{split}-{idx}"),
        source="finchain",
        prompt=str(prompt),
        response=str(response),
        final_answer=final_answer,
        difficulty=_parse_difficulty(_first_present(row, ("level", "difficulty"))),
        category=str(row["topic"]) if row.get("topic") else None,
        domain=str(row["domain"]) if row.get("domain") else None,
        topic=str(row["topic"]) if row.get("topic") else None,
        subtopic=str(row["subtopic"]) if row.get("subtopic") else None,
    )


def load_finchain_jsonl(path: str | Path, *, split: str = "train") -> list[Example]:
    """Load a local FinChain JSONL export into common ``Example`` records."""
    path = Path(path)
    examples: list[Example] = []
    with path.open("r", encoding="utf-8") as fp:
        for idx, line in enumerate(fp):
            stripped = line.strip()
            if not stripped:
                continue
            row = json.loads(stripped)
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{idx + 1} must be a JSON object")
            examples.append(_normalize_row(row, idx=idx, split=split))

    if not examples:
        raise ValueError(f"no FinChain examples found in {path}")
    return examples


def resolve_finchain_path(split: str) -> Path:
    """Resolve the local FinChain JSONL path for a split."""
    env_key = f"FINPOST_FINCHAIN_{split.upper()}_JSONL"
    env_value = os.environ.get(env_key)
    if env_value:
        return Path(env_value)
    return _DEFAULT_DATA_DIR / f"{split}.jsonl"


def load_finchain(split: str = "test") -> list[Example]:
    """Load a FinChain split from ``data/finchain`` or an env override."""
    if split not in ("train", "validation", "test"):
        raise ValueError("split must be 'train', 'validation', or 'test'")
    path = resolve_finchain_path(split)
    if not path.exists():
        raise FileNotFoundError(
            f"FinChain JSONL not found at {path}. Set "
            f"FINPOST_FINCHAIN_{split.upper()}_JSONL to a local audited export."
        )
    return load_finchain_jsonl(path, split=split)
