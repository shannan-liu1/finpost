"""FinChain dataset loader, parser, and exact-answer scorer.

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

_DEFAULT_V2_DATA_DIR = Path("data/finchain_v2_template_disjoint")
_LEGACY_DATA_DIR = Path("data/archive/finchain_pilot")
_FINAL_MARKER_RE = re.compile(
    (
        r"(?:final[ \t]+(?:answers?|classification)|answer|classification)[ \t]*"
        r"(?:(?:is|are)[ \t]*:[ \t]*|(?:is|are)[ \t]+|[:=][ \t]*)"
        r"(?P<answer>[^\r\n]+)$"
    ),
    flags=re.IGNORECASE | re.MULTILINE,
)
_NUMERIC_RE = re.compile(r"-?(?:\$)?\d[\d,]*(?:\.\d+)?(?:e[+-]?\d+)?%?", re.IGNORECASE)
_YES_NO_RE = re.compile(r"(?:->|→|:)\s*(?P<answer>yes|no)\b", flags=re.IGNORECASE)
_CONCLUSION_RE = re.compile(r"(?:conclusion|result)\s*:\s*(?P<answer>.+)", flags=re.IGNORECASE)
_STEP_LABEL_RE = re.compile(r"\bstep\s+\d+(?:\.\d+)?\b", flags=re.IGNORECASE)
_EMPTY_FINAL_MARKER_RE = re.compile(
    (
        r"(?:the[ \t]+)?(?:final[ \t]+(?:answers?|classification)|answer|classification)"
        r"[ \t]*(?:(?:is|are)[ \t]*)?[:=][ \t]*$"
    ),
    flags=re.IGNORECASE,
)
_ANSWER_PREFIX_RE = re.compile(
    r"^(?:the\s+)?(?:final\s+)?(?:answer|classification)\s*(?:is|are|[:=])\s*",
    flags=re.IGNORECASE,
)
_ENUM_LABEL_RE = re.compile(r"^\s*(?:[-*]\s*)?(?:\(?[A-Za-z0-9]{1,3}\)?[.)])\s+")
_BUDGET_COMPONENT_RE = re.compile(
    (
        r"^\s*(?:[-*]\s*)?(?:\(?[A-Za-z0-9]{1,3}\)?[.)]\s+)?"
        r"(?:new\s+)?"
        r"(?P<label>marketing|operations)(?:\s+budget)?\s*(?::|=|is)?\s*"
    ),
    flags=re.IGNORECASE,
)

_RETIREMENT_FEASIBLE = "The plan is feasible with a projected surplus"
_RETIREMENT_NOT_FEASIBLE = (
    "The plan is NOT feasible as-is; consider higher savings later retirement "
    "lower withdrawals or different asset mix"
)
_CATEGORY_LABELS = {
    "yes": "Yes",
    "no": "No",
    "monitoring_enhanced": "Enhanced monitoring required",
    "monitoring_standard": "Standard monitoring sufficient",
    "retirement_feasible": _RETIREMENT_FEASIBLE,
    "retirement_not_feasible": _RETIREMENT_NOT_FEASIBLE,
}


def normalize_finchain_answer(value: str) -> str:
    """Normalize a FinChain final answer for exact/numeric comparison."""
    cleaned = str(value).strip()
    cleaned = cleaned.replace("−", "-")
    cleaned = cleaned.lstrip("$").rstrip(".,;")
    cleaned = cleaned.replace(",", "")
    if cleaned.endswith("%"):
        cleaned = cleaned[:-1]
    return cleaned.strip()


def _strip_latex_text_wrappers(value: str) -> str:
    """Remove simple wrappers that commonly surround categorical decisions."""
    cleaned = str(value)
    cleaned = cleaned.replace("\\(", " ").replace("\\)", " ")
    cleaned = cleaned.replace("\\[", " ").replace("\\]", " ")
    cleaned = cleaned.replace("\\boxed{", "").replace("\\text{", "")
    cleaned = cleaned.replace("}", "")
    cleaned = cleaned.replace("**", "")
    return cleaned.strip()


def _plain_categorical_text(value: str) -> str:
    cleaned = _strip_latex_text_wrappers(value)
    cleaned = _ANSWER_PREFIX_RE.sub("", cleaned).strip()
    cleaned = _ENUM_LABEL_RE.sub("", cleaned).strip()
    cleaned = normalize_finchain_answer(cleaned)
    cleaned = cleaned.lower()
    cleaned = re.sub(r"[^a-z0-9]+", " ", cleaned)
    return re.sub(r"\s+", " ", cleaned).strip()


def _contains_any(text: str, phrases: tuple[str, ...]) -> bool:
    return any(phrase in text for phrase in phrases)


def _categorical_answer_key(value: str, *, expected_key: str | None = None) -> str | None:
    """Map common FinChain categorical paraphrases to normalized answer keys."""
    text = _plain_categorical_text(value)
    if not text:
        return None

    monitoring_standard = (
        "standard monitoring sufficient",
        "standard monitoring is sufficient",
        "no enhanced monitoring",
        "not required",
        "is not required",
        "enhanced monitoring not required",
        "enhanced monitoring is not required",
    )
    monitoring_enhanced = (
        "enhanced monitoring required",
        "enhanced monitoring is required",
        "requires enhanced monitoring",
        "monitoring is required",
    )
    monitoring_standard_specific = tuple(
        phrase
        for phrase in monitoring_standard
        if phrase not in {"not required", "is not required"}
    )
    retirement_not_feasible = (
        "plan is not feasible",
        "not feasible as is",
        "not feasible",
        "not sufficient",
        "insufficient",
        "shortfall",
        "deficit",
    )
    retirement_feasible = (
        "plan is feasible",
        "feasible with a projected surplus",
        "projected surplus",
        "savings are sufficient",
        "savings will be sufficient",
        "will be sufficient",
    )
    kyc_negative = (
        "not valid",
        "not typically considered valid",
        "invalid",
        "not eligible",
        "not acceptable",
        "not sufficient",
        "not compliant",
        "cannot be used",
        "not be used",
        "does not qualify",
        "may not be sufficient",
    )
    kyc_positive = (
        "generally valid",
        "is valid",
        "valid for",
        "eligible",
        "acceptable",
        "compliant",
        "can be used",
        "approved",
    )

    if text in {"yes", "true"}:
        if expected_key == "monitoring_enhanced":
            return "monitoring_enhanced"
        if expected_key == "retirement_feasible":
            return "retirement_feasible"
        return "yes"
    if text in {"no", "false"}:
        if expected_key == "monitoring_standard":
            return "monitoring_standard"
        if expected_key == "retirement_not_feasible":
            return "retirement_not_feasible"
        return "no"
    if text in {"sufficient", "feasible"} and expected_key == "retirement_feasible":
        return "retirement_feasible"
    if text in {"not sufficient", "infeasible"} and expected_key == "retirement_not_feasible":
        return "retirement_not_feasible"

    if expected_key == "monitoring_standard" and _contains_any(text, monitoring_standard):
        return "monitoring_standard"
    if expected_key == "monitoring_enhanced" and _contains_any(text, monitoring_enhanced):
        return "monitoring_enhanced"
    if expected_key == "retirement_not_feasible" and _contains_any(text, retirement_not_feasible):
        return "retirement_not_feasible"
    if expected_key == "retirement_feasible" and _contains_any(text, retirement_feasible):
        return "retirement_feasible"
    if expected_key == "no" and _contains_any(text, kyc_negative):
        return "no"
    if expected_key == "yes" and _contains_any(text, kyc_positive):
        return "yes"

    if _contains_any(text, monitoring_standard_specific):
        return "monitoring_standard"
    if _contains_any(text, monitoring_enhanced):
        return "monitoring_enhanced"
    if _contains_any(text, kyc_negative):
        return "no"
    if _contains_any(text, kyc_positive):
        return "yes"
    if _contains_any(text, retirement_not_feasible):
        return "retirement_not_feasible"
    if _contains_any(text, retirement_feasible):
        return "retirement_feasible"
    return None


def _categorical_answer_label(value: str) -> str | None:
    key = _categorical_answer_key(value)
    if key is None:
        return None
    return _CATEGORY_LABELS[key]


def _answer_from_candidate(candidate: str) -> str:
    """Normalize one candidate answer line into the shortest scoreable answer."""
    cleaned = _strip_latex_text_wrappers(candidate.strip().splitlines()[0].strip())
    cleaned = _ANSWER_PREFIX_RE.sub("", cleaned).strip()
    cleaned = _ENUM_LABEL_RE.sub("", cleaned).strip()
    budget_components = _budget_components_answer(cleaned)
    if budget_components is not None:
        return budget_components

    yes_no_matches = list(_YES_NO_RE.finditer(cleaned))
    if yes_no_matches:
        return normalize_finchain_answer(yes_no_matches[-1].group("answer"))

    if cleaned.lower() in {"yes", "no"}:
        return normalize_finchain_answer(cleaned)

    conclusion_match = _CONCLUSION_RE.search(cleaned)
    if conclusion_match:
        cleaned = conclusion_match.group("answer").strip()

    numeric_matches = _NUMERIC_RE.findall(_STEP_LABEL_RE.sub("", cleaned))
    if numeric_matches:
        return normalize_finchain_answer(numeric_matches[-1])

    categorical = _categorical_answer_label(cleaned)
    if categorical is not None:
        return categorical

    return normalize_finchain_answer(cleaned)


def _budget_components_answer(text: str) -> str | None:
    """Extract the requested two-budget answer instead of the unchanged total."""
    components: dict[str, str] = {}
    segments = [
        segment
        for line in text.splitlines()
        for segment in line.split(";")
    ]
    for segment in segments:
        match = _BUDGET_COMPONENT_RE.search(segment)
        if match is None:
            continue
        numeric_matches = _NUMERIC_RE.findall(segment)
        if not numeric_matches:
            continue
        label = match.group("label").lower()
        amount = normalize_finchain_answer(numeric_matches[-1])
        components[label] = amount

    if {"marketing", "operations"} <= components.keys():
        return (
            f"marketing={components['marketing']};"
            f"operations={components['operations']}"
        )
    return None


def parse_finchain_final_answer(text: str) -> str:
    """Extract a final answer from a FinChain-style solution or generation."""
    marker_matches = list(_FINAL_MARKER_RE.finditer(text))
    if marker_matches:
        candidate_text = marker_matches[-1].group("answer").strip()
        if candidate_text:
            candidate = candidate_text.splitlines()[0]
            normalized = _answer_from_candidate(candidate)
            if normalized:
                return normalized

    lines = [line.strip() for line in text.splitlines() if line.strip()]
    for idx in range(len(lines) - 1, -1, -1):
        if _EMPTY_FINAL_MARKER_RE.search(lines[idx]):
            answer_block = "\n".join(lines[idx + 1 : idx + 6])
            budget_components = _budget_components_answer(answer_block)
            if budget_components is not None:
                return budget_components
            break

    budget_components = _budget_components_answer(text)
    if budget_components is not None:
        return budget_components

    for line in reversed(lines):
        normalized = _answer_from_candidate(line)
        if (
            normalized.lower() in {"yes", "no"}
            or normalized in _CATEGORY_LABELS.values()
            or _CONCLUSION_RE.search(line)
        ):
            return normalized

    numeric_matches = _NUMERIC_RE.findall(_STEP_LABEL_RE.sub("", text))
    if numeric_matches:
        return normalize_finchain_answer(numeric_matches[-1])

    raise ValueError(f"FinChain text has no extractable final answer: {text!r}")


def try_parse_finchain_final_answer(text: str) -> str | None:
    """Extract a final answer, returning None when model output is unparsable."""
    try:
        return parse_finchain_final_answer(text)
    except ValueError:
        return None


def _raw_answer_should_fall_back_to_response(raw_answer: str) -> bool:
    """Detect exported answer fields that are explanatory lines, not answers."""
    stripped = raw_answer.strip()
    lowered = stripped.lower()
    if "reason" in lowered:
        return True
    answer_text = _STEP_LABEL_RE.sub("", stripped)
    if _NUMERIC_RE.search(answer_text) or _YES_NO_RE.search(stripped):
        return False
    if _CONCLUSION_RE.search(stripped):
        return False
    return lowered.startswith(("step", "thus", "therefore"))


def _row_needs_response_derived_answer(row: dict[str, Any], prompt: str) -> bool:
    """Identify FinChain rows where the exported scalar answer is not the target."""
    template_name = str(row.get("template_name") or "")
    if template_name == "template_budget_reallocation_due_to_costs":
        return True

    normalized_prompt = prompt.lower()
    return (
        "new marketing and operations budgets" in normalized_prompt
        or "marketing and operations budget" in normalized_prompt
    )


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
        pass
    else:
        if pred_decimal.is_finite() and gold_decimal.is_finite():
            return pred_decimal == gold_decimal

    gold_key = _categorical_answer_key(gold_norm)
    predicted_key = _categorical_answer_key(predicted_norm, expected_key=gold_key)
    return gold_key is not None and predicted_key == gold_key


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
    if _row_needs_response_derived_answer(row, str(prompt)):
        final_answer = _budget_components_answer(str(response))
        if final_answer is None:
            final_answer = parse_finchain_final_answer(str(response))
    elif raw_answer is None or _raw_answer_should_fall_back_to_response(str(raw_answer)):
        final_answer = parse_finchain_final_answer(str(response))
    else:
        try:
            final_answer = parse_finchain_final_answer(str(raw_answer))
        except ValueError:
            final_answer = parse_finchain_final_answer(str(response))
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
    default_path = _DEFAULT_V2_DATA_DIR / f"{split}.jsonl"
    if default_path.exists():
        return default_path
    return _LEGACY_DATA_DIR / f"{split}.jsonl"


def load_finchain(split: str = "test") -> list[Example]:
    """Load the default FinChain split, or an explicit env override."""
    if split not in ("train", "validation", "test"):
        raise ValueError("split must be 'train', 'validation', or 'test'")
    path = resolve_finchain_path(split)
    if not path.exists():
        raise FileNotFoundError(
            f"FinChain JSONL not found at {path}. Set "
            f"FINPOST_FINCHAIN_{split.upper()}_JSONL to a local audited export."
        )
    return load_finchain_jsonl(path, split=split)
