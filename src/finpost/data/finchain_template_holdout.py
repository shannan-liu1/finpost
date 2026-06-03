"""Build stratified FinChain splits with whole-template holdout.

The input catalog lists upstream FinChain template functions and metadata. This
module assigns each template function to exactly one split, which prevents
template leakage across train, validation, and test.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

SPLITS = ("train", "validation", "test")
TEMPLATE_KEY_FIELDS = ("domain", "template_file", "template_name")
DIFFICULTY_ORDER = ("basic", "intermediate", "advanced")
DEFAULT_SPLIT_RATIOS = {"train": 0.8, "validation": 0.1, "test": 0.1}


def template_key(row: Mapping[str, Any]) -> tuple[str, str, str]:
    """Return the stable identity for one FinChain template function."""
    missing = [field for field in TEMPLATE_KEY_FIELDS if not row.get(field)]
    if missing:
        raise ValueError(f"FinChain row missing template key fields: {missing}")
    return tuple(str(row[field]) for field in TEMPLATE_KEY_FIELDS)


def _difficulty(row: Mapping[str, Any]) -> str:
    value = row.get("difficulty") or row.get("level")
    if value is None:
        raise ValueError(f"FinChain row missing difficulty/level: {row}")
    lowered = str(value).strip().lower()
    mapping = {
        "easy": "basic",
        "basic": "basic",
        "medium": "intermediate",
        "intermediate": "intermediate",
        "hard": "advanced",
        "advanced": "advanced",
    }
    if lowered not in mapping:
        raise ValueError(f"unsupported FinChain difficulty: {value!r}")
    return mapping[lowered]


def _stable_sort_key(value: object, seed: int) -> str:
    return hashlib.sha256(f"{seed}::{value!r}".encode()).hexdigest()


def _flatten_rows(rows_by_split: Mapping[str, Sequence[Mapping[str, Any]]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for source_split in SPLITS:
        for row in rows_by_split.get(source_split, []):
            copied = dict(row)
            copied.setdefault("original_split", source_split)
            rows.append(copied)
    return rows


def _group_by_template(
    rows: Iterable[Mapping[str, Any]],
) -> dict[tuple[str, str, str], list[dict[str, Any]]]:
    groups: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[template_key(row)].append(dict(row))
    if not groups:
        raise ValueError("no FinChain rows found")

    for key, group_rows in groups.items():
        difficulties = {_difficulty(row) for row in group_rows}
        if len(difficulties) != 1:
            raise ValueError(f"template {key} has inconsistent difficulties: {difficulties}")
    return dict(groups)


def _hamilton_counts(total: int, ratios: Mapping[str, float]) -> dict[str, int]:
    ratio_sum = sum(ratios.values())
    if not math.isclose(ratio_sum, 1.0):
        raise ValueError(f"split ratios must sum to 1.0, got {ratio_sum}")

    raw = {split: ratios[split] * total for split in SPLITS}
    counts = {split: int(math.floor(raw[split])) for split in SPLITS}
    remaining = total - sum(counts.values())
    remainder_order = sorted(
        SPLITS,
        key=lambda split: (raw[split] - counts[split], split),
        reverse=True,
    )
    for split in remainder_order[:remaining]:
        counts[split] += 1
    return counts


def _difficulty_split_targets(
    difficulty_totals: Mapping[str, int],
    split_template_targets: Mapping[str, int],
) -> dict[str, dict[str, int]]:
    """Allocate difficulty counts to splits while preserving split totals."""
    difficulties = tuple(d for d in DIFFICULTY_ORDER if difficulty_totals.get(d, 0))
    if len(difficulties) != 3:
        raise ValueError(
            "expected FinChain basic/intermediate/advanced difficulties: "
            f"{difficulty_totals}"
        )

    total = sum(difficulty_totals.values())
    expected = {
        split: {
            difficulty: difficulty_totals[difficulty] * split_template_targets[split] / total
            for difficulty in DIFFICULTY_ORDER
        }
        for split in SPLITS
    }
    targets = {
        split: {
            difficulty: int(math.floor(expected[split][difficulty]))
            for difficulty in DIFFICULTY_ORDER
        }
        for split in SPLITS
    }
    row_remaining = {
        difficulty: difficulty_totals[difficulty]
        - sum(targets[split][difficulty] for split in SPLITS)
        for difficulty in DIFFICULTY_ORDER
    }
    column_remaining = {
        split: split_template_targets[split]
        - sum(targets[split][difficulty] for difficulty in DIFFICULTY_ORDER)
        for split in SPLITS
    }

    while any(column_remaining.values()):
        candidates = [
            (
                expected[split][difficulty] - targets[split][difficulty],
                split,
                difficulty,
            )
            for split in SPLITS
            for difficulty in DIFFICULTY_ORDER
            if column_remaining[split] > 0 and row_remaining[difficulty] > 0
        ]
        if not candidates:
            raise ValueError("could not allocate difficulty targets")
        _, split, difficulty = max(candidates)
        targets[split][difficulty] += 1
        row_remaining[difficulty] -= 1
        column_remaining[split] -= 1

    return targets


def _assign_one_difficulty(
    template_ids: Sequence[tuple[str, str, str]],
    *,
    split_targets: Mapping[str, int],
    seed: int,
) -> dict[tuple[str, str, str], str]:
    """Assign one difficulty stratum, balancing domains as a secondary objective."""
    by_domain: dict[str, list[tuple[str, str, str]]] = defaultdict(list)
    for template_id in template_ids:
        by_domain[template_id[0]].append(template_id)

    remaining = dict(split_targets)
    assigned_by_domain: dict[str, Counter[str]] = defaultdict(Counter)
    domain_targets = {
        domain: {
            split: len(domain_templates) * split_targets[split] / len(template_ids)
            for split in SPLITS
        }
        for domain, domain_templates in by_domain.items()
    }
    assignments: dict[tuple[str, str, str], str] = {}

    domains = sorted(by_domain, key=lambda value: _stable_sort_key(value, seed))
    ordered_templates: list[tuple[str, str, str]] = []
    while any(by_domain.values()):
        for domain in domains:
            domain_templates = by_domain[domain]
            if not domain_templates:
                continue
            domain_templates.sort(key=lambda value: _stable_sort_key(value, seed))
            ordered_templates.append(domain_templates.pop(0))

    for template_id in ordered_templates:
        domain = template_id[0]
        candidates = [split for split in SPLITS if remaining[split] > 0]
        if not candidates:
            raise ValueError("split assignment exhausted early")

        def candidate_score(
            split: str,
            *,
            current_domain: str = domain,
            current_template_id: tuple[str, str, str] = template_id,
        ) -> tuple[float, int, str]:
            domain_deficit = (
                domain_targets[current_domain][split]
                - assigned_by_domain[current_domain][split]
            )
            return (
                domain_deficit,
                remaining[split],
                _stable_sort_key((current_template_id, split), seed),
            )

        split = max(candidates, key=candidate_score)
        assignments[template_id] = split
        remaining[split] -= 1
        assigned_by_domain[domain][split] += 1

    if any(remaining.values()):
        raise ValueError(f"split assignment left unused targets: {remaining}")
    return assignments


def _template_id_str(template_id: tuple[str, str, str]) -> str:
    return "::".join(template_id)


def _overlap_counts(split_template_ids: Mapping[str, set[tuple[str, str, str]]]) -> dict[str, int]:
    return {
        f"{left}__{right}": len(split_template_ids[left] & split_template_ids[right])
        for index, left in enumerate(SPLITS)
        for right in SPLITS[index + 1 :]
    }


def _count_rows(rows: Sequence[Mapping[str, Any]], field: str) -> dict[str, int]:
    return dict(sorted(Counter(str(row.get(field)) for row in rows).items()))


def build_template_holdout_splits(
    rows_by_split: Mapping[str, Sequence[Mapping[str, Any]]],
    *,
    seed: int = 20260524,
    split_ratios: Mapping[str, float] | None = None,
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    """Return template-disjoint FinChain splits and an audit manifest."""
    split_ratios = DEFAULT_SPLIT_RATIOS if split_ratios is None else split_ratios
    groups = _group_by_template(_flatten_rows(rows_by_split))
    template_difficulty = {
        template_id: _difficulty(group_rows[0]) for template_id, group_rows in groups.items()
    }
    difficulty_totals = Counter(template_difficulty.values())
    split_template_targets = _hamilton_counts(len(groups), split_ratios)
    difficulty_targets = _difficulty_split_targets(difficulty_totals, split_template_targets)

    assignments: dict[tuple[str, str, str], str] = {}
    for difficulty in DIFFICULTY_ORDER:
        template_ids = [
            template_id
            for template_id, template_difficulty_value in template_difficulty.items()
            if template_difficulty_value == difficulty
        ]
        assignments.update(
            _assign_one_difficulty(
                template_ids,
                split_targets={
                    split: difficulty_targets[split][difficulty] for split in SPLITS
                },
                seed=seed + DIFFICULTY_ORDER.index(difficulty),
            )
        )

    split_rows: dict[str, list[dict[str, Any]]] = {split: [] for split in SPLITS}
    for template_id, group_rows in groups.items():
        split = assignments[template_id]
        for row in sorted(group_rows, key=lambda item: str(item.get("seed", ""))):
            copied = dict(row)
            copied["split_strategy"] = "template_holdout"
            copied["template_holdout_seed"] = seed
            split_rows[split].append(copied)

    for split, rows in split_rows.items():
        rng = random.Random(seed + len(split))
        rng.shuffle(rows)

    split_template_ids = {
        split: {template_key(row) for row in rows} for split, rows in split_rows.items()
    }
    manifest = {
        "created_at": datetime.now(UTC).isoformat(),
        "seed": seed,
        "split_strategy": "template_holdout",
        "template_key_fields": list(TEMPLATE_KEY_FIELDS),
        "source_split_row_counts": {
            split: len(rows_by_split.get(split, [])) for split in SPLITS
        },
        "split_row_counts": {split: len(rows) for split, rows in split_rows.items()},
        "split_template_counts": {
            split: len(split_template_ids[split]) for split in SPLITS
        },
        "split_template_ids": {
            split: sorted(
                _template_id_str(template_id) for template_id in split_template_ids[split]
            )
            for split in SPLITS
        },
        "template_overlap_counts": _overlap_counts(split_template_ids),
        "split_difficulty_template_counts": {
            split: {
                difficulty: sum(
                    1
                    for template_id in split_template_ids[split]
                    if template_difficulty[template_id] == difficulty
                )
                for difficulty in DIFFICULTY_ORDER
            }
            for split in SPLITS
        },
        "split_difficulty_row_counts": {
            split: _count_rows(rows, "difficulty") for split, rows in split_rows.items()
        },
        "split_domain_template_counts": {
            split: dict(
                sorted(Counter(template_id[0] for template_id in split_template_ids[split]).items())
            )
            for split in SPLITS
        },
        "split_domain_row_counts": {
            split: _count_rows(rows, "domain") for split, rows in split_rows.items()
        },
        "notes": [
            "Template functions are assigned to exactly one split.",
            "This is a stricter companion eval to the same-template FinChain split.",
            (
                "FinChain has one advanced template per topic file, so balancing is "
                "global by domain and difficulty rather than one holdout per "
                "topic-file difficulty."
            ),
        ],
    }
    return split_rows, manifest


def load_rows_from_jsonl(paths_by_split: Mapping[str, Path]) -> dict[str, list[dict[str, Any]]]:
    """Load raw JSONL rows for each split."""
    rows_by_split: dict[str, list[dict[str, Any]]] = {}
    for split in SPLITS:
        path = paths_by_split[split]
        rows_by_split[split] = [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    return rows_by_split


def write_template_holdout_splits(
    split_rows: Mapping[str, Sequence[Mapping[str, Any]]],
    manifest: Mapping[str, Any],
    output_dir: Path,
) -> None:
    """Write split JSONLs and an audit manifest."""
    output_dir.mkdir(parents=True, exist_ok=True)
    for split in SPLITS:
        path = output_dir / f"{split}.jsonl"
        with path.open("w", encoding="utf-8") as handle:
            for row in split_rows[split]:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
