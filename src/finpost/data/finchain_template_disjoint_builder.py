"""Build the FinChain template-disjoint fresh-seed dataset.

The ignored audited source catalog contains upstream FinChain template functions
and their difficulty/domain metadata. This module uses that catalog for
stratification, then invokes pinned upstream template functions to create fresh
instances after assigning each template function to exactly one split.
"""

from __future__ import annotations

import calendar
import hashlib
import json
import random
import sys
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from types import ModuleType
from typing import Any

from finpost.data.finchain_dataset import parse_finchain_final_answer
from finpost.data.finchain_template_holdout import (
    SPLITS,
    build_template_holdout_splits,
    template_key,
)

DEFAULT_INSTANCES_PER_TEMPLATE = {"train": 10, "validation": 30, "test": 30}
_SEED_POPULATION = range(1_000_000_000, 4_000_000_001)
UPSTREAM_COMPATIBILITY_SHIMS = {
    "corporate_finance/eps.py": (
        "Inject Python stdlib calendar: upstream EPS template functions "
        "reference calendar.month_name without a module-level import."
    ),
    "risk_management/sensitivity_analysis.py": (
        "Inject generate_random_value(low, high) as round(random.uniform(low, high), 2): "
        "upstream volatility template calls this missing helper."
    ),
}
UPSTREAM_SOURCE_PATCHES = {
    "mergers_and_acquisitions/post-merger_integration.py": {
        "^{-{years}}": "^(-{years})",
    },
    "personal_finance/saveretire.py": {
        "^{-{retirement_years}}": "^(-{retirement_years})",
    },
}


def _module_from_template_file(path: Path, *, template_relative_path: str) -> ModuleType:
    """Import one upstream template module while supporting sibling imports."""
    if not path.exists():
        raise FileNotFoundError(f"upstream FinChain template module not found: {path}")
    module_name = "finchain_template_" + hashlib.sha256(str(path).encode()).hexdigest()[:20]
    module = ModuleType(module_name)
    module.__file__ = str(path)
    source = path.read_text(encoding="utf-8")
    for old, new in UPSTREAM_SOURCE_PATCHES.get(template_relative_path, {}).items():
        if old not in source:
            raise ValueError(
                "expected upstream source patch target not found in "
                f"{template_relative_path}: {old}"
            )
        source = source.replace(old, new)
    inserted = str(path.parent)
    sys.path.insert(0, inserted)
    try:
        exec(compile(source, str(path), "exec"), module.__dict__)
    finally:
        sys.path.remove(inserted)
    return module


def _generate_one_row(
    *,
    catalog_row: Mapping[str, Any],
    upstream_templates_dir: Path,
    generated_seed: int,
    split: str,
    template_assignment_seed: int,
    generation_seed: int,
    module_cache: dict[tuple[str, str], ModuleType],
) -> dict[str, Any]:
    key = template_key(catalog_row)
    domain, template_file, template_name = key
    module_key = (domain, template_file)
    module = module_cache.get(module_key)
    if module is None:
        template_relative_path = f"{domain}/{template_file}.py"
        module = _module_from_template_file(
            upstream_templates_dir / domain / f"{template_file}.py",
            template_relative_path=template_relative_path,
        )
        if template_relative_path == "corporate_finance/eps.py":
            module.calendar = calendar
        elif template_relative_path == "risk_management/sensitivity_analysis.py":
            module.generate_random_value = lambda low, high: round(random.uniform(low, high), 2)
        module_cache[module_key] = module
    template_function = getattr(module, template_name, None)
    if not callable(template_function):
        raise AttributeError(
            f"upstream template function not found: {domain}/{template_file}.py::{template_name}"
        )

    previous_state = random.getstate()
    random.seed(generated_seed)
    try:
        generated = template_function()
    finally:
        random.setstate(previous_state)
    if (
        not isinstance(generated, tuple)
        or len(generated) < 2
        or not all(isinstance(value, str) for value in generated)
    ):
        raise ValueError(
            f"template {key} did not return (question: str, solution: str[, ...])"
        )
    question, solution, *alternative_solutions = generated
    final_answer = parse_finchain_final_answer(solution)

    return {
        "source": "finchain",
        "domain": domain,
        "template_file": template_file,
        "template_name": template_name,
        "template_id": catalog_row.get("template_id"),
        "level": catalog_row.get("level"),
        "difficulty": catalog_row.get("difficulty"),
        "title": catalog_row.get("title"),
        "topic": template_file,
        "subtopic": template_name,
        "seed": generated_seed,
        "question": question,
        "problem": question,
        "prompt": question,
        "solution": solution,
        "reasoning": solution,
        "response": solution,
        "answer": final_answer,
        "final_answer": final_answer,
        "extra_outputs": {"alternative_solutions": alternative_solutions},
        "split_strategy": "template_disjoint_fresh_seed_v2",
        "template_assignment_seed": template_assignment_seed,
        "generation_seed": generation_seed,
        "split": split,
    }


def build_fresh_template_holdout_splits(
    rows_by_split: Mapping[str, Sequence[Mapping[str, Any]]],
    *,
    upstream_root: Path,
    template_assignment_seed: int = 20260524,
    generation_seed: int = 20260525,
    instances_per_template: Mapping[str, int] | None = None,
    upstream_revision: str | None = None,
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    """Render fresh rows for template-disjoint splits from upstream functions."""
    instances_per_template = (
        DEFAULT_INSTANCES_PER_TEMPLATE
        if instances_per_template is None
        else dict(instances_per_template)
    )
    if set(instances_per_template) != set(SPLITS):
        raise ValueError(f"instances_per_template must define exactly {SPLITS}")
    if any(instances_per_template[split] < 1 for split in SPLITS):
        raise ValueError("instances_per_template values must all be positive")

    assigned_rows, assignment_manifest = build_template_holdout_splits(
        rows_by_split,
        seed=template_assignment_seed,
    )
    catalog: dict[tuple[str, str, str], Mapping[str, Any]] = {}
    assignments: dict[tuple[str, str, str], str] = {}
    for split in SPLITS:
        for row in assigned_rows[split]:
            key = template_key(row)
            catalog.setdefault(key, row)
            assignments[key] = split

    total_rows = sum(
        instances_per_template[split]
        for key, split in assignments.items()
    )
    seed_rng = random.Random(generation_seed)
    generated_seeds = iter(seed_rng.sample(_SEED_POPULATION, total_rows))
    upstream_templates_dir = upstream_root / "data" / "templates"
    module_cache: dict[tuple[str, str], ModuleType] = {}
    split_rows: dict[str, list[dict[str, Any]]] = {split: [] for split in SPLITS}
    for key in sorted(assignments):
        split = assignments[key]
        for _ in range(instances_per_template[split]):
            split_rows[split].append(
                _generate_one_row(
                    catalog_row=catalog[key],
                    upstream_templates_dir=upstream_templates_dir,
                    generated_seed=next(generated_seeds),
                    split=split,
                    template_assignment_seed=template_assignment_seed,
                    generation_seed=generation_seed,
                    module_cache=module_cache,
                )
            )

    for split in SPLITS:
        random.Random(generation_seed + len(split)).shuffle(split_rows[split])
    seed_sets = {
        split: {int(row["seed"]) for row in split_rows[split]} for split in SPLITS
    }
    seed_overlap_count = sum(
        len(seed_sets[left] & seed_sets[right])
        for idx, left in enumerate(SPLITS)
        for right in SPLITS[idx + 1 :]
    )
    manifest = {
        **assignment_manifest,
        "split_strategy": "template_disjoint_fresh_seed_v2",
        "template_assignment_seed": template_assignment_seed,
        "generation_seed": generation_seed,
        "upstream_revision": upstream_revision,
        "upstream_compatibility_shims": dict(UPSTREAM_COMPATIBILITY_SHIMS),
        "upstream_source_patches": dict(UPSTREAM_SOURCE_PATCHES),
        "instances_per_template": dict(instances_per_template),
        "split_row_counts": {split: len(split_rows[split]) for split in SPLITS},
        "split_difficulty_row_counts": {
            split: dict(
                sorted(Counter(str(row["difficulty"]) for row in split_rows[split]).items())
            )
            for split in SPLITS
        },
        "split_domain_row_counts": {
            split: dict(sorted(Counter(str(row["domain"]) for row in split_rows[split]).items()))
            for split in SPLITS
        },
        "generated_seed_overlap_count": seed_overlap_count,
        "multiple_solution_policy": (
            "Use the first upstream returned solution as the training and ChainEval "
            "reference; preserve later returned solutions in extra_outputs."
        ),
        "notes": [
            "Each template function is assigned to exactly one split.",
            "Rows were freshly rendered from pinned upstream template functions.",
            (
                "Validation and test contain repeated seeded instances per held-out "
                "template; report row-level and macro-by-template metrics."
            ),
        ],
    }
    return split_rows, manifest


def write_fresh_template_holdout_splits(
    split_rows: Mapping[str, Sequence[Mapping[str, Any]]],
    manifest: dict[str, Any],
    output_dir: Path,
) -> dict[str, Any]:
    """Write v2 JSONLs and attach file SHA256 checksums to the manifest."""
    output_dir.mkdir(parents=True, exist_ok=True)
    checksums: dict[str, str] = {}
    for split in SPLITS:
        path = output_dir / f"{split}.jsonl"
        with path.open("w", encoding="utf-8") as handle:
            for row in split_rows[split]:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        checksums[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()

    written_manifest = {**manifest, "checksums_sha256": checksums}
    (output_dir / "manifest.json").write_text(
        json.dumps(written_manifest, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return written_manifest
