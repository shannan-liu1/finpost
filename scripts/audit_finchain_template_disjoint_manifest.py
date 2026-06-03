"""Audit the FinChain template-disjoint split manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from collections.abc import Iterable
from pathlib import Path
from typing import Any

SPLITS = ("train", "validation", "test")
EXPECTED_ROW_COUNTS = {"train": 2320, "validation": 870, "test": 870}
EXPECTED_STRATEGY = "template_disjoint_fresh_seed_v2"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            row = json.loads(stripped)
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_number} is not a JSON object")
            rows.append(row)
    return rows


def _template_id(row: dict[str, Any]) -> str:
    return "::".join(
        [
            str(row.get("domain", "")),
            str(row.get("template_file", "")),
            str(row.get("template_name", "")),
        ]
    )


def _pairwise_overlap(sets: dict[str, set[Any]]) -> dict[str, int]:
    overlaps: dict[str, int] = {}
    for index, left in enumerate(SPLITS):
        for right in SPLITS[index + 1 :]:
            overlaps[f"{left}__{right}"] = len(sets[left] & sets[right])
    return overlaps


def _counter_by(rows: Iterable[dict[str, Any]], key: str) -> dict[str, int]:
    return dict(sorted(Counter(str(row.get(key)) for row in rows).items()))


def audit(data_dir: Path) -> dict[str, Any]:
    manifest_path = data_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    rows_by_split = {
        split: _load_jsonl(data_dir / f"{split}.jsonl")
        for split in SPLITS
    }
    row_counts = {split: len(rows_by_split[split]) for split in SPLITS}
    file_checksums = {
        f"{split}.jsonl": _sha256(data_dir / f"{split}.jsonl")
        for split in SPLITS
    }
    template_sets = {
        split: {_template_id(row) for row in rows_by_split[split]}
        for split in SPLITS
    }
    seed_sets = {
        split: {int(row["seed"]) for row in rows_by_split[split]}
        for split in SPLITS
    }
    template_overlap_counts = _pairwise_overlap(template_sets)
    seed_overlap_counts = _pairwise_overlap(seed_sets)
    split_template_counts = {split: len(template_sets[split]) for split in SPLITS}
    split_difficulty_row_counts = {
        split: _counter_by(rows_by_split[split], "difficulty")
        for split in SPLITS
    }
    split_domain_row_counts = {
        split: _counter_by(rows_by_split[split], "domain")
        for split in SPLITS
    }
    instances_per_template = manifest.get("instances_per_template", {})
    per_template_counts_ok = True
    per_template_counts: dict[str, dict[str, int]] = {}
    for split, rows in rows_by_split.items():
        counts = Counter(_template_id(row) for row in rows)
        per_template_counts[split] = dict(sorted(counts.items()))
        expected = int(instances_per_template.get(split, -1))
        per_template_counts_ok = per_template_counts_ok and all(
            count == expected for count in counts.values()
        )

    checks = {
        "strategy_matches": manifest.get("split_strategy") == EXPECTED_STRATEGY,
        "row_counts_match_expected": row_counts == EXPECTED_ROW_COUNTS,
        "row_counts_match_manifest": row_counts == manifest.get("split_row_counts"),
        "checksums_match_manifest": file_checksums == manifest.get("checksums_sha256"),
        "template_overlap_zero": all(value == 0 for value in template_overlap_counts.values()),
        "seed_overlap_zero": all(value == 0 for value in seed_overlap_counts.values()),
        "manifest_seed_overlap_zero": manifest.get("generated_seed_overlap_count") == 0,
        "difficulty_counts_match_manifest": (
            split_difficulty_row_counts == manifest.get("split_difficulty_row_counts")
        ),
        "domain_counts_match_manifest": (
            split_domain_row_counts == manifest.get("split_domain_row_counts")
        ),
        "template_counts_match_manifest": (
            split_template_counts == manifest.get("split_template_counts")
        ),
        "per_template_instance_counts_match_manifest": per_template_counts_ok,
    }
    result = {
        "ok": all(checks.values()),
        "data_dir": str(data_dir),
        "manifest_path": str(manifest_path),
        "manifest_checksum_sha256": _sha256(manifest_path),
        "split_paths": {split: str(data_dir / f"{split}.jsonl") for split in SPLITS},
        "checks": checks,
        "row_counts": row_counts,
        "split_template_counts": split_template_counts,
        "split_difficulty_row_counts": split_difficulty_row_counts,
        "split_domain_row_counts": split_domain_row_counts,
        "file_checksums_sha256": file_checksums,
        "template_overlap_counts": template_overlap_counts,
        "seed_overlap_counts": seed_overlap_counts,
        "instances_per_template": instances_per_template,
    }
    return result


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data/finchain_v2_template_disjoint"),
    )
    parser.add_argument("--out", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    result = audit(args.data_dir)
    text = json.dumps(result, indent=2, sort_keys=True)
    print(text)
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text + "\n", encoding="utf-8")
    if not result["ok"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
