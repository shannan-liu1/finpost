"""Merge rollout-parallel DPO pair-generation shards.

Each `scripts/build_dpo_pairs.py --shard-id N --num-shards K` worker writes
its own directory containing:

    completions.jsonl
    pairs.jsonl
    manifest.json

This script combines those shard outputs into the same shape produced by a
single non-sharded run. It intentionally does not regenerate or re-score any
completion; the expensive rollout work stays embarrassingly parallel.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

try:
    from scripts.build_dpo_pairs import CompletionRecord, summarize_completion_groups
except ModuleNotFoundError:  # pragma: no cover - direct `python scripts/...` path
    from build_dpo_pairs import CompletionRecord, summarize_completion_groups


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as fp:
        for line_no, line in enumerate(fp, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            row = json.loads(stripped)
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_no} must be a JSON object")
            rows.append(row)
    return rows


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as fp:
        for row in rows:
            fp.write(json.dumps(row, ensure_ascii=True) + "\n")


def _load_manifest(shard_dir: Path) -> dict[str, Any]:
    manifest_path = shard_dir / "manifest.json"
    with manifest_path.open("r", encoding="utf-8") as fp:
        manifest = json.load(fp)
    if not isinstance(manifest, dict):
        raise ValueError(f"{manifest_path} must contain a JSON object")
    return manifest


def _sort_completions(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(
        rows,
        key=lambda row: (
            str(row.get("source", "")),
            str(row.get("prompt_id", "")),
            int(row.get("sample_index", -1)),
            str(row.get("completion", "")),
        ),
    )


def _sort_pairs(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(
        rows,
        key=lambda row: (
            str(row.get("source", "")),
            str(row.get("prompt_id", "")),
            json.dumps(row.get("chosen_grade", {}), sort_keys=True),
            json.dumps(row.get("rejected_grade", {}), sort_keys=True),
        ),
    )


def _completion_records(rows: list[dict[str, Any]]) -> list[CompletionRecord]:
    return [
        CompletionRecord(
            prompt_id=str(row["prompt_id"]),
            prompt=str(row["prompt"]),
            source=row["source"],
            gold_answer=str(row["gold_answer"]),
            sample_index=int(row["sample_index"]),
            completion=str(row["completion"]),
            predicted_answer=row.get("predicted_answer"),
            correct=bool(row["correct"]),
        )
        for row in rows
    ]


def merge_shards(
    *,
    shard_dirs: list[Path],
    out_dir: Path,
) -> dict[str, Any]:
    """Merge shard directories and return the merged manifest."""
    if not shard_dirs:
        raise ValueError("at least one shard directory is required")

    manifests: list[dict[str, Any]] = []
    completions: list[dict[str, Any]] = []
    pairs: list[dict[str, Any]] = []

    for shard_dir in shard_dirs:
        shard_dir = Path(shard_dir)
        manifests.append(_load_manifest(shard_dir))
        completions.extend(_read_jsonl(shard_dir / "completions.jsonl"))
        pairs.extend(_read_jsonl(shard_dir / "pairs.jsonl"))

    seen_shards = sorted(
        (
            int(manifest.get("shard_id", 0)),
            int(manifest.get("num_shards", 1)),
        )
        for manifest in manifests
    )
    num_shards_values = {num_shards for _, num_shards in seen_shards}
    if len(num_shards_values) != 1:
        raise ValueError(f"shards disagree on num_shards: {seen_shards}")
    expected_num_shards = next(iter(num_shards_values))
    shard_ids = [shard_id for shard_id, _ in seen_shards]
    if shard_ids != list(range(expected_num_shards)):
        raise ValueError(
            "provided shard ids do not cover the full expected range: "
            f"got {shard_ids}, expected {list(range(expected_num_shards))}"
        )

    completions = _sort_completions(completions)
    pairs = _sort_pairs(pairs)
    completion_records = _completion_records(completions)
    source_counts = Counter(record.source for record in completion_records)
    correct_count = sum(record.correct for record in completion_records)

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    completions_path = out_dir / "completions.jsonl"
    pairs_path = out_dir / "pairs.jsonl"
    manifest_path = out_dir / "manifest.json"

    _write_jsonl(completions_path, completions)
    _write_jsonl(pairs_path, pairs)

    first = manifests[0]
    manifest = {
        "created_at": datetime.now(UTC).isoformat(),
        "merge_of": [str(Path(path).resolve()) for path in shard_dirs],
        "model_checkpoint": first.get("model_checkpoint"),
        "verifier": first.get("verifier"),
        "sources": first.get("sources"),
        "source_counts": dict(sorted(source_counts.items())),
        "heldout_train_n": first.get("heldout_train_n"),
        "num_shards": expected_num_shards,
        "samples_per_prompt": first.get("samples_per_prompt"),
        "generation_batch_size": first.get("generation_batch_size"),
        "max_new_tokens": first.get("max_new_tokens"),
        "max_new_tokens_by_source": first.get("max_new_tokens_by_source"),
        "temperature": first.get("temperature"),
        "top_p": first.get("top_p"),
        "max_pairs_per_prompt": first.get("max_pairs_per_prompt"),
        "seed": first.get("seed"),
        "dtype": first.get("dtype"),
        "device": "merged-shards",
        "completion_count": len(completions),
        "correct_completion_count": correct_count,
        "incorrect_completion_count": len(completions) - correct_count,
        "pair_count": len(pairs),
        "group_summary": summarize_completion_groups(completion_records),
        "completions_path": str(completions_path),
        "pairs_path": str(pairs_path),
        "shard_manifests": [
            {
                "path": str((Path(shard_dir) / "manifest.json").resolve()),
                "manifest": manifest,
            }
            for shard_dir, manifest in zip(shard_dirs, manifests, strict=True)
        ],
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return manifest


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shard-dirs", nargs="+", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    manifest = merge_shards(shard_dirs=args.shard_dirs, out_dir=args.out_dir)
    print(f"[merge] wrote {manifest['completion_count']} completions")
    print(f"[merge] wrote {manifest['pair_count']} pairs")
    print(f"[merge] manifest: {args.out_dir / 'manifest.json'}")


if __name__ == "__main__":
    main()
