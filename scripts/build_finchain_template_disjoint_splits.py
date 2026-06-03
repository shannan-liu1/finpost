"""Generate fresh-seed template-disjoint FinChain splits.

The input ``data/archive/finchain_pilot`` JSONLs provide the audited catalog of
290 upstream template functions and their difficulty/domain metadata. The pinned
upstream FinChain clone provides executable template functions that are called
under new deterministic seeds.

Example:

    python scripts/build_finchain_template_disjoint_splits.py \
      --source-dir data/archive/finchain_pilot \
      --upstream-root /workspace/finchain \
      --output-dir data/finchain_v2_template_disjoint
"""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

from finpost.data.finchain_template_holdout import SPLITS, load_rows_from_jsonl
from finpost.data.finchain_template_disjoint_builder import (
    DEFAULT_INSTANCES_PER_TEMPLATE,
    build_fresh_template_holdout_splits,
    write_fresh_template_holdout_splits,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-dir",
        type=Path,
        default=Path("data/archive/finchain_pilot"),
        help="Audited FinChain catalog JSONLs used to identify template functions.",
    )
    parser.add_argument(
        "--upstream-root",
        type=Path,
        required=True,
        help="Checked-out mbzuai-nlp/finchain repository with data/templates.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/finchain_v2_template_disjoint"),
        help="Destination for train.jsonl, validation.jsonl, test.jsonl, and manifest.json.",
    )
    parser.add_argument("--template-assignment-seed", type=int, default=20260524)
    parser.add_argument("--generation-seed", type=int, default=20260525)
    parser.add_argument("--train-instances-per-template", type=int, default=10)
    parser.add_argument("--validation-instances-per-template", type=int, default=30)
    parser.add_argument("--test-instances-per-template", type=int, default=30)
    parser.add_argument(
        "--upstream-revision",
        default=None,
        help="Pinned upstream revision. Defaults to git rev-parse HEAD under --upstream-root.",
    )
    return parser.parse_args()


def _resolve_upstream_revision(upstream_root: Path, explicit_revision: str | None) -> str:
    if explicit_revision:
        return explicit_revision
    try:
        return subprocess.check_output(
            ["git", "-C", str(upstream_root), "rev-parse", "HEAD"],
            text=True,
        ).strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise SystemExit(
            "Could not read upstream git revision; pass --upstream-revision explicitly."
        ) from exc


def main() -> None:
    args = _parse_args()
    source_paths = {split: args.source_dir / f"{split}.jsonl" for split in SPLITS}
    rows_by_split = load_rows_from_jsonl(source_paths)
    revision = _resolve_upstream_revision(args.upstream_root, args.upstream_revision)
    instance_counts = {
        "train": args.train_instances_per_template,
        "validation": args.validation_instances_per_template,
        "test": args.test_instances_per_template,
    }
    split_rows, manifest = build_fresh_template_holdout_splits(
        rows_by_split,
        upstream_root=args.upstream_root,
        template_assignment_seed=args.template_assignment_seed,
        generation_seed=args.generation_seed,
        instances_per_template=instance_counts,
        upstream_revision=revision,
    )
    manifest["source_paths"] = {split: str(path) for split, path in source_paths.items()}
    manifest["output_dir"] = str(args.output_dir)
    manifest["default_instance_counts"] = DEFAULT_INSTANCES_PER_TEMPLATE
    manifest = write_fresh_template_holdout_splits(split_rows, manifest, args.output_dir)

    print(f"[finchain-template-disjoint] upstream revision: {revision}")
    print(f"[finchain-template-disjoint] wrote splits to {args.output_dir}")
    print("[finchain-template-disjoint] row counts:", manifest["split_row_counts"])
    print("[finchain-template-disjoint] template counts:", manifest["split_template_counts"])
    print("[finchain-template-disjoint] template overlaps:", manifest["template_overlap_counts"])
    print(
        "[finchain-template-disjoint] generated seed overlap:",
        manifest["generated_seed_overlap_count"],
    )


if __name__ == "__main__":
    main()
