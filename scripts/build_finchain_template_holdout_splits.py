"""Build a FinChain template-holdout split from an audited source catalog.

This script assigns each template function to exactly one split while
preserving the broad 40/40/20 basic/intermediate/advanced mix. It is the
assignment helper used by the fresh-seed v2 split builder.

Example:

    python scripts/build_finchain_template_holdout_splits.py \
      --source-dir data/archive/finchain_pilot \
      --output-dir data/archive/finchain_template_holdout \
      --seed 20260524
"""

from __future__ import annotations

import argparse
from pathlib import Path

from finpost.data.finchain_template_holdout import (
    SPLITS,
    build_template_holdout_splits,
    load_rows_from_jsonl,
    write_template_holdout_splits,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-dir",
        type=Path,
        default=Path("data/archive/finchain_pilot"),
        help="Directory containing train.jsonl, validation.jsonl, and test.jsonl.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/archive/finchain_template_holdout"),
        help="Directory where strict train/validation/test JSONLs and manifest.json are written.",
    )
    parser.add_argument("--seed", type=int, default=20260524)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    source_paths = {split: args.source_dir / f"{split}.jsonl" for split in SPLITS}
    rows_by_split = load_rows_from_jsonl(source_paths)
    split_rows, manifest = build_template_holdout_splits(rows_by_split, seed=args.seed)
    manifest["source_paths"] = {split: str(path) for split, path in source_paths.items()}
    manifest["output_dir"] = str(args.output_dir)
    write_template_holdout_splits(split_rows, manifest, args.output_dir)

    print(f"[finchain-template-holdout] wrote splits to {args.output_dir}")
    print("[finchain-template-holdout] row counts:", manifest["split_row_counts"])
    print("[finchain-template-holdout] template counts:", manifest["split_template_counts"])
    print("[finchain-template-holdout] template overlaps:", manifest["template_overlap_counts"])


if __name__ == "__main__":
    main()
