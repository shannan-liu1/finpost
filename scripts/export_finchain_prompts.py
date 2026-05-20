"""Export FinChain prompt-only JSONL for TRL/Axolotl-style RLVR trainers.

Example:

    python scripts/export_finchain_prompts.py \
      --split train \
      --n 2000 \
      --out-path results/finchain_rlvr/prompts_train_2000.jsonl

Set FINPOST_FINCHAIN_TRAIN_JSONL, FINPOST_FINCHAIN_VALIDATION_JSONL, or
FINPOST_FINCHAIN_TEST_JSONL before running if the audited export is not under
data/finchain/{split}.jsonl.
"""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

from finpost.data.finchain import load_finchain, resolve_finchain_path
from finpost.posttraining.finchain_rlvr import (
    build_finchain_prompt_rows,
    deterministic_sample,
    write_jsonl,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=["train", "validation", "test"], default="train")
    parser.add_argument("--out-path", type=Path, required=True)
    parser.add_argument("--n", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--raw-prompt",
        action="store_true",
        help="Write the raw FinChain problem text instead of the repo training prompt format.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    examples = load_finchain(args.split)
    selected = deterministic_sample(examples, n=args.n, seed=args.seed)
    rows = build_finchain_prompt_rows(selected, format_prompt=not args.raw_prompt)
    write_jsonl(args.out_path, rows)

    manifest = {
        "created_at": datetime.now(UTC).isoformat(),
        "split": args.split,
        "source_path": str(resolve_finchain_path(args.split)),
        "out_path": str(args.out_path),
        "n": len(rows),
        "seed": args.seed,
        "format_prompt": not args.raw_prompt,
        "columns": list(rows[0]) if rows else [],
    }
    args.out_path.with_suffix(".manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(f"[finchain-prompts] wrote {len(rows)} rows to {args.out_path}")


if __name__ == "__main__":
    main()
