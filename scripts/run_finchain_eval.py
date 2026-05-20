"""FinChain exact-answer eval wrapper.

Usage:

    python scripts/run_finchain_eval.py \
        --checkpoints base=Qwen/Qwen2.5-1.5B \
        --n 200 \
        --out-dir results/evals/finchain_base

Set FINPOST_FINCHAIN_TEST_JSONL to an audited local FinChain JSONL export
before running. The wrapper delegates to finpost.evals.eval_exact with the
source fixed to ``finchain``.
"""

from __future__ import annotations

import sys

from finpost.evals.eval_exact import main as eval_exact_main


def main(argv: list[str] | None = None) -> None:
    args = list(sys.argv[1:] if argv is None else argv)
    if "--sources" not in args:
        args.extend(["--sources", "finchain"])
    eval_exact_main(args)


if __name__ == "__main__":
    main()
