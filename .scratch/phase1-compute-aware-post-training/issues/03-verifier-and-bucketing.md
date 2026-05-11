# 03. Verifier ladder and difficulty bucketing

- **Status:** Not Started
- **Ready for agent:** Yes (parser code can be developed without a real rollout cache; integration is gated on issue 02)
- **Created:** 2026-05-11
- **Estimated time:** ~2 days
- **Depends on:** issue 02 for integration; standalone for the verifier module itself

## Goal

Implement the cheapest-first verifier ladder for GSM8K and MATH, plus the bucketing module that turns rollout grades into `(prompt_id, p_correct, bucket, n_samples)` records.

## Scope

**In scope:**
- `src/finpost/postraining/verifier.py` with three layers:
  1. Exact answer parser (reuses existing GSM8K / MATH parsers from `phase1-data-loading`).
  2. Symbolic / numeric equivalence check (via `sympy` for MATH, integer / decimal tolerance for GSM8K).
  3. Optional small local verifier model — design only in this issue, not implemented unless layers 1–2 leave a meaningful unresolved fraction.
- `src/finpost/postraining/bucket.py` with the three-bucket assignment (`easy`, `ambiguous`, `hard`) and configurable thresholds.
- `scripts/run_bucketing.py` CLI: `--rollouts <cache>` → bucket file.
- Tests: positive and negative example per verifier layer; bucket counts on synthetic inputs.

**Out of scope:**
- LLM-as-judge integration. **Disallowed for numerical correctness in this workstream.**
- Preference-pair construction (issue 04).

## Acceptance criteria

1. `pytest tests/test_verifier.py` passes — exact parser matches/mismatches return 1/0; numeric tolerance accepts `1.00000` for `1`; symbolic check accepts `\\frac{1}{2}` for `0.5`.
2. `pytest tests/test_bucket.py` passes — `p_correct = 1.0` → `easy`, `p_correct = 0.5` → `ambiguous`, `p_correct = 0.0` → `hard`.
3. `python scripts/run_bucketing.py --rollouts <cache>` writes a bucket assignment file and prints per-bucket counts.
4. `grep -R "anthropic\\|openai" src/finpost/postraining/verifier.py` returns no matches.
5. The verifier module records a `verifier_calls` counter that the cost ledger reads.
