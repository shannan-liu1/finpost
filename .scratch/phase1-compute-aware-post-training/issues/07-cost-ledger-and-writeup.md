# 07. Cost ledger and headline writeup (Stage 6)

- **Status:** Not Started
- **Ready for agent:** No (gated on issue 06)
- **Created:** 2026-05-11
- **Estimated time:** ~2 days
- **Depends on:** issue 06

## Goal

Combine the per-run cost JSONs and eval JSONs into the cost ledger, then write the one-page headline writeup that names the winner per axis.

## Scope

**In scope:**
- `src/finpost/postraining/cost_ledger.py` — schema validation and table assembly.
- `scripts/build_cost_report.py --run-glob 'results/compute_aware/*'` → markdown table at `results/compute_aware/cost_ledger.md`.
- Plot: accuracy vs. dollars and accuracy vs. GPU-hours, one line per method, two budget points each.
- `results/compute_aware/writeup.md`: one page. Names the winner on accuracy, on `accuracy / $`, on `accuracy / GPU-hour`. Lists failure cases. Identifies which prompts the adaptive method spent extra rollout tokens on that the uniform method did not.

**Out of scope:**
- Anything that would change earlier stage artifacts.
- A blog-post-length narrative — that lives in Phase 3.

## Acceptance criteria

1. `python scripts/build_cost_report.py --run-glob 'results/compute_aware/*'` produces `cost_ledger.md` with at least the ten rows from issue 06.
2. Each row contains: method, budget, base, rollout tokens, verifier calls, train tokens, GPU-hours, USD cost, GSM8K accuracy, MATH accuracy, pass@4, accuracy / $, accuracy / GPU-hour.
3. `results/compute_aware/writeup.md` exists and explicitly states whether method E (adaptive-compute OPD) beat methods A–D on `accuracy / GPU-hour` at both budgets, and on absolute accuracy at the medium budget.
4. The writeup references the per-prompt extra-rollout list so the reader can see *which* prompts adaptive spent on.
