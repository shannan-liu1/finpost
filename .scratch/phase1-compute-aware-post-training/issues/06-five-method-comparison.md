# 06. Five-method comparison at fixed budgets (Stage 5)

- **Status:** Not Started
- **Ready for agent:** No (gated on issues 01–05)
- **Created:** 2026-05-11
- **Estimated time:** ~1 week of execution time across two GPU budgets
- **Depends on:** issues 01, 02, 03, 04, 05

## Goal

Run the five-method comparison at two preset budgets (small and medium) and produce the comparison table.

## Methods

- **A. uniform SFT.** Stage 0 baseline. Cost reference.
- **B. rejection SFT.** SFT on chosen-only completions.
- **C. uniform OPD.** OPD with `train_weight = 1.0` for all pairs.
- **D. verifier-weighted OPD.** OPD with default bucket schedule `{easy: 0.25, ambiguous: 1.0, hard: 0.5}`.
- **E. adaptive-compute OPD.** D plus Stage 2 extra rollouts on ambiguous prompts.

## Scope

**In scope:**
- One config per method under `experiments/compute_aware/`.
- One **small-budget** run (≈30 min on a single A100, ~$1) and one **medium-budget** run (≈2 hours on a single A100, ~$4) per method. H100 is an acceptable substitute for either; same dollar envelope, ~2× faster wall-clock.
- Combined ten-run total spend target: **< $25**, matching the llm.c GPT-2 124M reproduction envelope.
- Eval at the end of every run on the held-out GSM8K and MATH test sets, plus pass@4.
- All evaluation uses the same harness as Phase 1 SFT.

**Out of scope:**
- The cost-ledger report builder and writeup (issue 07).

## Acceptance criteria

1. Ten runs complete: 5 methods × 2 budgets. Each has a checkpoint, an eval JSON, and a `cost_gate.md`.
2. Each run records `rollout_tokens`, `verifier_calls`, `train_tokens`, `gpu_hours`, `usd_cost` in its `results/<run_name>/cost.json`.
3. Bootstrapped 95% confidence intervals (10K resamples, paired where comparing two checkpoints on the same prompts) on every per-method accuracy figure.
4. A short results note `results/compute_aware/stage5_results.md` lists the per-method numbers and flags which arms beat the baseline at p < 0.05.
