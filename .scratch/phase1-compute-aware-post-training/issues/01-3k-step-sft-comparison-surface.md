# 01. 3K-step SFT comparison surface (Stage 0)

- **Status:** Not Started
- **Ready for agent:** No (gated on `phase1-sft-trainer` reaching the "first Qwen 0.5B SFT baseline" milestone)
- **Created:** 2026-05-11
- **Estimated time:** ~1 day of execution time, dependent on chosen GPU
- **Depends on:** [`phase1-sft-trainer`](../../phase1-sft-trainer/PRD.md), [`phase1-training-runbook`](../../phase1-training-runbook/PRD.md)

## Goal

Produce the three reference SFT checkpoints that the rest of this workstream measures against: `gsm8k_only`, `math_only`, and `combined`. Each is trained on `Qwen/Qwen2.5-0.5B` for 3,000 optimizer steps using the existing Phase 1 trainer.

## Scope

**In scope:**
- Three configs under `experiments/compute_aware/`.
- `eval_every_n_steps = 500` and `checkpoint_every_n_steps = 500` on every config.
- Best-checkpoint selection by combined validation accuracy, not by final step.
- A summary report `results/compute_aware/stage0_summary.md` naming the winner per arm and the per-checkpoint accuracy curve.

**Out of scope:**
- The rollout pipeline (issue 02).
- Any preference-pair work (issue 04).

## Acceptance criteria

1. `python -m finpost.training.train --config experiments/compute_aware/sft_gsm8k_only_3k.yaml` runs to completion, writes 6 checkpoints, and emits an evaluation curve.
2. Same for `sft_math_only_3k.yaml` and `sft_combined_3k.yaml`.
3. `results/compute_aware/stage0_summary.md` lists the best step per arm and the chosen substrate for Stage 1.
4. A cost-gate checklist (`.scratch/templates/cost-gate-checklist.md`) is filled in and committed for each of the three runs under `results/<run_name>/cost_gate.md`.

## Decision rule (from PRD)

- train loss falling and eval accuracy rising at step 3,000 → schedule one 5,000-step run on the winning arm.
- train loss falling and eval accuracy flat or dropping → stop, log as overfit or data-mix issue.
- specialist arm beats combined on its own test set but combined beats specialist averaged → keep combined as the policy substrate for Stage 1.
