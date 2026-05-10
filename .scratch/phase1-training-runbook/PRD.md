# Phase 1 command-by-command training runbook and execution issues

- **Status:** In Progress
- **Created:** 2026-05-07
- **Owner:** Shannan
- **Estimated time:** ~1-2 days for docs + issue prep, ~1 week execution
- **Depends on:** [`phase1-data-loading`](../phase1-data-loading/PRD.md), [`phase1-sft-trainer`](../phase1-sft-trainer/PRD.md)

## Goal

Produce an operator-grade, command-by-command runbook for Phase 1:

1. local setup,
2. data and smoke checks,
3. production trainer readiness,
4. TinyGPT local canary,
5. `Qwen/Qwen2.5-0.5B` Supervised Fine-Tuning soft launch,
6. full Qwen Supervised Fine-Tuning baseline,
7. Qwen Supervised Fine-Tuning ablations,
8. evaluation,
9. Direct Preference Optimization pair preparation, loss sanity, training, and comparison.

The runbook is for operators and learners. Each gate must explain what it validates, why it exists in the TinyGPT -> Qwen 0.5B -> Supervised Fine-Tuning -> Direct Preference Optimization path, and which artifacts prove the gate passed.

## Scope

**In scope:**
- A single canonical runbook file with exact shell commands, expected artifacts, and pass/fail checkpoints.
- Conversion of runbook stages into execution issues under `.scratch/phase1-training-runbook/issues`.
- Explicit decision gates where human input is required before expensive or irreversible runs.
- A cheap local model ladder: `sshleifer/tiny-gpt2` first for a 4 GB CPU canary, then `Qwen/Qwen2.5-0.5B` for the real Phase 1 soft launch.
- Google Colab T4 as the default target environment for Qwen soft launches and first baselines, with paid remote GPU kept as fallback.
- Evidence ledgers for runs: command, config, git SHA, seed, run ID, checkpoint path, log path, and pass/fail notes.
- Clear boundaries between built code, planned trainer entry points, and blocked future commands.

**Out of scope:**
- Implementing new training code; this workstream documents and orchestrates existing or planned components.
- Phase 2 finance-domain execution; this runbook is only for Phase 1.
- Group Relative Policy Optimization. That is a separate research track after Supervised Fine-Tuning and Direct Preference Optimization are evaluated.

## Deliverables

- `docs/runbooks/phase1-training-runbook.md`
  - Prerequisites and environment verification commands.
  - Data loading, schema, safety, masking, and smoke-check commands.
  - Production trainer readiness gate.
  - TinyGPT local Supervised Fine-Tuning canary command.
  - Qwen 0.5B Supervised Fine-Tuning soft-launch command.
  - Full Qwen baseline command gated behind both soft launches.
  - Ablation execution matrix commands and artifact ledger.
  - Evaluation handoff criteria.
  - Direct Preference Optimization prep, loss parity, soft-launch, and comparison gates.
- `.scratch/phase1-training-runbook/issues/01-confirm-preflight-and-keys.md`
- `.scratch/phase1-training-runbook/issues/02-execute-sft-smoke-and-baseline.md`
- `.scratch/phase1-training-runbook/issues/03-run-phase1-sft-ablations.md`
- `.scratch/phase1-training-runbook/issues/04-prepare-dpo-pairs-and-loss-sanity.md`
- `.scratch/phase1-training-runbook/issues/05-decision-gates-and-signoff.md`

## Acceptance criteria

1. `docs/runbooks/phase1-training-runbook.md` exists and includes commands or explicitly marked future commands for each Phase 0 and Phase 1 gate.
2. Every issue has: `Status`, `Ready for agent`, `Goal`, `Scope`, `Acceptance criteria`, and explicit dependencies.
3. All unresolved decisions from Plan Q-B/Q-C and operational choices in this runbook are captured in issue `05-decision-gates-and-signoff.md`.
4. The runbook references exact project files where they exist and labels not-yet-built entry points as blocked or future commands.
5. The runbook states that full Qwen Supervised Fine-Tuning and Direct Preference Optimization execution are blocked until the TinyGPT canary and Qwen 20-step soft launch pass.
6. Each gate states what it validates and which artifacts prove it passed.
7. Direct Preference Optimization gates require a real Qwen Supervised Fine-Tuning checkpoint and cannot start from TinyGPT or the base model.
8. Qwen gates record Colab GPU type, available VRAM, checkpoint persistence path, and fallback criteria for paid GPU.

## Amendment - 2026-05-09

The canonical path is now TinyGPT local infrastructure canary -> `Qwen/Qwen2.5-0.5B` Supervised Fine-Tuning soft launch -> full Qwen Supervised Fine-Tuning baseline -> Qwen Supervised Fine-Tuning ablations -> Direct Preference Optimization from the best real Qwen Supervised Fine-Tuning checkpoint.

`Qwen/Qwen2.5-0.5B-Instruct` is a reference baseline only, not the Phase 1 training substrate.

Qwen execution should default to free Google Colab T4 while the model and experiments are small. Paid GPU rental is reserved for cases where Colab availability, runtime resets, or Direct Preference Optimization memory pressure prevents useful progress.

## Notes / open questions

- This workstream intentionally separates documentation/orchestration from trainer implementation so execution can proceed with less ambiguity.
- Any command that is expected to run outside the local 4 GB canary environment is labeled clearly to avoid accidental local execution.
