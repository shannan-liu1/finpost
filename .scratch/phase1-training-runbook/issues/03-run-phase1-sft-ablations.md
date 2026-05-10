# 03 - Run Phase 1 SFT ablations

- **Status:** Not Started
- **Ready for agent:** no
- **Depends on:** 02-execute-sft-smoke-and-baseline

## Goal

Execute planned Qwen 0.5B Supervised Fine-Tuning ablation cells and record comparable outputs after the TinyGPT canary, Qwen soft launch, and first full Qwen Supervised Fine-Tuning baseline have passed.

## Scope

**In scope:**
- per-cell config files,
- per-cell launches,
- run tracking,
- checkpoint paths,
- consolidated metrics table,
- best-checkpoint selection for evaluation and Direct Preference Optimization preparation.

**Out of scope:**
- Direct Preference Optimization training,
- Phase 2 finance data,
- expanding the grid after failures without owner review.

## Planned axes

From `PLAN.md`:

- data scale: 10%, 50%, 100%,
- learning rate: `1e-5`, `5e-5`, `1e-4`,
- training budget: the step-equivalent or epoch-equivalent convention used by the production config.

## Command template

```bash
python -m finpost.training.train --config experiments/<ablation-config>.yaml
```

## Acceptance criteria

- No ablation run starts until issue 02 soft-launch evidence is linked.
- Cost gate checklist is completed before the ablation batch.
- Each planned cell has a recorded run ID, config path, seed, data scale, learning rate, training budget, wall-clock time, final validation loss, and checkpoint path.
- Every cell starts from `Qwen/Qwen2.5-0.5B` and uses the same production trainer path validated by issue 02.
- Failures are tagged with root cause.
- At least one best checkpoint candidate is selected for evaluation and Direct Preference Optimization preparation.

## What this validates

This validates sensitivity to data scale and learning rate after the trainer substrate is proven. It does not validate Direct Preference Optimization.

## Required proof artifacts

- Config file per launched cell.
- Tracking run ID per launched cell.
- Checkpoint path per completed cell.
- Consolidated ablation table under `results/` or copied into this issue.
- Best-checkpoint selection note with rationale.

## Stop conditions

- Stop expanding the grid if loss diverges, NaNs appear, checkpoints fail to write, or throughput is below the owner-approved floor.
- Stop if cumulative spend reaches the cost gate cap.
- Stop if the first Qwen baseline has not produced a usable checkpoint.
