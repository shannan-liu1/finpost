# 06. CLI entry point + baseline config + acceptance verification

- **Status:** Not Started
- **Created:** 2026-05-06
- **Estimated time:** ~2 hours
- **Depends on:** [`05-trainer`](./05-trainer.md)

## Goal

Wire everything from issues 01–05 behind a single `python -m finpost.training.train --config <path>` command, ship the reference baseline YAML, and run the PRD's acceptance criteria end-to-end.

## Scope

**In scope:**
- A `src/finpost/training/train.py` `__main__` entry point that:
  - Parses `--config` (required), `--tiny-model`, `--device`, `--max-steps` (override), `--resume-from` (override).
  - Loads `Config.from_yaml(args.config)`. Applies CLI overrides (CLI flags win over YAML values).
  - Prints the effective config + the steps↔epochs estimate at startup.
  - Constructs and runs `Trainer(config).train()`.
- `experiments/baseline.yaml` — the reference Phase 1 SFT config. Reasonable defaults for Gemma 3 1B on combined GSM8K + MATH on a single A100. Documented inline.
- A two-paragraph addition to the project `README.md` showing how to launch a baseline run.

**Out of scope:**
- Actually running on the rented A100. The acceptance criteria in this issue verify the *tiny-model* run succeeds; the real-A100 run is operational, not engineering.
- Multiple baseline configs for the ablation matrix. Those are produced in a downstream workstream that uses this trainer.

## Deliverables

```
src/finpost/training/train.py    # __main__ entry point
experiments/baseline.yaml         # reference config
```

Plus README.md update.

## Acceptance criteria

These are the PRD's acceptance criteria, instantiated here for verification:

1. `python -m finpost.training.train --config experiments/baseline.yaml --tiny-model --device cpu --max-steps 20` runs to completion in under 2 minutes and prints a loss curve that decreases from ~10 (≈ ln(vocab_size) for tiny-gpt2) to noticeably lower.
2. The same command run twice with the same `data.seed` produces identical loss values per step.
3. Killing the run mid-way and resuming with `--resume-from` continues with the same trajectory (validated already in issue 05's acceptance).
4. The wandb run page for the tiny-model run contains all expected curves: `train/loss`, `train/lr`, `train/grad_norm`, `val/loss`. (Run with `WANDB_MODE=offline` if you don't want to push to the cloud; the local files still validate the keys.)
5. Final checkpoint exists at `results/checkpoints/<run_name>/step-00000020/` with both `model.safetensors` and `state.pt`.

## Baseline config sketch

The actual `experiments/baseline.yaml` should look roughly like:

```yaml
model:
  base_model_id: google/gemma-3-1b-it
  dtype: bfloat16
  use_safetensors: true

data:
  sources: [gsm8k, math]
  val_split_pct: 5.0
  seed: 42

training:
  max_steps: 3000
  warmup_steps: 100
  lr: 2.0e-5
  weight_decay: 0.01
  grad_accum_steps: 4
  grad_clip: 1.0
  val_every_n_steps: 250
  checkpoint_every_n_steps: 500
  per_device_batch_size: 8

packing:
  max_seq_len: 4096
  isolate_documents: true

logging:
  wandb_project: finpost-phase1
  run_name: null  # auto-generated: <gemma-3-1b-it>-<lr>-<seed>-<timestamp>

checkpointing:
  save_dir: results/checkpoints
  retention_last_n: 3
  resume_from: null
```

Numbers chosen as starting points — the Phase 1 ablation matrix (downstream workstream) will vary `lr`, `max_steps`, and the dataset composition.

## Notes

- This issue is the closing-the-loop piece. Once all 6 issues are Done, the parent PRD's status flips to Done.
- After landing this issue, the natural next workstream is **`phase1-sft-ablation-matrix`**: actually running the matrix from PLAN.md section 1.3 against the trainer this PRD produces.
- Don't add `pip install` instructions to README — the existing README already covers it.
