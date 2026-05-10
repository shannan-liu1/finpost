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
  - Parses `--config` (required), `--device`, `--max-steps` (override), `--resume-from` (override).
  - Loads `Config.from_yaml(args.config)`. Applies CLI overrides (CLI flags win over YAML values).
  - Prints the effective config + the steps↔epochs estimate at startup.
  - Constructs and runs `Trainer(config).train()`.
- `experiments/baseline.yaml` — the reference Phase 1 SFT config. Reasonable defaults for `Qwen/Qwen2.5-0.5B` on combined GSM8K + MATH in the target environment. Documented inline.
- A two-paragraph addition to the project `README.md` showing how to launch a baseline run.

**Out of scope:**
- Actually running the full Qwen baseline. This issue verifies the TinyGPT local canary and Qwen 20-step soft launch; the full run is operational, not engineering.
- Multiple baseline configs for the ablation matrix. Those are produced in a downstream workstream that uses this trainer.

## Deliverables

```
src/finpost/training/train.py    # __main__ entry point
experiments/baseline.yaml         # reference config
```

Plus README.md update.

## Acceptance criteria

These original acceptance criteria are superseded by the 2026-05-09 amendment below. The closing check for this issue is the local TinyGPT config plus the Qwen 20-step soft launch, not a `--tiny-model` override on `baseline.yaml`.

See "Updated acceptance criteria" under the amendment.

## Baseline config sketch

The actual `experiments/baseline.yaml` should look roughly like:

```yaml
model:
  base_model_id: Qwen/Qwen2.5-0.5B
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
  run_name: null  # auto-generated: <qwen2.5-0.5b>-<lr>-<seed>-<timestamp>

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
## Amendment 2026-05-09 - Local model ladder before Qwen baseline

The closing issue now has a two-step model ladder:

1. **TinyGPT local canary:** run the production trainer path with `sshleifer/tiny-gpt2` on CPU. This is the 4 GB local-machine check. It must validate loss measurement, validation loss, offline tracking, checkpoint save/load, and resume before any Qwen run.
2. **Qwen 0.5B soft launch:** run `Qwen/Qwen2.5-0.5B` for a short 20-step SFT soft launch in the target environment. This checks that the real Phase 1 model connects to the same trainer path.

Additional deliverable:

```
experiments/local_tiny_gpt2.yaml  # local canary config
```

Updated acceptance criteria:

1. `WANDB_MODE=offline python -m finpost.training.train --config experiments/local_tiny_gpt2.yaml --device cpu --max-steps 20` runs to completion on the local machine and prints/logs a loss curve. PowerShell equivalent: set `$env:WANDB_MODE="offline"` before the command.
2. The TinyGPT run writes offline tracking artifacts for `train/loss`, `train/lr`, `train/grad_norm`, `val/loss`, and `train/tokens_per_sec`.
3. Final TinyGPT checkpoint exists at `results/checkpoints/<run_name>/step-00000020/` with both `model.safetensors` and `state.pt`.
4. The same TinyGPT command run twice with the same `data.seed` produces identical loss values per step within the tolerance defined in issue 05.
5. Killing the TinyGPT run mid-way and resuming with `--resume-from` continues with the same trajectory.
6. `python -m finpost.training.train --config experiments/baseline.yaml --max-steps 20` launches a Qwen 0.5B soft launch in the target environment and writes loss/tracking/checkpoint artifacts.
7. Full Qwen 0.5B SFT is explicitly out of scope until the TinyGPT canary and Qwen 20-step soft launch both pass.

`experiments/local_tiny_gpt2.yaml` sketch:

```yaml
model:
  base_model_id: sshleifer/tiny-gpt2
  dtype: float32
  use_safetensors: false  # tiny-gpt2 ships pickle weights; local canary only

data:
  sources: [gsm8k, math]
  val_split_pct: 5.0
  seed: 42

training:
  max_steps: 20
  warmup_steps: 1
  lr: 1.0e-4
  weight_decay: 0.01
  grad_accum_steps: 1
  grad_clip: 1.0
  val_every_n_steps: 5
  checkpoint_every_n_steps: 10
  per_device_batch_size: 2

packing:
  max_seq_len: 128
  isolate_documents: true

logging:
  wandb_project: finpost-phase1-local
  run_name: tiny-gpt2-local-canary

checkpointing:
  save_dir: results/checkpoints
  retention_last_n: 2
  resume_from: null
```

Run the local canary with `WANDB_MODE=offline`. Do not add `logging.mode` to the YAML unless issue 01 is intentionally reopened and the config schema is updated.
