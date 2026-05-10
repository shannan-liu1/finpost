# 02 - Execute TinyGPT canary and Qwen SFT baseline ladder

- **Status:** Blocked
- **Ready for agent:** no
- **Depends on:** 01-confirm-preflight-and-keys, phase1-sft-trainer completion

## Goal

Run the shortest end-to-end production Supervised Fine-Tuning ladder:

1. production trainer readiness checks,
2. TinyGPT local infrastructure canary,
3. Qwen 0.5B soft launch,
4. full Qwen 0.5B Supervised Fine-Tuning baseline.

The full Qwen baseline stays blocked until TinyGPT and the Qwen 20-step soft launch both pass.

## Scope

**In scope:**
- production trainer readiness command,
- TinyGPT local canary command,
- Qwen 0.5B soft-launch command,
- full Qwen baseline launch,
- artifact checks for loss, validation loss, learning rate, gradient norm, tokens/sec, tracking, checkpointing, and resume.
- Colab T4 environment recording for Qwen runs: `nvidia-smi`, available VRAM, checkpoint persistence path, and runtime-reset notes.

**Out of scope:**
- ablation matrix,
- Direct Preference Optimization,
- evaluation harness implementation.

## Required trainer files

This issue is blocked until these planned files exist:

- `src/finpost/training/dataset.py`
- `src/finpost/training/optim.py`
- `src/finpost/training/checkpoint.py`
- `src/finpost/training/trainer.py`
- `src/finpost/training/train.py`
- `experiments/local_tiny_gpt2.yaml`
- `experiments/baseline.yaml`
- `tests/test_dataset.py`
- `tests/test_optim.py`
- `tests/test_checkpoint.py`
- `tests/test_trainer.py`

## Commands

Production trainer readiness:

```bash
pytest tests/test_dataset.py tests/test_config.py tests/test_optim.py tests/test_checkpoint.py tests/test_trainer.py -v
python -m finpost.training.train --help
```

TinyGPT local canary:

```bash
WANDB_MODE=offline \
python -m finpost.training.train \
  --config experiments/local_tiny_gpt2.yaml \
  --device cpu \
  --max-steps 20
```

PowerShell:

```powershell
$env:WANDB_MODE = "offline"
python -m finpost.training.train --config experiments/local_tiny_gpt2.yaml --device cpu --max-steps 20
```

TinyGPT resume check:

```bash
WANDB_MODE=offline \
python -m finpost.training.train \
  --config experiments/local_tiny_gpt2.yaml \
  --device cpu \
  --resume-from results/checkpoints/<run-name>/step-<N>.pt \
  --max-steps 25
```

Qwen 0.5B soft launch:

```bash
nvidia-smi
python -m finpost.training.train \
  --config experiments/baseline.yaml \
  --max-steps 20
```

Full Qwen baseline:

```bash
nvidia-smi
python -m finpost.training.train --config experiments/baseline.yaml
```

## Acceptance criteria

- Production trainer tests pass and `python -m finpost.training.train --help` exposes `--config`.
- TinyGPT canary completes on the local machine.
- TinyGPT run emits train loss, validation loss, learning rate, gradient norm, tokens/sec, offline tracking artifact path, and checkpoint path.
- TinyGPT checkpoint can be resumed with `--resume-from`, and continuation loss matches the deterministic tolerance owned by `.scratch/phase1-sft-trainer/issues/05-trainer.md`.
- Qwen soft launch completes 20 optimizer steps in the target environment.
- Qwen soft launch logs the same artifact classes as TinyGPT.
- Qwen run evidence includes Colab GPU type, available VRAM, and checkpoint persistence path.
- Full Qwen baseline is not launched until TinyGPT and Qwen soft launch evidence are linked.
- Cost gate checklist is completed before spend-bearing Qwen work.

## What this validates

- Production trainer readiness proves the planned trainer files and tests exist.
- TinyGPT validates infrastructure cheaply: loss computation, validation, logging, checkpoints, and resume.
- Qwen soft launch validates the real Phase 1 base model can train through the same path, with free Colab T4 as the default target environment.
- Full Qwen baseline produces the first real checkpoint eligible for evaluation and Direct Preference Optimization.

## Required proof artifacts

For each run, record:

- command,
- config path,
- git SHA,
- seed,
- run ID or offline tracking path,
- checkpoint path,
- checkpoint persistence path outside ephemeral Colab storage,
- GPU type and available VRAM from `nvidia-smi`,
- validation loss entry,
- tokens/sec entry,
- pass/fail result,
- failure root cause if failed.

## Comments

- 2026-05-07 (agent): Blocked by issue `01-confirm-preflight-and-keys`; package install/import was not yet successful, so smoke and baseline execution could not start.
- 2026-05-07 (agent): Local handoff scripts prepared: `scripts/local_phase1_bootstrap.sh` then `scripts/local_phase1_minitest.sh`.
- 2026-05-09 (agent): Updated execution order to TinyGPT local canary first, Qwen 0.5B soft launch second, full Qwen baseline third.
- 2026-05-09 (agent): Added production trainer readiness as an explicit blocker because this runbook should not imply `python -m finpost.training.train` exists before the SFT trainer workstream lands.
