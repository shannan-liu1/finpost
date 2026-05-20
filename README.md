# finpost

A notebook-first learning project for building an end-to-end post-training stack and applying it to verifiable financial reasoning.

The active direction is **FinChain-first RLVR**: use FinChain's executable financial reasoning chains as the primary substrate for Supervised Fine-Tuning (SFT), rejection SFT, On-Policy Distillation (OPD), and Group Relative Policy Optimization (GRPO), then test transfer on FinQA.

For project intent, glossary, and target capability, see [`CONTEXT.md`](./CONTEXT.md).
For the executable phase plan and current decisions, see [`PLAN.md`](./PLAN.md).
For the professor-style study guide, see [`STUDY.md`](./STUDY.md) or open [`STUDY.html`](./STUDY.html).
For the study flow, see [`docs/runbooks/finchain-rlvr-study-flow.md`](./docs/runbooks/finchain-rlvr-study-flow.md).
For distributed training and platform choices, see [`docs/distributed-training-and-platforms.md`](./docs/distributed-training-and-platforms.md) or open [`docs/distributed-training-and-platforms.html`](./docs/distributed-training-and-platforms.html).
For specific workstreams, see [`.scratch/`](./.scratch/).
For supply-chain posture and security policy, see [`SECURITY.md`](./SECURITY.md).

## Active Study Shape

The project is organized around a small number of high-leverage artifacts:

1. FinChain dataset and verifier notebook
2. FinChain LoRA/QLoRA SFT notebook
3. rollout cache, bucketing, and cost ledger notebook
4. rejection SFT and OPD notebook
5. one controlled GRPO notebook
6. final comparison notebook with FinQA transfer
7. study guide tying SFT, DPO, OPD, PPO, GRPO, RLHF, RLVR, KL control, and reward hacking back to repo code

The default serious model is `Qwen/Qwen2.5-1.5B` for faster FinChain iteration. `Qwen/Qwen2.5-0.5B` remains the local canary and cheap trainer regression model.

New RLVR notebook scaffolds live under `notebooks/finchain_*.ipynb`. Existing SFT and DPO notebooks are preserved as phase artifacts.

## Run A Training Job

The Phase 1 Supervised Fine-Tuning trainer is wired behind one entry point: `python -m finpost.training.train --config <path>`. For a fast local sanity check that exercises the production path on CPU (tiny-gpt2, real GSM8K + MATH data, packing collator, validation, checkpointing), run the canary config:

```bash
# PowerShell:
$env:WANDB_MODE = "offline"
.venv/Scripts/python.exe -m finpost.training.train --config experiments/local_tiny_gpt2.yaml --device cpu

# Bash:
WANDB_MODE=offline python -m finpost.training.train --config experiments/local_tiny_gpt2.yaml --device cpu
```

The old Qwen2.5-0.5B SFT path remains useful for infrastructure checks. The next finance experiments should use the FinChain study flow rather than expanding the 0.5B benchmark grid.

## Install

This project targets Python 3.11 or newer.

```bash
# Create and activate a virtual environment (any tool works; venv shown).
python -m venv .venv
source .venv/bin/activate            # macOS / Linux
.venv\Scripts\activate               # Windows PowerShell

# Install the package in editable mode with the dev extras.
pip install -e ".[dev]"

# Verify the install.
python -c "import finpost; print(finpost.__version__)"
```

Optional extras for later phases:

```bash
pip install -e ".[peft]"             # LoRA/QLoRA and bitsandbytes on GPU boxes
pip install -e ".[dpo-reference]"    # validates our DPO loss against TRL
pip install -e ".[edgar]"            # SEC filing tooling for later transfer work
```

## Layout

```text
finpost/
|-- CONTEXT.md          # ubiquitous language, glossary, project intent
|-- PLAN.md             # active FinChain-first RLVR plan
|-- .scratch/           # one PRD per workstream
|-- src/finpost/        # library code
|-- tests/              # pytest tests
|-- scripts/            # runnable scripts and one-off jobs
|-- experiments/        # configuration files for training runs
|-- notebooks/          # notebook-first experiment surface
|-- data/               # raw and processed data (gitignored)
|-- results/            # checkpoints and eval outputs (gitignored)
`-- docs/               # runbooks, ADRs, and longer-form documentation
```

## Status

Active. The current planning workstream is [`.scratch/finchain-rlvr-posttraining/PRD.md`](./.scratch/finchain-rlvr-posttraining/PRD.md).
