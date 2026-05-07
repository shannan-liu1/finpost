# finpost

A learning project: build an end-to-end post-training stack — Supervised Fine-Tuning, Direct Preference Optimization, evaluation under noise — and apply it to numerical reasoning over United States Securities and Exchange Commission filings.

For project intent, glossary, and target capability, see [`CONTEXT.md`](./CONTEXT.md).
For the executable plan with phase breakdown and open decisions, see [`PLAN.md`](./PLAN.md).
For specific workstreams, see [`.scratch/`](./.scratch/).
For supply-chain posture and security policy, see [`SECURITY.md`](./SECURITY.md).

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
pip install -e ".[peft]"             # Phase 2: LoRA + bitsandbytes (Linux/A100 box)
pip install -e ".[dpo-reference]"    # validates our DPO loss against TRL
pip install -e ".[edgar]"            # SEC filing tooling
```

## Layout

```
finpost/
├── CONTEXT.md          # ubiquitous language: glossary and project intent
├── PLAN.md             # phase plan and open decisions
├── .scratch/           # one PRD per workstream (`<slug>/PRD.md`)
├── src/finpost/        # library code
├── tests/              # pytest tests
├── scripts/            # runnable scripts (smoke tests, one-off jobs)
├── experiments/        # configuration files for training runs
├── notebooks/          # exploratory only — not load-bearing
├── data/               # raw and processed data (gitignored)
├── results/            # checkpoints and eval outputs (gitignored)
└── docs/               # longer-form documentation
```

## Status

Active. See [`.scratch/README.md`](./.scratch/README.md) for the in-flight workstreams.
