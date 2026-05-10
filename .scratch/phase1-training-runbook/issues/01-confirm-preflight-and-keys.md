# 01 - Confirm preflight, local install, and credentials

- **Status:** In Progress (TODO: execute on local machine when available)
- **Ready for agent:** no
- **Depends on:** none

## Goal

Validate local environment and access prerequisites before any training job. This issue covers runbook Gates 0 and 1.

## Scope

**In scope:**
- local Python version check,
- editable install,
- package import,
- core tests,
- local data/masking/safety/smoke checks,
- credential presence checks for later Hugging Face and tracking usage.

**Out of scope:**
- production trainer launch,
- TinyGPT or Qwen training,
- spend-bearing remote work.

## Commands

Preferred local path:

```bash
./scripts/local_phase1_bootstrap.sh
./scripts/local_phase1_minitest.sh
```

Manual PowerShell equivalent:

```powershell
python --version
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install -e ".[dev]"
python -c "import finpost; print(finpost.__version__)"
python -m pytest tests/test_config.py tests/test_data_schema.py tests/test_masking.py tests/test_cli_stats.py -v
python -m finpost.data.cli stats --help
python scripts/sft_smoke.py --help
python scripts/sft_smoke.py
```

Credential checks before remote work:

```bash
python -c "import os; print('HF_TOKEN', bool(os.getenv('HF_TOKEN') or os.getenv('HUGGING_FACE_HUB_TOKEN')))"
python -c "import os; print('WANDB_API_KEY', bool(os.getenv('WANDB_API_KEY')))"
```

## Acceptance criteria

- `python --version` reports a runtime compatible with `pyproject.toml` (`>=3.11`).
- `pip install -e ".[dev]"` succeeds.
- `python -c "import finpost; print(finpost.__version__)"` succeeds.
- `pytest tests/test_config.py tests/test_data_schema.py tests/test_masking.py tests/test_cli_stats.py -v` passes.
- `python -m finpost.data.cli stats --help` succeeds.
- `python scripts/sft_smoke.py --help` and `python scripts/sft_smoke.py` succeed.
- Credential presence is recorded before any remote Qwen or tracking run.

## What this validates

This proves the package, config schema, normalized examples, prompt masking, CLI stats path, and current smoke-training primitive are usable. It does not prove the production trainer exists or that Qwen can train.

## Required proof artifacts

- Python version.
- Install log summary.
- Import output.
- Pytest summary.
- Smoke script output.
- Credential presence summary with secrets redacted.

## Comments

- 2026-05-07 (agent): `python --version` reported `3.14.4`, which does not satisfy the validated project runtime expectation even though `pyproject.toml` says `>=3.11`.
- 2026-05-07 (agent): `pip install -e ".[dev]"` was blocked by package index/network policy when resolving build dependency `hatchling` (`Tunnel connection failed: 403 Forbidden`).
- 2026-05-07 (agent): Since package install failed, `import finpost` failed in that environment.
- 2026-05-07 (agent): Requested owner decision: confirm whether to proceed using an offline/air-gapped wheelhouse or provide a reachable package index mirror for dependency resolution.
- 2026-05-07 (owner): Path A approved: run install + preflight on local machine first.
- 2026-05-07 (agent): TODO staged for handoff: as soon as local machine is available, run `scripts/local_phase1_bootstrap.sh` then `scripts/local_phase1_minitest.sh`.
- 2026-05-09 (agent): Updated this issue to match the operator-grade runbook Gates 0 and 1 and to distinguish local smoke validation from production trainer readiness.
