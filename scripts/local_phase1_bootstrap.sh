#!/usr/bin/env bash
set -euo pipefail

# Phase 1 local bootstrap (Path A)
# Goal: make local setup repeatable before any A100 usage.

python --version
pip --version

# Create venv if missing.
if [[ ! -d .venv ]]; then
  python -m venv .venv
fi

# shellcheck disable=SC1091
source .venv/bin/activate

python --version
pip install --upgrade pip
pip install -e ".[dev]"

# Verify package import.
python -c "import finpost; print(finpost.__version__)"

# Optional: install DPO reference tooling used later in Phase 1.
pip install -e ".[dpo-reference]"

echo "[OK] local bootstrap complete"
