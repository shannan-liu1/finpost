#!/usr/bin/env bash
set -euo pipefail

# Mini/local validation suite before A100 training.

# shellcheck disable=SC1091
source .venv/bin/activate

pytest tests/test_config.py tests/test_data_schema.py tests/test_masking.py tests/test_cli_stats.py -v
python -m finpost.data.cli stats --help
python scripts/sft_smoke.py --help
python scripts/sft_smoke.py

echo "[OK] mini/local validation complete"
