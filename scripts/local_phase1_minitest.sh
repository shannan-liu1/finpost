#!/usr/bin/env bash
set -euo pipefail

# Mini/local validation suite before A100 training.

# shellcheck disable=SC1091
source .venv/bin/activate

pytest tests/test_config.py tests/test_data_schema.py tests/test_masking.py tests/test_cli_stats.py -v
python -m finpost.data.cli stats --help
python scripts/sft_smoke.py --help
python scripts/sft_smoke.py

# =============================================================================
# Eval harness smoke test (Phase 1 base-vs-SFT eval)
# =============================================================================

echo ""
echo "================= Eval harness smoke test (tiny-gpt2) ================="

# Run the eval CLI on sshleifer/tiny-gpt2 with n=10 on GSM8K and MATH.
# This verifies that the full eval pipeline (CLI + source registry + answer
# extractors + output writers) works end-to-end on a CPU-friendly tiny model
# and real GSM8K and MATH data.

WANDB_MODE=offline python -m finpost.evals.eval_exact \
  --checkpoints tiny=sshleifer/tiny-gpt2 \
  --sources gsm8k math \
  --n 10 \
  --seed 42 \
  --out-dir results/evals/smoke_tiny_gpt2/ \
  --batch-size-gsm8k 2 \
  --batch-size-math 2 \
  --device cpu

echo ""
echo "[eval_exact] Verifying smoke test output artifacts..."

# Verify all 5 expected files exist.
expected_files=(
  "results/evals/smoke_tiny_gpt2/accuracy_summary.json"
  "results/evals/smoke_tiny_gpt2/accuracy_summary.csv"
  "results/evals/smoke_tiny_gpt2/details_tiny_gsm8k.csv"
  "results/evals/smoke_tiny_gpt2/details_tiny_math.csv"
  "results/evals/smoke_tiny_gpt2/run_metadata.json"
  "results/evals/smoke_tiny_gpt2/cost_summary.json"
)

for file in "${expected_files[@]}"; do
  if [[ ! -f "$file" ]]; then
    echo "[FAIL] Expected output file not found: $file"
    exit 1
  fi
done

echo "  [OK] All 6 expected artifact files exist."

# Verify accuracy_summary.csv has 3 lines (header + 2 data rows for {tiny} × {gsm8k, math}).
accuracy_lines=$(wc -l < results/evals/smoke_tiny_gpt2/accuracy_summary.csv)
if [[ "$accuracy_lines" -ne 3 ]]; then
  echo "[FAIL] accuracy_summary.csv has $accuracy_lines lines (expected 3: header + 2 data rows)"
  exit 1
fi
echo "  [OK] accuracy_summary.csv has 3 lines (1 header + 2 data rows)."

# Verify details_tiny_gsm8k.csv has exactly 10 data rows (header + 10) via csv.reader.
echo "--> Verifying details_tiny_gsm8k.csv has exactly 10 data rows (header + 10) via csv.reader..."
gsm8k_rows=$(.venv/Scripts/python.exe -c "
import csv, sys
with open('results/evals/smoke_tiny_gpt2/details_tiny_gsm8k.csv', newline='') as f:
    print(sum(1 for _ in csv.reader(f)))
")
if [[ "$gsm8k_rows" -ne 11 ]]; then
  echo "[FAIL] details_tiny_gsm8k.csv has $gsm8k_rows rows (header+rows), expected 11."
  exit 1
fi
echo "  [OK] details_tiny_gsm8k.csv has 11 rows (header + 10 data)."

# Verify details_tiny_math.csv has exactly 10 data rows (header + 10) via csv.reader.
echo "--> Verifying details_tiny_math.csv has exactly 10 data rows (header + 10) via csv.reader..."
math_rows=$(.venv/Scripts/python.exe -c "
import csv, sys
with open('results/evals/smoke_tiny_gpt2/details_tiny_math.csv', newline='') as f:
    print(sum(1 for _ in csv.reader(f)))
")
if [[ "$math_rows" -ne 11 ]]; then
  echo "[FAIL] details_tiny_math.csv has $math_rows rows (header+rows), expected 11."
  exit 1
fi
echo "  [OK] details_tiny_math.csv has 11 rows (header + 10 data)."

# Verify run_metadata.json contains required fields.
if ! grep -q '"dtype"' results/evals/smoke_tiny_gpt2/run_metadata.json; then
  echo "[FAIL] run_metadata.json missing dtype field"
  exit 1
fi
if ! grep -q '"device"' results/evals/smoke_tiny_gpt2/run_metadata.json; then
  echo "[FAIL] run_metadata.json missing device field"
  exit 1
fi
if ! grep -q '"seed": 42' results/evals/smoke_tiny_gpt2/run_metadata.json; then
  echo "[FAIL] run_metadata.json missing or incorrect seed field"
  exit 1
fi
if ! grep -q '"git_sha"' results/evals/smoke_tiny_gpt2/run_metadata.json; then
  echo "[FAIL] run_metadata.json missing git_sha field"
  exit 1
fi
echo "  [OK] run_metadata.json contains all required fields (dtype, device, seed, git_sha)."

# Verify cost_summary.json contains required fields and values are correct.
echo "--> Verifying cost_summary.json fields..."
.venv/Scripts/python.exe -c "
import json, sys
d = json.load(open('results/evals/smoke_tiny_gpt2/cost_summary.json'))
assert d['elapsed_sec'] > 0, f'elapsed_sec is {d[\"elapsed_sec\"]}, expected > 0'
assert d['generated_tokens'] > 0, f'generated_tokens is {d[\"generated_tokens\"]}, expected > 0'
assert d['generated_tokens_decoded'] > 0, f'generated_tokens_decoded is {d[\"generated_tokens_decoded\"]}, expected > 0'
assert d['generated_tokens_decoded'] <= d['generated_tokens'], (
    f\"generated_tokens_decoded={d['generated_tokens_decoded']} must be \"
    f\"<= generated_tokens={d['generated_tokens']} (decoded is a subset)\"
)
assert d['tokens_per_second'] > 0, f'tokens_per_second is {d[\"tokens_per_second\"]}, expected > 0'
assert d['estimated_cost_usd'] is None, f'estimated_cost_usd is {d[\"estimated_cost_usd\"]}, expected None'
"
if [[ $? -ne 0 ]]; then
  echo "[FAIL] cost_summary.json validation failed (see error above)"
  exit 1
fi
echo "  [OK] cost_summary numeric fields all positive; estimated_cost_usd is null."

# Verify byte-identity: run again with the same seed and compare.
echo ""
echo "[eval_exact] Running second smoke test for byte-identity verification..."

# Suppress stdout for cleanliness, but let stderr through so a real
# crash on the second run is visible.
WANDB_MODE=offline python -m finpost.evals.eval_exact \
  --checkpoints tiny=sshleifer/tiny-gpt2 \
  --sources gsm8k math \
  --n 10 \
  --seed 42 \
  --out-dir results/evals/smoke_tiny_gpt2_verify/ \
  --batch-size-gsm8k 2 \
  --batch-size-math 2 \
  --device cpu > /dev/null

# Compare details files from both runs.
if ! diff -q results/evals/smoke_tiny_gpt2/details_tiny_gsm8k.csv results/evals/smoke_tiny_gpt2_verify/details_tiny_gsm8k.csv > /dev/null 2>&1; then
  echo "[FAIL] GSM8K details files are not byte-identical between runs"
  exit 1
fi
if ! diff -q results/evals/smoke_tiny_gpt2/details_tiny_math.csv results/evals/smoke_tiny_gpt2_verify/details_tiny_math.csv > /dev/null 2>&1; then
  echo "[FAIL] MATH details files are not byte-identical between runs"
  exit 1
fi
echo "  [OK] Both runs produced byte-identical details files (deterministic)."

# Clean up the verification run.
rm -rf results/evals/smoke_tiny_gpt2_verify/

echo ""
echo "[OK] Eval harness smoke test complete. All checks passed."

echo ""
echo "[OK] mini/local validation complete"
