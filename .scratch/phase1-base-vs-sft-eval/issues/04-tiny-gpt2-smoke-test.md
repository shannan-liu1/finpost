# 04 - End-to-end smoke test with tiny-gpt2

- **Status:** Not Started
- **Ready for agent:** yes (after issue 03)
- **Depends on:** 02-eval-exact-cli-and-output, 03-colab-and-kaggle-notebooks

## Goal

Verify that the full eval pipeline (CLI + source registry + answer extractors + output writers) works end-to-end on a CPU-friendly tiny model and real GSM8K and MATH data. This is the "all green locally" gate before any paid GPU spin-up.

## Scope

**In scope:**
- A single shell command in the smoke section of `scripts/local_phase1_minitest.sh` (or a new minimal script) that invokes the eval CLI on `sshleifer/tiny-gpt2` (the project's existing infra-canary model) with n=10 on GSM8K and MATH.
- A verification step that inspects the produced artifacts and confirms:
  - All five expected files exist in the output directory.
  - `accuracy_summary.csv` has 2 rows (1 checkpoint × 2 sources).
  - `details_tiny_gsm8k.csv` has exactly 10 rows.
  - `details_tiny_math.csv` has exactly 10 rows.
  - `parse_success_rate` is reported (it will be near zero for tiny-gpt2 on math — that's fine; the point is the pipeline ran).
- A second run with the same seed and the same args; verify byte-identical `details_*.csv` between runs.

**Out of scope:**
- Real-model eval (that's the actual run on RunPod / Colab / Kaggle once smoke passes).
- Any training step.
- Tuning the answer extractors against tiny-gpt2's output (it will fail to produce parseable answers on most prompts — that's expected and is exactly what the `parse_success` column is for).

## The smoke command

```bash
WANDB_MODE=offline python -m finpost.evals.eval_exact \
  --checkpoints tiny=sshleifer/tiny-gpt2 \
  --sources gsm8k math \
  --n 10 \
  --seed 42 \
  --out-dir results/evals/smoke_tiny_gpt2/ \
  --batch-size-gsm8k 2 \
  --batch-size-math 2 \
  --device cpu
```

This should finish in under 2 minutes on a modern CPU. tiny-gpt2 is ~10 MB.

## Acceptance criteria

1. The smoke command above runs to completion without error on a CPU-only machine (Windows PowerShell + macOS / Linux bash).
2. After the run, `ls results/evals/smoke_tiny_gpt2/` lists exactly: `accuracy_summary.json`, `accuracy_summary.csv`, `details_tiny_gsm8k.csv`, `details_tiny_math.csv`, `run_metadata.json`, `cost_summary.json`.
3. `wc -l < results/evals/smoke_tiny_gpt2/details_tiny_gsm8k.csv` returns `11` (1 header + 10 rows).
4. `wc -l < results/evals/smoke_tiny_gpt2/details_tiny_math.csv` returns `11`.
5. Running the same command twice with the same seed produces byte-identical `details_*.csv` files (`diff` returns empty).
6. `run_metadata.json` contains a populated `dtype` field (`"float32"` on CPU), `device` field, `seed: 42`, and a non-empty `git_short_sha`.
7. `cost_summary.json` contains `elapsed_sec > 0`, `generated_tokens > 0`, `tokens_per_second > 0`, and `estimated_cost_usd: null` (no `--gpu-cost-per-hour` was passed).
8. `accuracy_summary.csv` contains exactly 2 rows (header + 2 data rows for `{tiny} × {gsm8k, math}`).
9. The smoke command is documented in either `scripts/local_phase1_minitest.sh` or `docs/runbooks/phase1-training-runbook.md` so future operators can re-run it.
10. The full pytest suite still passes (`pytest -q`).

## Notes / open questions

- tiny-gpt2 has only 4 attention heads and 4 layers. Generation on math will be incoherent. That is *exactly the point* of using it for infrastructure testing — we are not testing model quality, only that the eval pipeline runs without crashing on real data.
- If parse_success_rate is 0% on math but the pipeline still produces all files, that is a *success* for this test, not a failure. The metric we are validating is that the harness *runs* and *records* parse failures correctly, not that the model is good.
- This smoke test is the gate. If it passes, the eval CLI is trusted enough to launch on RunPod / Colab against Qwen 0.5B base + the combined SFT checkpoint for the real n=500 run.
- Do not invent a separate `smoke.py` Python script. The CLI is the entry point; smoke is the CLI invoked with tiny inputs. Adding more code-paths defeats the purpose of having a single CLI.
