# Phase 1 training runbook

This runbook turns `PLAN.md` Phase 0 and Phase 1 into operator gates for the current path:

1. local setup and tests,
2. TinyGPT infrastructure canary,
3. `Qwen/Qwen2.5-0.5B` Supervised Fine-Tuning soft launch,
4. full Qwen Supervised Fine-Tuning baseline,
5. Qwen Supervised Fine-Tuning ablations,
6. Direct Preference Optimization data, loss sanity, training, and comparison.

Phase 1 is a learning-first math post-training stack on GSM8K and MATH. Phase 2 filing work starts only after Phase 1 produces evidence about whether Direct Preference Optimization improves over Supervised Fine-Tuning.

## Operating rules

- Do not run a later gate until every required earlier gate has passing evidence recorded.
- Do not treat TinyGPT results as model-quality evidence. TinyGPT only proves the training infrastructure.
- Do not start full Qwen training, ablations, Direct Preference Optimization, or spend-bearing remote work without a completed `.scratch/templates/cost-gate-checklist.md`.
- Do not use `Qwen/Qwen2.5-0.5B-Instruct` as the Phase 1 training substrate. It is a reference baseline only. The Phase 1 base model is `Qwen/Qwen2.5-0.5B`.
- Do not start Direct Preference Optimization from an untrained base model or a TinyGPT checkpoint. It starts from a real Qwen Supervised Fine-Tuning checkpoint.
- Record command, config path, git SHA, seed, run ID, artifact paths, and pass/fail notes for every gate.
- Treat free Google Colab T4 as the default target environment for Qwen soft launches and first baselines. Use paid remote GPU only if Colab availability, runtime resets, or DPO memory pressure blocks progress.

## Artifact ledger

Create one ledger entry per run in the relevant issue comment. Use paths, not prose-only summaries.

```markdown
## Run evidence

- Gate:
- Date/time UTC:
- Operator:
- Git SHA:
- Command:
- Config:
- Seed:
- Environment: local CPU / target GPU / other
- Run ID or offline tracking path:
- Checkpoint path:
- Log path:
- Result: pass / fail
- Notes:
```

## Gate 0 - Local bootstrap

Purpose: prove the local Python environment can install the package, import `finpost`, and run the smallest trusted tests before any training work.

Run from the repository root:

```bash
./scripts/local_phase1_bootstrap.sh
```

Manual equivalent:

```bash
python --version
pip --version
python -m venv .venv
source .venv/bin/activate
python --version
pip install --upgrade pip
pip install -e ".[dev]"
python -c "import finpost; print(finpost.__version__)"
pytest tests/test_config.py tests/test_data_schema.py -v
```

PowerShell equivalent after creating or reusing `.venv`:

```powershell
.\.venv\Scripts\Activate.ps1
python --version
pip --version
pip install --upgrade pip
pip install -e ".[dev]"
python -c "import finpost; print(finpost.__version__)"
python -m pytest tests/test_config.py tests/test_data_schema.py -v
```

Pass criteria:
- Python is compatible with `pyproject.toml` (`>=3.11`).
- Editable install succeeds.
- `import finpost` succeeds.
- config and schema tests pass.

Proof artifacts:
- terminal log or issue comment with Python version, install result, import output, and pytest summary.

If this fails:
- fix environment or package-index access before continuing.
- do not run training commands.

## Gate 1 - Local mini validation

Purpose: prove the already-built data, schema, masking, safety, and smoke-test surfaces still work before relying on them for trainer work.

```bash
./scripts/local_phase1_minitest.sh
```

Manual equivalent:

```bash
pytest tests/test_config.py tests/test_data_schema.py tests/test_masking.py tests/test_cli_stats.py -v
python -m finpost.data.cli stats --help
python scripts/sft_smoke.py --help
python scripts/sft_smoke.py
```

PowerShell equivalent:

```powershell
.\.venv\Scripts\Activate.ps1
python -m pytest tests/test_config.py tests/test_data_schema.py tests/test_masking.py tests/test_cli_stats.py -v
python -m finpost.data.cli stats --help
python scripts/sft_smoke.py --help
python scripts/sft_smoke.py
```

Pass criteria:
- unit tests pass,
- dataset statistics command resolves,
- SFT smoke script help resolves,
- current one-batch smoke script runs.

Proof artifacts:
- pytest summary,
- smoke script output,
- issue comment linking any generated local log.

If this fails:
- fix the failing local surface before using it as a trainer dependency.

## Gate 2 - Production trainer readiness

Purpose: prove the planned production Supervised Fine-Tuning path exists before the operator tries to launch it.

Required files before this gate can pass:

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

Run:

```bash
pytest tests/test_dataset.py tests/test_config.py tests/test_optim.py tests/test_checkpoint.py tests/test_trainer.py -v
python -m finpost.training.train --help
```

Pass criteria:
- production trainer tests pass,
- command-line entry point exists,
- help output shows `--config`,
- checkpoint and resume behavior are covered by tests.

Proof artifacts:
- pytest summary,
- `python -m finpost.training.train --help` output,
- issue comment linking the implementation issue that added the trainer.

If this fails:
- return to `.scratch/phase1-sft-trainer/` issues. The runbook is not a substitute for missing trainer code.

## Gate 3 - TinyGPT local Supervised Fine-Tuning canary

Purpose: prove the full training infrastructure works on a cheap local model before using Qwen or cloud spend. This validates loss measurement, validation loss, tracking, checkpoint writes, and resume. It does not validate research quality.

Run:

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

Resume check:

```bash
WANDB_MODE=offline \
python -m finpost.training.train \
  --config experiments/local_tiny_gpt2.yaml \
  --device cpu \
  --resume-from results/checkpoints/<run-name>/step-<N>.pt \
  --max-steps 25
```

Pass criteria:
- command exits successfully,
- train loss is logged,
- validation loss is logged,
- learning rate and gradient norm are logged,
- tokens per second is logged,
- checkpoint is written under `results/checkpoints/`,
- resume continues from the checkpoint and preserves the deterministic tolerance owned by `.scratch/phase1-sft-trainer/issues/05-trainer.md`.

Proof artifacts:
- offline Weights & Biases directory or run log,
- checkpoint path,
- validation loss entry,
- resume command and result,
- issue comment with the artifact ledger.

If this fails:
- diagnose infrastructure first: config, data loader, masking, optimizer, checkpointing, or resume.
- do not launch Qwen.

## Gate 3.5 - Evaluation harness smoke test

Purpose: verify that the full eval pipeline (CLI + source registry + answer extractors + output writers) works end-to-end on a CPU-friendly tiny model and real GSM8K and MATH data. This is the "all green locally" gate before any paid GPU spin-up for evaluation.

Run from the repository root:

```bash
WANDB_MODE=offline \
python -m finpost.evals.eval_exact \
  --checkpoints tiny=sshleifer/tiny-gpt2 \
  --sources gsm8k math \
  --n 10 \
  --seed 42 \
  --out-dir results/evals/smoke_tiny_gpt2/ \
  --batch-size-gsm8k 2 \
  --batch-size-math 2 \
  --device cpu
```

Or use the automated smoke test in the minitest suite:

```bash
./scripts/local_phase1_minitest.sh
```

PowerShell equivalent after activating `.venv`:

```powershell
$env:WANDB_MODE = "offline"
python -m finpost.evals.eval_exact `
  --checkpoints tiny=sshleifer/tiny-gpt2 `
  --sources gsm8k math `
  --n 10 `
  --seed 42 `
  --out-dir results/evals/smoke_tiny_gpt2/ `
  --batch-size-gsm8k 2 `
  --batch-size-math 2 `
  --device cpu
```

This should finish in under 2 minutes on a modern CPU. `sshleifer/tiny-gpt2` is ~10 MB.

Pass criteria:
- smoke command completes without error on CPU,
- all six expected artifact files exist: `accuracy_summary.json`, `accuracy_summary.csv`, `details_tiny_gsm8k.csv`, `details_tiny_math.csv`, `run_metadata.json`, `cost_summary.json`,
- `accuracy_summary.csv` has 3 lines (1 header + 2 data rows for `{tiny} × {gsm8k, math}`),
- `details_tiny_gsm8k.csv` and `details_tiny_math.csv` each contain 10 evaluated examples,
- `run_metadata.json` contains populated `dtype`, `device`, `seed: 42`, and `git_sha` fields,
- `cost_summary.json` contains `elapsed_sec > 0`, `generated_tokens > 0`, `tokens_per_second > 0`, and `estimated_cost_usd: null`,
- second run with the same seed produces byte-identical `details_*.csv` files.

Proof artifacts:
- output directory path (`results/evals/smoke_tiny_gpt2/`),
- minitest or manual command output,
- note on parse_success_rate (it will be near zero on tiny-gpt2 for math — that is expected and is exactly what the test validates).

Notes:
- tiny-gpt2 has only 4 attention heads and 4 layers. Generation on math will be incoherent. The point is to test the infrastructure, not model quality.
- If parse_success_rate is 0% on math, that is a success for this test, not a failure. The metric we validate is that the pipeline runs and records parse failures correctly.

If this fails:
- do not launch eval on real models. Fix the eval harness first.
- diagnose: can the CLI parse arguments? Does the source registry load GSM8K and MATH? Do answer extractors run? Are output writers working?

## Gate 4 - Qwen 0.5B Supervised Fine-Tuning soft launch

Purpose: prove the real Phase 1 base model loads and trains through the exact same production trainer path as TinyGPT. The default target environment is free Google Colab with a T4-class GPU; paid GPU rental is fallback, not the first plan.

Before this gate:
- Gate 3 must pass.
- Confirm the Colab runtime has a GPU via `!nvidia-smi`.
- Mount Google Drive or otherwise persist checkpoints outside the ephemeral Colab VM.
- Complete `.scratch/templates/cost-gate-checklist.md` only if the run uses spend-bearing remote compute.

Colab setup sketch:

```bash
git clone <repo-url> finpost
cd finpost
pip install -e ".[dev]"
nvidia-smi
```

Keep the first Qwen config conservative on Colab:
- `max_seq_len`: start at 512 or 1024, not 4096.
- `per_device_batch_size`: start at 1 or 2.
- `grad_accum_steps`: increase this to recover effective batch size.
- checkpoint frequently to a persistent path, preferably Google Drive.

Run:

```bash
python -m finpost.training.train \
  --config experiments/baseline.yaml \
  --max-steps 20
```

Pass criteria:
- `Qwen/Qwen2.5-0.5B` loads,
- 20 optimizer steps complete,
- train loss, validation loss, learning rate, gradient norm, and tokens per second are logged,
- checkpoint is written,
- the run records config, seed, git SHA, and library versions.
- the Colab GPU type and available VRAM are recorded from `nvidia-smi`.

Proof artifacts:
- tracking run ID or offline path,
- checkpoint path,
- run manifest or logged metadata,
- issue comment with the artifact ledger.

If this fails:
- fix the production path before full baseline training.
- do not hide a Qwen failure behind TinyGPT success; they validate different things.
- if the failure is Colab runtime instability or VRAM pressure, first shrink sequence length/batch size and enable gradient checkpointing before moving to paid GPU.

## Gate 5 - Full Qwen Supervised Fine-Tuning baseline

Purpose: produce the first real Supervised Fine-Tuning checkpoint that can be evaluated and later used for Direct Preference Optimization.

Before this gate:
- Gate 3 TinyGPT canary passes.
- Gate 4 Qwen soft launch passes.
- Cost gate checklist is completed.

Run:

```bash
python -m finpost.training.train --config experiments/baseline.yaml
```

Pass criteria:
- run reaches configured `max_steps`,
- final validation loss is recorded,
- best checkpoint and final checkpoint are saved,
- run metadata is complete enough to reproduce the run.

Proof artifacts:
- tracking run ID,
- final and best checkpoint paths,
- validation-loss curve,
- config snapshot,
- git SHA,
- seed,
- operator notes.

If this fails:
- record the root cause and choose one of: resume from checkpoint, revise config, or return to trainer implementation.
- do not start ablations until there is one passing full Qwen baseline.

## Gate 6 - Supervised Fine-Tuning ablations

Purpose: compare a small set of Qwen Supervised Fine-Tuning settings using the same validated trainer path.

Planned axes from `PLAN.md`:
- data scale: 10%, 50%, 100%,
- learning rate: `1e-5`, `5e-5`, `1e-4`,
- epochs or step-equivalent budget according to the production config convention.

Use one config file per cell:

```bash
python -m finpost.training.train --config experiments/<ablation-config>.yaml
```

For each cell, record:
- run ID,
- config path,
- seed,
- data scale,
- learning rate,
- training budget,
- wall-clock time,
- final validation loss,
- best checkpoint path,
- failure root cause, if any.

Pass criteria:
- every launched cell has a complete ledger entry,
- failed cells are marked failed with cause,
- at least one best checkpoint candidate is selected for evaluation and Direct Preference Optimization preparation.

Proof artifacts:
- ablation table under `results/` or linked issue comment,
- tracking run IDs,
- checkpoint paths,
- selected best-checkpoint rationale.

If this fails:
- stop expanding the grid. Fix the failure mode or shrink the matrix.

## Gate 7 - Evaluation handoff for Base vs Supervised Fine-Tuning

Purpose: prove the Supervised Fine-Tuning checkpoint improved, regressed, or failed to move on the held-out math evaluation surface.

Minimum required metrics:
- GSM8K final-answer exact-match accuracy,
- MATH final-answer equivalence using a known normalizer,
- MATH per-difficulty breakdown,
- bootstrapped 95% confidence intervals,
- response-length statistics.

Command placeholder until the evaluation entry point lands:

```bash
python -m finpost.evaluation.evaluate_math --config experiments/eval_phase1.yaml
```

Pass criteria:
- Base and Supervised Fine-Tuning checkpoints are evaluated on the same held-out prompts,
- confidence intervals are reported,
- the result states whether Supervised Fine-Tuning improved, hurt, or was statistically indistinguishable from base.

Proof artifacts:
- evaluation output under `results/`,
- checkpoint IDs,
- exact evaluation command,
- metrics table.

If this fails:
- do not use the checkpoint for Direct Preference Optimization until the failure is explained.

## Gate 8 - Direct Preference Optimization pair preparation

Purpose: build preference data from the best real Qwen Supervised Fine-Tuning checkpoint, not from TinyGPT and not from the base model.

Before this gate:
- Gate 5 has a passing Qwen Supervised Fine-Tuning checkpoint.
- Preferably Gate 7 has evaluated it.
- Q-B policy for all-correct and all-incorrect prompt groups is recorded in `.scratch/phase1-training-runbook/issues/05-decision-gates-and-signoff.md`.

Expected command once implemented:

```bash
python scripts/build_dpo_pairs.py \
  --checkpoint results/checkpoints/<sft-run>/best.pt \
  --output data/processed/phase1_dpo_pairs.jsonl \
  --samples-per-prompt 8 \
  --temperature 0.8 \
  --seed <seed>
```

Pass criteria:
- preference-pair file contains prompt, chosen response, rejected response, source dataset ID, source checkpoint ID, grading result, and grading reason,
- no test-set prompts are used,
- all-correct and all-incorrect prompt groups are counted separately,
- pair construction is deterministic for fixed seed and input completions.

Proof artifacts:
- preference-pair file path,
- generation log,
- grading summary,
- all-correct and all-incorrect rates,
- source checkpoint path.

If this fails:
- fix grading or pair construction before DPO loss work proceeds.

## Gate 9 - Direct Preference Optimization loss parity

Purpose: prove the local Direct Preference Optimization loss matches a trusted reference before training with it.

Install reference tooling if needed:

```bash
pip install -e ".[dpo-reference]"
```

Run once DPO tests exist:

```bash
pytest tests/test_dpo.py tests/test_preference_data.py -v
```

Pass criteria:
- local DPO loss matches reference within `1e-5` on a fixed batch,
- beta sensitivity is covered,
- swapping chosen and rejected changes loss direction,
- reference model is frozen and receives no gradients.

Proof artifacts:
- pytest summary,
- fixed-batch parity value,
- issue comment linking the reference version.

If this fails:
- do not train DPO. Numerical mismatch here invalidates the training signal.

## Gate 10 - Direct Preference Optimization soft launch and baseline

Purpose: verify the DPO trainer runs end to end, then produce the first Qwen SFT + DPO checkpoint.

DPO is more memory-sensitive than SFT because it compares a trainable policy model against a frozen reference model. On Colab, keep the first DPO run small: short sequence length, batch size 1, frequent checkpoints, and a clear fallback plan to precompute reference log-probabilities if holding both models in memory becomes the blocker.

Expected soft-launch command once implemented:

```bash
WANDB_MODE=offline \
python -m finpost.training.dpo_train \
  --config experiments/dpo_tiny_gpt2.yaml \
  --max-steps 20
```

Expected Qwen command once implemented:

```bash
python -m finpost.training.dpo_train --config experiments/dpo_baseline.yaml
```

Pass criteria:
- DPO soft launch logs loss and writes checkpoint,
- Qwen DPO run starts from the selected Qwen SFT checkpoint,
- frozen reference model is recorded,
- DPO checkpoint is saved with config, seed, git SHA, and source SFT checkpoint metadata.

Proof artifacts:
- tracking run ID,
- DPO checkpoint path,
- source SFT checkpoint path,
- reference model metadata,
- final DPO loss curve.

If this fails:
- record whether the failure is data, loss, trainer, memory, or cost.
- do not compare SFT vs DPO until a real DPO checkpoint exists.

## Gate 11 - Base vs SFT vs SFT + DPO comparison

Purpose: answer the Phase 1 question: did Direct Preference Optimization add value over Supervised Fine-Tuning?

Compare on the same held-out evaluation set:
- Base,
- best Qwen SFT checkpoint,
- best Qwen SFT + DPO checkpoint.

Required report fields:
- GSM8K exact-match accuracy,
- MATH equivalence accuracy,
- bootstrapped 95% confidence intervals,
- response-length statistics,
- failure examples,
- clear conclusion: DPO improved, hurt, or was statistically indistinguishable.

Proof artifacts:
- comparison report under `results/`,
- metric table,
- confidence intervals,
- exact commands and checkpoint IDs.

Only after this gate should Phase 2 filing-data training begin.
