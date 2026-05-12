# Phase 1 base-vs-Supervised-Fine-Tuning exact-answer evaluation harness

- **Status:** Not Started
- **Created:** 2026-05-11
- **Owner:** Shannan
- **Estimated time:** 1-2 days implementation, ~1 hour eval runtime per checkpoint pair
- **Depends on:** [`phase1-sft-trainer`](../phase1-sft-trainer/PRD.md)
- **Consumed by:** [`phase1-compute-aware-post-training`](../phase1-compute-aware-post-training/PRD.md) — Stage 0 calls this workstream's CLI per checkpoint and aggregates the resulting JSON summaries into the per-checkpoint evaluation curve required by that workstream's acceptance criterion 1.

## Goal

Build a reproducible exact-answer evaluation harness that compares the Phase 1 combined Supervised Fine-Tuning checkpoint against the `Qwen/Qwen2.5-0.5B` base model on GSM8K and MATH at `n=500` per source. The harness produces machine-readable artifacts (per-example CSVs, summary JSON and CSV, cost metadata, reproducibility metadata) and serves as the reference measurement instrument for every downstream post-training comparison.

This workstream answers the single question gating all further post-training work: does the combined Supervised Fine-Tuning checkpoint produce measurably better exact-answer accuracy than the base model on the Phase 1 mathematical reasoning surface?

Beyond the immediate base-versus-Supervised-Fine-Tuning question, the CLI built here is the eval primitive that the larger [`phase1-compute-aware-post-training`](../phase1-compute-aware-post-training/PRD.md) workstream consumes for its per-checkpoint accuracy curves. Both workstreams share this eval mechanism so that all downstream method comparisons (rejection Supervised Fine-Tuning, uniform On-Policy Distillation, verifier-weighted On-Policy Distillation, adaptive-compute On-Policy Distillation) are measured on the same instrument.

## Scope

**In scope:**

- A `python -m finpost.evals.eval_exact` command-line interface taking `--checkpoints name=path` pairs, `--sources gsm8k math`, `--n`, `--seed`, `--out-dir`, and per-source batch sizes.
- A domain-agnostic source registry (`src/finpost/evals/sources.py`) defining `EvalSource(name, load_examples, extract_answer, score, default_max_new_tokens)`. Initial registry entries: `gsm8k`, `math`. Phase 2 finance sources (`filing_extraction`, `filing_reasoning`) slot in by registering new `EvalSource` entries with no change to the orchestration code.
- Six fixes to known eval-logic problems carried over from prior ad-hoc evaluations:
  1. Seeded random sample of `n` examples (not the first `n`).
  2. `parse_success` boolean column on every per-example output row.
  3. Full generated text saved per row, no truncation.
  4. `max_new_tokens` set per source: `gsm8k=256`, `math=768`.
  5. Model loaded in `bfloat16` on Ampere or newer (`torch.cuda.get_device_capability()[0] >= 8`), otherwise `float16`.
  6. Batched generation with halve-on-out-of-memory fallback down to `batch_size=1`, then fail loudly.
- An inline cost-tracking helper inside `eval_exact.py` (approximately thirty lines) that records start and end timestamps, GPU device name, dtype, elapsed seconds, generated token count, tokens per second, and an optional dollar-cost estimate when `--gpu-cost-per-hour` is supplied. Writes `cost_summary.json`.
- Reproducibility metadata (`run_metadata.json`): device name, CUDA version, torch version, transformers version, dtype, eval `n`, seed, per-source generation settings, and the git short SHA.
- Two thin notebook wrappers that call the CLI and render summary tables inline:
  - `notebooks/colab_phase1_eval_and_cost_tracking.ipynb` (Google Colab, Drive paths).
  - `notebooks/kaggle_phase1_eval_and_cost_tracking.ipynb` (Kaggle, `/kaggle/working/` paths).
- Unit tests covering: seeded subsampling determinism, GSM8K and MATH answer extractors (positive and negative cases), batched-versus-single generation equivalence on a tiny model with shared seed, and the out-of-memory fallback halving its batch size and retrying.

**Out of scope:**

- Large-language-model-as-judge evaluation. Phase 1 uses programmatic exact-match verification only; this matches `CONTEXT.md` and the existing PRDs.
- Sampling-based evaluation (`pass@k`, `majority@k`, answer consistency). Belongs to a downstream stability workstream after base-versus-Supervised-Fine-Tuning lands a meaningful delta.
- Bucket evaluation by baseline difficulty. Same destination as the stability metrics.
- Efficiency-per-compute metrics (accuracy gain per training token, per generated token, per GPU hour). Belongs to the training-side workstreams that produce the denominator, not to this evaluation harness.
- A separate `src/finpost/utils/run_tracker.py` module. Cost tracking is inline until a second consumer earns its promotion to a module.
- A separate `src/finpost/analysis/` subpackage or `make_phase1_report.py` module. Notebooks render summary tables and a single bar chart inline.
- Direct Preference Optimization, on-policy distillation, Group Relative Policy Optimization, rollout generation, scoring, pair construction, or any weighted-loss scaffolding. Each lives in its own workstream.

## Deliverables

```
src/finpost/evals/
├── __init__.py
├── eval_exact.py          # CLI entry point; orchestration, generation, output writing, inline cost helper
└── sources.py             # EvalSource dataclass + registry: gsm8k, math

notebooks/
├── colab_phase1_eval_and_cost_tracking.ipynb
└── kaggle_phase1_eval_and_cost_tracking.ipynb

tests/
└── test_eval_exact.py     # seeded subsampling, answer extractors, batched-vs-single parity, OOM fallback
```

Each evaluation run writes the following five files into `--out-dir`:

```
accuracy_summary.json
accuracy_summary.csv
details_<checkpoint_name>_<source>.csv
run_metadata.json
cost_summary.json
```

`accuracy_summary.{json,csv}` contains one row per `(checkpoint, source)` pair with columns: `checkpoint`, `source`, `n`, `accuracy`, `parse_success_rate`, `generated_tokens`, `elapsed_sec`.

`details_<checkpoint>_<source>.csv` contains one row per evaluated example with columns: `example_id`, `prompt`, `generated`, `gold_answer`, `predicted_answer`, `parse_success`, `is_correct`.

## Acceptance criteria

1. The command `python -m finpost.evals.eval_exact --checkpoints base=<path> combined=<path> --sources gsm8k math --n 500 --seed 42 --out-dir results/evals/<run_name>/` runs to completion on a Colab T4 or Kaggle T4 environment and writes all five artifact files listed under Deliverables.
2. `accuracy_summary.csv` contains exactly four rows: `{base, combined} × {gsm8k, math}`, each with the columns listed in Deliverables.
3. Re-running the same command with the same `--seed` produces byte-identical `details_*.csv` files. (Greedy decoding, fixed dtype, seeded shuffling.)

   **Amendment 2026-05-12:** Byte-identity on CUDA is best-effort, not guaranteed. `_set_cuda_determinism` sets cuDNN determinism flags and `CUBLAS_WORKSPACE_CONFIG` before any model loading, but `torch.use_deterministic_algorithms(True, warn_only=True)` means operations that have no deterministic CUDA implementation emit a warning and continue rather than aborting. Full byte-identity across re-runs is only guaranteed on CPU. CUDA re-runs with the same seed on the same device are expected to be identical in practice for standard transformer forward+generate paths, but cannot be formally guaranteed if non-deterministic ops are present in the model graph.
4. Both notebook variants run top-to-bottom without `/kaggle/working/` paths leaking into the Colab notebook or vice versa, and persist their outputs to the platform-appropriate location (Google Drive for Colab, `/kaggle/working/` for Kaggle).
5. `pytest tests/test_eval_exact.py -v` passes.
6. The post-evaluation summary (rendered inline in the notebook or printed by the CLI) explicitly identifies exactly one of:
   - `combined > base on both sources`
   - `combined > base on MATH only`
   - `combined > base on GSM8K only`
   - `combined ≤ base on both sources`
   - `combined improves training loss but not exact accuracy` (only assertable in combination with the training-side loss logs)

   This single statement is the decision gate that selects the next workstream.

## Notes / open questions

- The combined Supervised Fine-Tuning checkpoint is expected at the Colab convention path `/content/drive/MyDrive/finpost_runs/checkpoints/combined_hf_step_<N>/` and at the Kaggle convention path `/kaggle/working/results/checkpoints/<run_name>/`. The CLI is path-agnostic; these are notebook-cell defaults only.
- MATH at `n=500` with `max_new_tokens=768` on a T4 batched at `batch_size=4` is estimated at 30 to 60 minutes. This is a one-time cost. vLLM and sglang would offer roughly 2-3× speedup but require approximately one engineering day to integrate cleanly on T4 with the right driver and the right page-attention path; out of scope here.
- The domain-agnostic source registry is the single forward-looking design decision. It exists so that Phase 2 finance evaluations (`filing_extraction`, `filing_reasoning`) can be added by registering new `EvalSource` entries without touching `eval_exact.py`.
- The Kaggle notebook is added because the Phase 1 Supervised Fine-Tuning runs that produced the combined checkpoint already executed on Kaggle. Re-evaluating on Kaggle avoids the multi-gigabyte checkpoint download to Colab.

## Future / optional follow-up

Captured here so that the immediate workstream stays focused and the durable record of further evaluation work is not lost:

- **Stability evaluation.** `pass@1`, `pass@K`, `majority@K`, answer consistency, correct consistency. Tells whether the model's reasoning distribution shifted toward the correct answer, not just whether one lucky sample exists. Belongs to a follow-up `phase1-stability-eval` workstream if base-versus-Supervised-Fine-Tuning produces an interesting delta.
- **Bucket evaluation by baseline difficulty.** Partition evaluation prompts into `easy` (baseline `p_correct ≥ 0.875`), `mixed` (`0.125 < p_correct < 0.875`), and `hopeless` (`p_correct ≤ 0.125`) using K samples from a fixed baseline before any post-training run, then report accuracy per bucket per checkpoint. Same destination as the stability metrics.
- **Efficiency-per-compute metrics.** Accuracy gain per training token, per generated token, per GPU hour. Lives in [`phase1-compute-aware-post-training`](../phase1-compute-aware-post-training/PRD.md), which already specifies a cost ledger with these columns at the run and method level.
- **Verifier-weighted and adaptive-compute On-Policy Distillation experiments.** These are the five-method comparison surface (uniform Supervised Fine-Tuning, rejection Supervised Fine-Tuning, uniform On-Policy Distillation, verifier-weighted On-Policy Distillation, adaptive-compute On-Policy Distillation) owned by [`phase1-compute-aware-post-training`](../phase1-compute-aware-post-training/PRD.md) Stages 1 through 5. This workstream supplies the eval primitive those stages call.
