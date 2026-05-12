# 02 - Eval-exact CLI: generation, output artifacts, cost helper

- **Status:** Not Started
- **Ready for agent:** yes (after issue 01)
- **Depends on:** 01-source-registry-and-extractors

## Goal

Build `python -m finpost.evals.eval_exact` — the command-line interface that loads one or more checkpoints, runs them on one or more sources from the registry, and emits the five artifact files specified in the PRD (`accuracy_summary.json`, `accuracy_summary.csv`, `details_<checkpoint>_<source>.csv`, `run_metadata.json`, `cost_summary.json`).

The CLI is the eval primitive that the `phase1-compute-aware-post-training` workstream will reuse for its per-checkpoint accuracy curves. Keep the dispatch logic clean enough that Stage 0 of that workstream can call the CLI in a loop over checkpoints.

## Scope

**In scope:**
- `src/finpost/evals/eval_exact.py` — single file containing:
  - Argument parsing (`argparse`).
  - Model + tokenizer loading (one checkpoint at a time; load, eval all sources, unload, next checkpoint).
  - Batched greedy generation with halve-on-OOM fallback.
  - Per-example answer extraction + scoring via the source registry from issue 01.
  - CSV/JSON writers for the five output files.
  - Inline cost-tracking helper (~30 lines) that records start/end timestamps, GPU device name, dtype, elapsed seconds, generated token count, tokens-per-second, and optional `estimated_cost_usd`.
  - `run_metadata.json` builder (device name, CUDA/torch/transformers versions, dtype, eval n, seed, generation settings per source, git short SHA).
- `tests/test_eval_exact.py` — tests for:
  - Seeded subsampling determinism (same seed → same indices).
  - Batched-vs-single generation parity on a tiny model with shared seed.
  - OOM fallback halves batch size and retries, eventually failing loudly at batch_size=1.
  - Output schemas: `accuracy_summary.csv` has the required columns; `details_*.csv` has the required columns; `cost_summary.json` has the required fields.
  - Re-run with same seed produces byte-identical `details_*.csv`.

**Out of scope:**
- The source registry (built in issue 01; this issue *consumes* it).
- Notebooks (issue 03).
- Real-model smoke runs (issue 04).
- Plot generation or markdown reports.
- A separate `run_tracker.py` module — keep cost tracking inline.

## CLI signature

```
python -m finpost.evals.eval_exact \
  --checkpoints base=<path_or_hf_id> combined=<path> [name=path ...] \
  --sources gsm8k math \
  --n 500 \
  --seed 42 \
  --out-dir results/evals/<run_name>/ \
  [--batch-size-gsm8k 8] \
  [--batch-size-math 4] \
  [--gpu-cost-per-hour 1.39] \
  [--device cuda]
```

- `--checkpoints` takes `name=path` pairs. `path` is either a local directory or a Hugging Face model id.
- `--sources` looks up names in `finpost.evals.sources.REGISTRY`.
- `--n` is the number of examples per source (seeded random subset of the test split).
- `--seed` controls the example subset *and* generation (greedy is deterministic but seeding still matters for shuffling).
- `--out-dir` is created if it does not exist.
- `--gpu-cost-per-hour` is optional. If supplied, `cost_summary.json.estimated_cost_usd` is populated; otherwise null.
- `--device` defaults to `cuda` if available, else `cpu`.

## Implementation notes — the parts that matter

### Model loading

- One checkpoint loaded at a time. Evaluate all `--sources` against it. Free GPU memory (`del model; torch.cuda.empty_cache()`) before loading the next checkpoint. The PRD is explicit: do not reload the same model separately for GSM8K and MATH.
- Use the existing `finpost.safety.load_model_safely` (or whatever name it uses) — see `src/finpost/safety.py`. **Do not bypass the safety wrapper** even if it adds a tiny bit of friction.
- Tokenizer loaded from the same path / id as the model. Padding side `left` for generation (HF convention for decoder-only).
- dtype: `torch.bfloat16` on CUDA capability >= (8, 0), else `torch.float16`, else `torch.float32` on CPU.

### Batched generation with OOM fallback

- For each source, partition the n examples into batches of `batch_size_<source>`.
- For each batch:
  1. Tokenize prompts with left padding to the longest prompt in the batch.
  2. Call `model.generate` with `do_sample=False`, `max_new_tokens` from the source registry, `use_cache=True`.
  3. Decode only the *new* tokens (slice off the prompt tokens before decoding).
  4. Run the source's `extract_answer` on each generation.
  5. Score and accumulate.
- On `torch.cuda.OutOfMemoryError`, halve the batch size and retry the SAME batch. If batch_size hits 1 and still OOMs, raise — do not silently downsize forever.
- Generated token count for cost: sum of `(generated_token_ids.shape[-1])` across all batches per source.

### Output artifacts

Five files under `--out-dir`:

```
accuracy_summary.json    # list of {checkpoint, source, n, accuracy, parse_success_rate, generated_tokens, elapsed_sec}
accuracy_summary.csv     # same data, CSV
details_<ckpt>_<src>.csv # per-example: example_id, prompt, generated, gold_answer, predicted_answer, parse_success, is_correct
run_metadata.json        # device, dtype, versions, seed, generation settings per source, git short SHA
cost_summary.json        # run_name, start_time, end_time, elapsed_sec, gpu_type, dtype, generated_tokens, tokens_per_second, estimated_cost_usd (nullable)
```

CSV writing: use the stdlib `csv` module. Quote everything that contains a newline or comma (the `generated` column will). Do not use pandas — keep dependencies thin.

### Cost helper — inline, not a module

A small class or context manager at the top of `eval_exact.py`. Roughly:

```python
class RunTracker:
    def __init__(self, out_dir, gpu_cost_per_hour=None): ...
    def __enter__(self): ...  # record start_time, GPU name
    def __exit__(self, ...): ...  # record end_time
    def add_generated_tokens(self, n): ...
    def write(self): ...  # write cost_summary.json
```

Approximately 30–50 lines. No separate file.

### Git short SHA

```python
import subprocess
sha = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"]).decode().strip()
```

Wrap in try/except — if the repo is in a detached or weird state, write `"unknown"` to the metadata and continue. Do not fail the eval over a git problem.

## Acceptance criteria

1. `python -m finpost.evals.eval_exact --help` prints all flags.
2. `python -m finpost.evals.eval_exact --checkpoints tiny=sshleifer/tiny-gpt2 --sources gsm8k --n 4 --seed 42 --out-dir /tmp/smoke --device cpu` runs to completion and writes all five files. (Tiny model + n=4 is the unit-test-sized run.)
3. `accuracy_summary.csv` has exactly N rows where N = `len(checkpoints) * len(sources)`, with columns: `checkpoint`, `source`, `n`, `accuracy`, `parse_success_rate`, `generated_tokens`, `elapsed_sec`.
4. `details_<ckpt>_<src>.csv` has exactly `n` rows with columns: `example_id`, `prompt`, `generated`, `gold_answer`, `predicted_answer`, `parse_success`, `is_correct`.
5. Re-running the same command produces byte-identical `details_*.csv` (greedy + fixed seed + fixed dtype + same device).
6. `run_metadata.json` contains all expected fields per the PRD's run_metadata schema.
7. `cost_summary.json` contains all expected fields; `estimated_cost_usd` is `null` when `--gpu-cost-per-hour` is not supplied, populated when it is.
8. OOM fallback test: a mock that raises `torch.cuda.OutOfMemoryError` on first call, succeeds on second, results in `batch_size` halving and the batch retrying. After successful retry, evaluation continues normally.
9. `pytest tests/test_eval_exact.py -v` passes.
10. `pytest` (whole suite) still passes.

## Notes / open questions

- The user's standing preference: every line of code intelligible, no library magic. The generation loop should be hand-explicit, not wrapped in a `Trainer.predict()` style API. Pedagogical comments are welcome where they explain why (e.g., "left padding because decoder-only models read from the right edge").
- This file will likely be 400–600 lines. That is fine. Resist splitting prematurely. The PRD explicitly chose against separate `run_tracker.py` / `make_phase1_report.py` modules — keep things inline until a second consumer earns the split.
- For OOM detection: catch `torch.cuda.OutOfMemoryError` specifically. Do not catch generic `RuntimeError` and string-match — that catches unrelated bugs.
- The `tiny-gpt2` model has no chat template and will likely produce garbage generations on math prompts. That is fine for testing the *infrastructure*. The smoke run in issue 04 uses a real Qwen 0.5B base for an actual answer.
- The smoke test in acceptance criterion 2 is intentionally small (n=4) so it runs in seconds on CPU. The full n=500 smoke run lives in issue 04.
