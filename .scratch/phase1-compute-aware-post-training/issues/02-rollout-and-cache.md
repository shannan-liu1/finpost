# 02. Rollout module and on-disk cache (Stage 1 + Stage 2 plumbing)

- **Status:** Not Started
- **Ready for agent:** No (gated on issue 01)
- **Created:** 2026-05-11
- **Estimated time:** ~3 days
- **Depends on:** issue 01

## Goal

Implement the rollout module. Given `(checkpoint, prompts, sampling_params)` it samples K completions per prompt, writes them to a deterministic on-disk cache, and refuses to regenerate samples that already exist for the same key.

## Scope

**In scope:**
- `src/finpost/posttraining/rollout.py` with a batched sampler.
- `scripts/run_rollout.py` CLI: `--checkpoint`, `--prompts`, `--k`, `--temperature`, `--max-new-tokens`, `--append`.
- Cache key: `sha256(model_revision || prompt_id || sampling_params_canonical_json)`.
- Cache file format: jsonl with `(prompt_id, sample_idx, completion, parsed_answer, model_revision, sampling_params_hash, generated_at)`.
- `--append` mode adds new samples without rewriting existing ones; re-running with the same args is a no-op.

**Out of scope:**
- Verifier execution (issue 03).
- Bucketing (issue 03).
- Preference-pair construction (issue 04).

## Acceptance criteria

1. `python scripts/run_rollout.py --checkpoint <best_sft> --prompts data/processed/train_prompts.jsonl --k 4` writes a rollout file with 4 completions per prompt.
2. Running the same command a second time generates zero new completions and exits in under 5 seconds.
3. `python scripts/run_rollout.py --checkpoint <best_sft> --prompts <ambiguous_subset> --k 12 --append` appends without regenerating; resulting cache has `4 + 12 = 16` completions per ambiguous prompt and unchanged 4 per non-ambiguous prompt.
4. `pytest tests/test_rollout_cache.py` passes — covers cache hit, cache miss, partial fill, and append semantics.
5. The cost ledger row written by the rollout records `rollout_tokens`, `wall_clock_seconds`, `gpu_type`, `effective_$/run` (using the rate from the cost-gate checklist).
