# 04. Preference-pair builder (Stage 3)

- **Status:** Not Started
- **Ready for agent:** No (gated on issues 02 and 03)
- **Created:** 2026-05-11
- **Estimated time:** ~1 day
- **Depends on:** issue 02 (rollout cache), issue 03 (verifier and bucketing)

## Goal

From a verified rollout cache and bucket assignment, emit a preference dataset with `(prompt, chosen, rejected, bucket, train_weight, source_checkpoint, source_revision, sample_idx_chosen, sample_idx_rejected)` records.

## Scope

**In scope:**
- `src/finpost/postraining/preference.py`.
- Configurable pair policy: `all_pairs`, `best_vs_worst`, `one_random_pair_per_prompt`.
- Default `train_weight` schedule: `{easy: 0.25, ambiguous: 1.0, hard: 0.5}` — overridable in config. **TODO: think about this schedule before Stage 5 launches.** No strong prior. Alternatives: continuous weight `1 - |2 * p_correct - 1|`; asymmetric `{easy: 0.25, ambiguous: 1.0, hard: 0.0}`; bucket-count-balanced.
- Reproducibility: deterministic pair selection given the same `(rollout_cache_hash, pair_policy, seed)`.
- The same dataset shape is reusable by [`phase1-dpo-comparison`](../../phase1-dpo-comparison/PRD.md) as its preference dataset.

**Out of scope:**
- The OPD trainer itself (issue 05).
- Any LLM-as-judge ranking. Pairs are derived solely from verifier correctness.

## Acceptance criteria

1. `pytest tests/test_preference.py` passes.
2. Determinism: emitting pairs twice with the same `(rollout_cache_hash, pair_policy, seed)` produces byte-identical output.
3. Bucket assignment on each pair record matches the bucketing output from issue 03.
4. Prompts with all-correct or all-incorrect rollouts contribute zero preference pairs (resolves PLAN.md Q-B by construction).
5. The default `train_weight` schedule is overridable from the config without code changes.
