# Phase 1 DPO trainer and SFT comparison

- **Status:** Not Started
- **Created:** 2026-05-09
- **Owner:** Shannan
- **Estimated time:** ~1 week after Phase 1 SFT trainer lands
- **Depends on:** [`phase1-sft-trainer`](../phase1-sft-trainer/PRD.md), [`phase1-compute-aware-post-training`](../phase1-compute-aware-post-training/PRD.md)

## Goal

Add Direct Preference Optimization after the Qwen 0.5B SFT baseline, then compare pure SFT against SFT + DPO on the same Phase 1 math evaluation surface.

This workstream starts only after the TinyGPT local canary, Qwen 20-step SFT soft launch, and first Qwen SFT baseline have completed.

## Scope

**In scope:**
- Generate model completions from the best SFT checkpoint on held-out training prompts.
- Programmatically grade completions using final-answer correctness.
- Build chosen/rejected preference pairs.
- Implement DPO loss from scratch and verify numerical parity against a reference implementation.
- Run a short DPO soft launch, then a full DPO baseline.
- Compare Base, SFT, and SFT + DPO on the same held-out math evaluation set.

**Out of scope:**
- GRPO or any online reinforcement-learning loop.
- Finance/10-K data.
- Larger model scale-up before Qwen 0.5B results are known.
- Treating an LLM judge as the source of truth for math correctness.

## Deliverables

- `.scratch/phase1-dpo-comparison/issues/01-build-preference-pairs.md`
- `.scratch/phase1-dpo-comparison/issues/02-dpo-loss-and-parity.md`
- `.scratch/phase1-dpo-comparison/issues/03-dpo-trainer-soft-launch.md`
- `.scratch/phase1-dpo-comparison/issues/04-compare-sft-vs-dpo.md`

Future code deliverables:

```
src/finpost/training/dpo.py
src/finpost/training/preference_data.py
scripts/build_dpo_pairs.py
experiments/dpo_baseline.yaml
tests/test_dpo.py
tests/test_preference_data.py
```

## Acceptance criteria

1. Preference-pair generation produces a dataset with prompt, chosen response, rejected response, grading reason, and source checkpoint metadata.
2. DPO loss matches the reference loss within `1e-5` on a fixed tiny batch.
3. A TinyGPT DPO soft launch runs end to end with offline tracking and checkpointing.
4. A Qwen 0.5B DPO soft launch runs after the Qwen SFT checkpoint exists.
5. The comparison report includes Base vs. SFT vs. SFT + DPO final-answer accuracy, confidence intervals, response-length statistics, and failure examples.
6. The result explicitly states whether DPO improved the SFT model, hurt it, or was statistically indistinguishable.

## Notes / open questions

- Open decision Q-B from `PLAN.md` lives here: how to handle prompts where all sampled completions are correct or all are incorrect. **Resolved 2026-05-11 by Phase 1.5 Stage 3 (preference-pair builder): all-correct and all-incorrect prompts contribute zero pairs.**
- This workstream now reuses the rollout cache, verifier, and preference-pair dataset produced by [`phase1-compute-aware-post-training`](../phase1-compute-aware-post-training/PRD.md). The "build preference pairs" issue in this workstream becomes a consumption-side issue: load the Phase 1.5 dataset, confirm bucket metadata is intact, and run the comparison.
- DPO should not start until the SFT checkpoint is real and Phase 1.5 Stage 3 has emitted a preference dataset. Otherwise there is no meaningful policy model to improve and no on-policy preference data to compare offline DPO against.

## Amendment 2026-05-11 — preference data sourced from Phase 1.5

The original deliverable list for this workstream included `scripts/build_dpo_pairs.py` and `src/finpost/training/preference_data.py`. These are superseded by the Phase 1.5 rollout, verifier, bucketing, and preference-pair modules under `src/finpost/postraining/`. This workstream now owns:

- the DPO loss implementation (`src/finpost/training/dpo.py`),
- the numerical parity test against the Phase 1.5 OPD loss on uniform inputs,
- the Base vs. SFT vs. SFT+DPO vs. SFT+OPD comparison report.

The comparison adds **SFT+OPD** as a fourth arm so that offline DPO and On-Policy Distillation can be measured on the same evaluation surface. Whether they should remain separate workstreams or be merged is a decision deferred until both have produced a first result.
