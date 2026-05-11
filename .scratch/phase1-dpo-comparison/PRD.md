# Phase 1 DPO trainer and SFT comparison

- **Status:** Not Started
- **Created:** 2026-05-09
- **Owner:** Shannan
- **Estimated time:** ~1 week after Phase 1 SFT trainer lands
- **Depends on:** [`phase1-sft-trainer`](../phase1-sft-trainer/PRD.md)

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

- Open decision Q-B from `PLAN.md` ("how to handle prompts where all sampled completions are correct or all are incorrect") is resolved by construction in this workstream's preference-pair builder: prompts with all-correct or all-incorrect samples contribute zero pairs and are tracked separately as a model-quality signal. The Phase 1.5 builder applies the same rule independently.
- DPO should not start until the SFT checkpoint is real. Otherwise there is no meaningful policy model to improve.

## Amendment 2026-05-11 — DPO stays offline; OPD stays on-policy

This workstream and [`phase1-compute-aware-post-training`](../phase1-compute-aware-post-training/PRD.md) deliberately use **separate** preference-pair pipelines. The split is itself a comparison axis:

- DPO uses a **fixed offline preference dataset**: sample N=8 completions per held-out training prompt from the SFT-best checkpoint **once**, grade with the same verifier ladder Phase 1.5 uses, build pairs once, and train DPO against that frozen dataset.
- OPD uses an **on-policy** preference dataset: rollouts are sampled fresh from the current training policy at each scheduled refresh, with adaptive K on ambiguous prompts.

Both pipelines share:
- the verifier ladder under `src/finpost/postraining/verifier.py`,
- the DPO-style pairwise loss math (per-example loss must match within `1e-5` on uniform inputs — the parity test lives here).

Each pipeline owns its own:
- rollout cache,
- preference-pair builder,
- training driver.

The combined Phase 1 evaluation surface then has at least four arms — Base, SFT, SFT+DPO (this workstream), SFT+OPD (Phase 1.5) — measured on the same harness. A merged "preference-pair builder" abstraction is explicitly deferred until both pipelines have produced a first result; merging earlier would collapse the offline-vs-on-policy distinction.

Deliverables restored: `scripts/build_dpo_pairs.py` and `src/finpost/training/preference_data.py` belong to this workstream.
