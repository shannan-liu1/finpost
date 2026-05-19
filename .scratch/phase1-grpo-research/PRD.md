# Group Relative Policy Optimization research track for verifiable numerical reasoning

- **Status:** Superseded by [`finchain-rlvr-posttraining`](../finchain-rlvr-posttraining/PRD.md)
- **Created:** 2026-05-08
- **Owner:** Shannan
- **Estimated time:** 3-5 days for research/design, 1-2 weeks for a first toy implementation after SFT and DPO land
- **Depends on:** [`phase1-sft-trainer`](../phase1-sft-trainer/PRD.md), [`phase1-compute-aware-post-training`](../phase1-compute-aware-post-training/PRD.md), Phase 1 Direct Preference Optimization workstream, evaluation harness

## Goal

Define a future Group Relative Policy Optimization (GRPO) research track that can extend the Qwen 0.5B post-training stack after Supervised Fine-Tuning and Direct Preference Optimization are implemented and evaluated.

This workstream exists to preserve the idea without letting it distort the current trainer scope. GRPO is post-training, but it is not a drop-in replacement for Supervised Fine-Tuning or Direct Preference Optimization.

## Amendment 2026-05-19: Superseded by FinChain RLVR path

The original Qwen 0.5B GRPO research track is no longer the active direction. The useful ideas here - verifier rewards, grouped samples, normalized advantages, and KL logging - move into [`finchain-rlvr-posttraining`](../finchain-rlvr-posttraining/PRD.md), where FinChain provides a cleaner symbolic-reasoning reward substrate and a 3B/4B model gives a more credible finance result.

Keep this PRD as background only. New GRPO implementation work should start from the FinChain PRD and [`docs/runbooks/finchain-rlvr-study-flow.md`](../../docs/runbooks/finchain-rlvr-study-flow.md).

## Scope

**In scope:**
- A research brief explaining how GRPO differs from Supervised Fine-Tuning, Direct Preference Optimization, and reinforcement fine-tuning with a learned value model.
- A reward contract for verifiable math and filing-excerpt numerical reasoning.
- A toy GRPO implementation plan that can run on Qwen 0.5B after the SFT/DPO path is stable.
- Tests for reward functions before any policy update loop is trusted.
- A decision gate that determines whether GRPO is worth implementing after DPO results are known.

**Out of scope:**
- Changing the current Phase 1 SFT trainer.
- Replacing Direct Preference Optimization in the current plan.
- Running GRPO before the evaluation harness and verifier exist.
- Using an LLM judge as the only reward for numerical correctness.
- Scaling to larger models before Qwen 0.5B has shown stable, reproducible learning.

## Deliverables

- `.scratch/phase1-grpo-research/PRD.md`
  - This scope and sequencing document.
- Future research brief:
  - Recommended path: `docs/primers/grpo.md`
  - Explain GRPO from first principles.
  - Compare GRPO to PPO, DPO, and supervised fine-tuning.
- Future reward module:
  - Recommended path: `src/finpost/rl/rewards.py`
  - Reward components for final-answer correctness, citation faithfulness, computation correctness, and format compliance.
- Future tests:
  - Recommended path: `tests/test_rewards.py`
  - Unit tests for each reward component and reward aggregation.
- Future toy runner:
  - Recommended path: `scripts/grpo_toy_run.py`
  - Runs on a tiny model or Qwen 0.5B with a very small prompt set.

## Acceptance criteria

1. No GRPO implementation starts until the Qwen 0.5B SFT trainer can run end to end and produce a checkpoint.
2. No GRPO implementation starts until DPO has at least one completed baseline or an explicit decision says DPO is being skipped.
3. The reward contract is written before the training loop and includes failure examples.
4. Numeric correctness rewards are programmatic, not judge-only.
5. LLM-as-judge rewards, if used, are limited to answerability, citation quality, explanation faithfulness, or preference labels where programmatic checks are insufficient.
6. The first GRPO test uses a tiny prompt set and proves that sampled completions, grouped rewards, normalized advantages, and KL logging are observable.
7. A final decision gate compares GRPO's added complexity against DPO's measured results before any larger run is approved.

## Implementation plan

### Slice 1 - Research brief and reward contract

**Behavior contract:** A future reader can explain when GRPO is appropriate and what each reward component means.

**Files:**
- Create `docs/primers/grpo.md`
- Optionally amend `STUDY.md` with a short cross-link

**Verification:**
- `git diff --check -- docs/primers/grpo.md STUDY.md`
- Manual check: every reward component has at least one positive and one negative example.

### Slice 2 - Programmatic reward functions

**Behavior contract:** Given a candidate answer and a reference item, reward functions return deterministic scores in `[0, 1]`.

**Files:**
- Create `src/finpost/rl/rewards.py`
- Create `tests/test_rewards.py`

**Test plan:**
- First RED test: exact numeric answer match returns 1 and mismatch returns 0.
- Command: `python -m pytest tests/test_rewards.py -v`
- Expected RED signal: import failure for `finpost.rl.rewards`.
- Expected GREEN signal: all reward tests pass.

### Slice 3 - Toy grouped sampling loop

**Behavior contract:** For each prompt, sample `k` completions, score them, normalize scores within the group, and log the resulting relative advantages without updating model weights.

**Files:**
- Create `scripts/grpo_toy_run.py`
- Add tests only around pure helpers; do not test heavyweight model generation in unit tests.

**Verification:**
- `python scripts/grpo_toy_run.py --tiny-model --prompts 4 --samples-per-prompt 4`
- Expected output includes grouped rewards, normalized advantages, and per-prompt summary statistics.

### Slice 4 - Policy update loop

**Behavior contract:** Only after the previous slices are stable, add policy-gradient updates with KL control against a frozen reference model.

**Stop condition:** If the reward functions are easy to exploit, stop and improve the verifier instead of training.

## Notes / open questions

- GRPO is attractive for this project because math and filing-excerpt numerical reasoning can produce verifiable rewards.
- It is dangerous if introduced too early because the model can learn to exploit weak reward prompts or superficial formatting.
- The pedagogical sequence remains: Supervised Fine-Tuning first, Direct Preference Optimization second, GRPO third.
- "GPRO" in casual notes should be normalized to "GRPO" in repo documentation.

## Amendment 2026-05-11 — reuse Phase 1.5 rollout and verifier plumbing

The original implementation plan implied this workstream would build its own rollout/sampler/verifier stack. Those components are now built in [`phase1-compute-aware-post-training`](../phase1-compute-aware-post-training/PRD.md) and exposed under `src/finpost/posttraining/` (`rollout.py`, `verifier.py`, `bucket.py`, `cost_ledger.py`). This workstream consumes them and adds only:

- grouped advantage normalization,
- KL control against a frozen reference model,
- the policy-gradient update step,
- reward-component aggregation beyond final-answer correctness (citation faithfulness, computation correctness, format compliance — when needed for Phase 2),
- tests for reward functions and the policy-update loop.

The cost ledger format and the rollout cache format are owned by Phase 1.5 and reused unchanged.
