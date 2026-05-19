# GRPO Study Guide

Status: backend primitive implemented; full FinChain trainer still pending.

## What GRPO Is Doing

Group Relative Policy Optimization samples multiple completions for the same
prompt, scores each completion, and learns from the completion's reward
relative to the other completions in that prompt group. The useful FinChain
shape is:

1. Sample `K` completions from the current policy for one prompt.
2. Verify each completion with the FinChain verifier.
3. Convert rewards into group-relative advantages:
   `advantage = (reward - group_mean) / group_std`.
4. Increase token log-probability for positive-advantage completions and
   decrease it for negative-advantage completions.
5. Penalize drift from a frozen reference policy with a KL term.

The core repo implementation lives in `src/finpost/posttraining/grpo.py`.

## Why This Backend Is Tensor-First

The backend does not know about tokenizers, Hugging Face model classes, rollout
files, or notebooks. It accepts tensors:

- `rewards`: grouped verifier rewards shaped `(prompt_count, group_size)`.
- `policy_logps`: token log-probs under the trainable policy.
- `old_logps`: token log-probs under the policy that sampled the rollout.
- `ref_logps`: token log-probs under the frozen reference model.
- `advantages`: one scalar per sampled completion.
- `response_mask`: masks prompt and padding positions out of the loss.

That separation is deliberate. Rollout generation is expensive and operational;
the GRPO objective should remain cheap to unit-test.

## Efficiency Reasoning

The expensive operations are model forwards, not reward normalization.

- Advantage normalization is vectorized across `(prompt, group)` tensors.
- The loss consumes precomputed `old_logps` and `ref_logps`; later trainer code
  can cache those during rollout/reference scoring instead of recomputing them
  inside every optimizer step.
- The KL term uses the common Schulman-style approximation:
  `exp(logp_ref - logp_policy) - (logp_ref - logp_policy) - 1`.
- Applying a sequence-level advantage to every response token is simple and
  matches the grouped rollout objective without a critic model.

## FinChain Starting Point

Use binary final-answer correctness first:

```text
correct completion -> reward 1.0
incorrect or parse-failed completion -> reward 0.0
```

Do not add shaped rewards until the binary reward exposes its failure modes.
Shaped parseability or chain-validity rewards can make curves look better while
teaching answer-format shortcuts. The first GRPO run should report:

- group reward mean and standard deviation,
- parse success rate,
- final-answer accuracy,
- KL/reference drift,
- examples where parseability improved without correctness,
- examples where the verifier was reward-hacked.

## References

- DeepSeekMath introduced GRPO as a critic-free grouped policy optimization
  method for mathematical reasoning.
- Hugging Face TRL's `GRPOTrainer` is the main implementation reference for
  the modern LLM post-training interface.
