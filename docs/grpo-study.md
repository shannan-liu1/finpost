# GRPO Study Guide

Status: backend primitive implemented; RunPod TRL launch script and notebook
implemented for FinChain 1.5B runs.

## 1. Plain-English Idea

Group Relative Policy Optimization samples several completions for the same
prompt, scores them, and asks: which completions were better than their local
group?

The useful FinChain loop is:

1. Take one financial reasoning prompt.
2. Sample `K` completions from the current policy.
3. Grade each completion with the deterministic FinChain verifier.
4. Normalize rewards inside that prompt group.
5. Increase token probability for above-average completions.
6. Decrease token probability for below-average completions.
7. Penalize drift away from a reference policy.

This is the cleanest RLVR story in the repo because FinChain supplies a cheap
verifier reward.

## 2. The Math

For one prompt group:

```text
reward_j = verifier(completion_j)
advantage_j = (reward_j - mean(reward_group)) / std(reward_group)
```

The policy update uses a token-level probability ratio:

```text
ratio_t = exp(log pi_theta(token_t) - log pi_old(token_t))
```

Then the simplified token objective is:

```text
loss_t = -(ratio_t * advantage - beta * KL(pi_theta || pi_ref))
```

In the repo primitive, the KL approximation is:

```text
exp(logp_ref - logp_policy) - (logp_ref - logp_policy) - 1
```

The core math lives in `src/finpost/posttraining/grpo.py`.

## 3. Why The Repo Keeps A Primitive

The primitive accepts tensors and does not know about tokenizers, Hugging Face
models, or rollout workers. That is intentional. The learning value is seeing:

- group-relative advantages,
- old-policy log-probs,
- reference log-probs,
- response masks,
- KL control.

The production value comes from using TRL or another RLHF/RLVR framework for
the systems work.

## 4. RunPod Notebook And Script

- Notebook: `notebooks/finchain_12_grpo_runpod.ipynb`
- Script: `scripts/train_finchain_trl_grpo.py`
- Reward adapter: `src/finpost/posttraining/finchain_rlvr.py`

Single GPU canary:

```bash
python scripts/train_finchain_trl_grpo.py \
  --model Qwen/Qwen2.5-1.5B \
  --train-n 256 \
  --max-steps 20 \
  --num-generations 4 \
  --max-completion-length 512 \
  --per-device-train-batch-size 1 \
  --gradient-accumulation-steps 8 \
  --output-dir results/checkpoints/qwen25-1p5b-finchain-grpo-canary
```

Two-GPU run:

```bash
accelerate launch --num_processes 2 scripts/train_finchain_trl_grpo.py \
  --model Qwen/Qwen2.5-1.5B \
  --train-n 2000 \
  --max-steps 300 \
  --num-generations 4 \
  --max-completion-length 512 \
  --per-device-train-batch-size 1 \
  --gradient-accumulation-steps 8 \
  --output-dir results/checkpoints/qwen25-1p5b-finchain-grpo-2gpu
```

## 5. Efficiency Reasoning

The cost center is generation. Start with:

- `num_generations=4`, not 8 or 16;
- `max_completion_length=512`, not 1024, until you inspect truncation;
- `train_n=256` for the canary;
- one A40 before multi-GPU unless the canary proves generation is the bottleneck.

Add vLLM only after the non-vLLM run works. vLLM can improve rollout throughput,
but it adds another integration surface.

## 6. Failure Modes

- Reward hacking: parseability rises but final-answer accuracy does not.
- KL spike: reward rises while the model drifts into brittle answer templates.
- All-zero advantages: every completion in a group has the same reward.
- Template overfit: FinChain accuracy improves but FinQA transfer does not.

## 7. Industry Package Mapping

- Hugging Face TRL `GRPOTrainer`: best first production API for this project.
- verl: useful when you want distributed RL rollouts and training architecture.
- OpenRLHF: useful for larger RLHF/RLVR system design with Ray/vLLM style
  orchestration.
- vLLM: useful specifically for high-throughput rollout generation.
