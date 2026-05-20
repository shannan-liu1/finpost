# Context: finpost

`finpost` is a learning project for becoming fluent in modern language-model post-training by building the training loops, evaluation harnesses, and experiment artifacts directly. The active domain is financial numerical reasoning, but the product is the learning artifact: notebooks, study guides, runbooks, and results that make the user credible when discussing supervised fine-tuning, preference learning, and reinforcement learning with verifiable rewards.

The active benchmark direction is **FinChain-first**. FinChain is a symbolic financial reasoning benchmark with executable chains and deterministic answer verification. It is a better primary substrate than FinQA for the next phase because it gives us dense, cheap, programmatic reward signals for Reinforcement Learning with Verifiable Rewards (RLVR). FinQA remains valuable as a transfer and realism check, not as the main training surface.

## Project Intent

- **Primary goal (~70%):** Learn post-training fundamentals deeply enough to explain, implement, debug, and compare them in interviews. Documentation should spell out terms on first use, connect concepts to code, and distinguish controls from headline methods.
- **Secondary goal (~30%):** Produce a credible finance-domain artifact. The final story should be: "I built a compute-aware RLVR study for financial reasoning, starting from small-model supervised fine-tuning and ending with verifier-driven OPD/GRPO comparisons under measured compute budgets."
- **Success bar:** A reproducible study flow with notebooks, configs, cost ledgers, and a writeup that explains what improved, what failed, and why. Not: a production finance analyst model.

## Active Benchmark Ladder

1. **FinChain as the primary RLVR substrate.** Use it for training and method comparison because examples include symbolic financial reasoning chains that can be executed or checked deterministically.
2. **FinQA as the transfer benchmark.** Use it to test whether FinChain-trained behavior survives contact with messier real filing excerpts and table/text grounding.
3. **TAT-QA or ConvFinQA as optional transfer checks.** Add one only after the FinChain -> FinQA result is clean.
4. **FinanceBench or SEC filing retrieval as later work.** These are higher-friction, less verifier-clean, and not the minimum effective dose for learning RLVR.

GSM8K and MATH remain infrastructure smoke tests for the trainer, verifier, and cost ledger. They are not the active finance benchmark.

## Target Capability

The model is being post-trained to perform verifiable financial numerical reasoning:

1. **Formula selection:** identify the relevant financial relation, such as gross margin, year-over-year growth, debt-to-equity, or free cash flow.
2. **Grounded computation:** select the correct values from the prompt context and execute the arithmetic.
3. **Chain validity:** produce reasoning steps that can be checked against an executable symbolic chain or final numeric verifier.
4. **Format discipline:** emit a parseable final answer so that evaluation failures reflect reasoning quality, not output-format drift.

Explicitly out of scope:

- Open-ended investment analysis or qualitative judgment
- Cross-document retrieval and reconciliation
- LLM-as-judge as the primary correctness signal
- Full fine-tuning of 3B/4B models before LoRA/QLoRA baselines are exhausted
- Large method/model grids whose results cannot be explained cleanly

## Method Ladder

The study intentionally separates controls from RLVR methods:

1. **Base / few-shot evaluation:** establishes what the model already knows.
2. **Supervised Fine-Tuning (SFT):** teaches format, domain vocabulary, and chain style from gold examples. This is the anchor, not the headline RL method.
3. **Rejection SFT:** trains on verified-correct model generations. This is the cheapest self-improvement baseline.
4. **Direct Preference Optimization (DPO):** trains on a fixed offline preference dataset. In this repo it is mainly a fundamentals and comparison artifact.
5. **On-Policy Distillation (OPD):** builds DPO-style preference pairs from the current policy's own verified rollouts. This is the practical bridge between preference learning and RLVR.
6. **Group Relative Policy Optimization (GRPO):** samples groups of completions, scores them with the verifier, normalizes rewards within each group, and updates the policy with KL control. This is the headline RLVR method for the FinChain phase.

SFT and DPO are not themselves RLVR in the strict sense. OPD and GRPO are the more direct RLVR-shaped methods because their supervision comes from current-policy rollouts scored by a programmatic verifier.

## Model And Hardware Posture

- **Canary model:** keep `Qwen/Qwen2.5-0.5B` for CPU/local tests, notebook debugging, and cheap trainer regression checks.
- **Default serious model:** use `Qwen/Qwen2.5-1.5B` for the main FinChain loop. It is large enough to move beyond the 0.5B canary, small enough for fast iteration, and keeps the Qwen tokenizer/tooling path stable.
- **Scale-up candidate:** use `Qwen/Qwen3-4B-Base` only after the 1.5B loop is interpretable and the scaling question is explicit.
- **Reference baselines:** optionally evaluate a strong instruct model such as Phi-3.5 Mini Instruct or Llama 3.2 3B, but do not make it the main train-from-base substrate.
- **Normal GPU:** use one 48GB RunPod-class GPU such as A40, L40S, RTX 6000 Ada, or A6000 for LoRA/QLoRA, rollouts, OPD, and a first GRPO run.
- **Cluster experiment:** use 2x or 4x A100/H100 only after a single-GPU study is reproducible. The cluster story should be about scaling rollout throughput and grouped RL updates, not rescuing an unclear experiment.

## Final Artifact Shape

The project should produce:

- A FinChain dataset/eval notebook
- A FinChain SFT notebook
- A rollout/verifier/cache notebook
- An OPD notebook
- A GRPO notebook
- A final comparison notebook with accuracy, pass@K, parseability, KL drift, rollout tokens, GPU-hours, and dollars
- A study guide explaining SFT, DPO, OPD, PPO, GRPO, RLHF, RLVR, KL control, and reward hacking using this repo's code

## Glossary

### FinChain

A financial symbolic reasoning benchmark with parameterized templates, generated examples, and executable reasoning chains. In this project, FinChain is the primary RLVR training and evaluation substrate because it gives deterministic final-answer and step-level verification.

### FinQA

A financial question-answering benchmark over S&P 500 annual and quarterly report excerpts. FinQA remains a transfer benchmark for testing whether FinChain-trained reasoning generalizes to messier real filing contexts.

### Reinforcement Learning with Verifiable Rewards

A post-training setup where model completions are scored by an objective verifier rather than a human preference label or language-model judge. Abbreviated RLVR after first use.

### Reinforcement Learning from Human Feedback

A post-training setup where human preference data is used directly or through a learned reward model to improve model behavior. Abbreviated RLHF after first use.

### Supervised Fine-Tuning

A post-training method that updates a model on prompt-response examples. The model imitates the provided response tokens, usually with loss applied only to the assistant response. Abbreviated SFT after first use.

### Direct Preference Optimization

A post-training method that updates a model from fixed pairs of responses to the same prompt: one preferred response and one non-preferred response. Abbreviated DPO after first use.

### On-Policy Distillation

A post-training method that samples completions from the current policy, verifies them, builds chosen/rejected pairs, and applies a DPO-style pairwise loss. Abbreviated OPD after first use.

### Group Relative Policy Optimization

A reinforcement-learning method that samples multiple responses for the same prompt, scores each with a reward function, normalizes rewards within the group, and applies a KL-controlled policy update. Abbreviated GRPO after first use.

### Proximal Policy Optimization

A policy-gradient reinforcement-learning method that constrains each update so the new policy does not move too far from the old policy. PPO is foundational for understanding RLHF, even if this repo does not implement it first.

### KL Control

A penalty or constraint that discourages a trained policy from drifting too far from a reference policy. In this project, KL control is tracked because verifier rewards can be gamed by format hacks, shortcut reasoning, or distribution collapse.

### Rollout

A batch of completions generated by a policy model for a fixed prompt set. Rollouts are cached by model revision, prompt revision, and sampling parameters so verification and pair construction do not regenerate expensive samples.

### Verifier

A deterministic or mostly deterministic function that scores a model completion. In this project the verifier should parse the final answer, execute or compare symbolic chains when available, and return correctness plus diagnostics.

### Difficulty Bucket

A label assigned to a prompt based on the fraction of K rollout samples that verify as correct. Default buckets are `easy`, `ambiguous`, and `hard`. Ambiguous prompts are usually the highest-value prompts for additional sampling and preference updates.

### Cost Ledger

A per-run record of rollout tokens, verifier calls, training tokens, wall-clock GPU-hours, estimated dollars, parseability, accuracy, and accuracy per cost unit. The cost ledger is a first-class result, not bookkeeping after the fact.
