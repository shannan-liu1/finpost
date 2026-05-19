# FinChain RLVR Study Flow

Status: active direction as of 2026-05-19.

## Thesis

What would this look like if it were easy?

It would be one clean study:

> FinChain gives deterministic financial reasoning rewards. Use it to compare SFT, rejection SFT, OPD, and GRPO on one serious small model under a fixed compute budget, then test transfer on FinQA.

The project should produce artifacts that are easy to run, inspect, and explain:

- notebook outputs
- configs
- result tables
- cost ledgers
- failure examples
- a study guide connected to the implementation

## Why FinChain

FinQA is a good finance benchmark, but it is not the easiest RLVR benchmark. It is realistic and messy, but the supervision is mostly final-answer numeric correctness over filing excerpts.

FinChain is more useful for the next phase because:

- symbolic chains make verification cheaper and more diagnostic
- parameterized templates make it possible to analyze topic-level and formula-level failures
- generated examples let us create controlled train/test splits
- deterministic rewards make OPD and GRPO easier to debug

The risk is template overfit. That is why FinQA stays in the plan as a transfer benchmark.

## Method Roles

### SFT

Supervised Fine-Tuning teaches the model the response format, financial vocabulary, and chain style. It is the anchor baseline.

Interview sentence:

> SFT taught the model the task interface and reasoning trace format; it did not prove the model could improve from verified self-generated reasoning.

### DPO

Direct Preference Optimization is a fundamentals and comparator artifact. It teaches the pairwise preference loss and reference-policy framing.

It is not the highest-leverage mainline FinChain path unless the OPD/GRPO results need a fixed offline preference comparator.

Interview sentence:

> I implemented DPO to understand preference optimization, but I did not make offline DPO the main finance experiment because FinChain gives a better on-policy verifier signal.

### Rejection SFT

Rejection SFT trains on the model's own verified-correct completions. It is the simplest self-improvement baseline.

Interview sentence:

> Rejection SFT answered whether verified self-generated positives alone were enough before adding pairwise or RL updates.

### OPD

On-Policy Distillation samples from the current policy, verifies completions, builds chosen/rejected pairs, and applies a DPO-style pairwise loss. This is the bridge method.

It is promising because it reuses DPO mechanics while moving the data distribution on-policy.

Interview sentence:

> OPD was the practical bridge from DPO to RLVR: same pairwise math, but the pairs came from the current policy and a deterministic verifier.

### GRPO

Group Relative Policy Optimization samples a group of completions for each prompt, scores them with the verifier, normalizes rewards within the group, and applies a KL-controlled update.

It is the most promising headline method because FinChain supplies exactly the kind of group-level verified rewards GRPO needs.

Interview sentence:

> GRPO was the clean RLVR experiment: grouped rollouts, verifier rewards, relative advantages, and KL control against a reference policy.

## Recommended Method Scope

Do not do all methods equally deeply.

Minimum effective dose:

1. Base / few-shot
2. SFT
3. Rejection SFT
4. Adaptive OPD
5. One GRPO run

Optional if time:

- fixed offline DPO as comparator
- uniform OPD as an ablation against adaptive OPD
- FinQA transfer after the FinChain result is clean

Cut for now:

- PPO implementation
- KTO/ORPO detours
- broad DPO sweeps
- 7B+ training before the 4B study works

## Model Choice

Default serious model:

- `Qwen/Qwen3-4B-Base`

Why:

- modern enough to discuss credibly
- small enough for one 48GB GPU with LoRA/QLoRA
- base model gives a clean post-training story
- Qwen family continuity with the existing 0.5B work

Fallback:

- `Qwen/Qwen2.5-3B-Base`

Use it if Qwen3 support, tokenizer behavior, or dependency churn slows down the experiment.

Canary:

- `Qwen/Qwen2.5-0.5B`

Use it for local and notebook debugging, not as the serious finance result.

Reference-only candidates:

- Phi-3.5 Mini Instruct
- Llama 3.2 3B Instruct

These can be evaluated for context, but they should not become the main train-from-base path unless they clearly dominate and the licensing/tooling story is acceptable.

## GPU Choice

Default:

- one 48GB GPU on RunPod or similar
- A40 is acceptable
- L40S or RTX 6000 Ada is preferred if the price/performance is close
- A6000 or L40 are viable

Why:

- enough memory for 3B/4B LoRA/QLoRA
- enough memory for rollout generation with moderate batch sizes
- simple enough that notebook iteration remains practical

Avoid as the default:

- full fine-tuning a 3B/4B model with AdamW
- 7B models before the 4B result is clean
- multi-GPU setup before the single-GPU bottleneck is measured

Cluster experiment:

Use 2x or 4x A100/H100 only after the single-GPU run is reproducible and the scaling question is explicit:

- Does larger K improve GRPO accuracy per dollar?
- Does rollout throughput dominate total time?
- Does multi-GPU let us run a cleaner KL sweep without waiting days?

## Study Flow

### Stage 0: Freeze The SFT Baseline

Goal:

Turn the existing SFT work into a finished artifact.

Outputs:

- SFT ablation table
- short writeup
- example predictions
- statement that 0.5B is now the canary path

What to learn:

- why more steps can hurt small models
- difference between parse score and exact-answer accuracy
- how to discuss overfit without overstating evidence

### Stage 1: FinChain Dataset And Verifier

Notebook:

- `notebooks/finchain_00_dataset_and_verifier.ipynb`

Build:

- loader
- prompt formatter
- answer parser
- chain executor or checker
- corrupt-answer tests
- topic/template summaries

Core checks:

- gold answers verify
- corrupted answers fail
- parse failures are separated from reasoning failures

### Stage 2: Model Bake-Off

Notebook:

- `notebooks/finchain_00_model_bakeoff.ipynb`

Run on 200-500 examples:

- Qwen3-4B-Base
- Qwen2.5-3B-Base
- optional reference instruct model

Pick based on:

- accuracy
- parseability
- tokens/sec
- memory footprint
- failure modes
- tooling friction

### Stage 3: FinChain SFT

Notebook:

- `notebooks/finchain_01_sft_lora.ipynb`

Train:

- one LoRA or QLoRA SFT baseline
- no broad sweep
- fixed eval split
- early stopping or short checkpoints

Report:

- train loss
- eval accuracy
- parseability
- examples before/after
- cost

### Stage 4: Rollouts, Buckets, Cost Ledger

Notebook:

- `notebooks/finchain_02_rollouts_and_buckets.ipynb`

For each prompt:

```text
sample K=4 completions
verify each completion
p_correct = correct / K

if p_correct >= 0.8:
    bucket = easy
    extra_samples = 0
    train_weight = 0.25
elif p_correct <= 0.2:
    bucket = hard
    extra_samples = 0
    train_weight = 0.5
else:
    bucket = ambiguous
    extra_samples = 12
    train_weight = 1.0
```

Report:

- bucket proportions
- rollout tokens
- verifier calls
- parse failures
- cost per prompt

### Stage 5: Rejection SFT And OPD

Notebook:

- `notebooks/finchain_03_rejection_sft_and_opd.ipynb`

Run:

- rejection SFT from verified-correct samples
- uniform OPD
- adaptive OPD

Compare:

- accuracy
- pass@K
- parseability
- KL/reference drift proxy
- cost
- examples where preference learning helped or hurt

### Stage 6: GRPO

Notebook:

- `notebooks/finchain_04_grpo.ipynb`

Run one controlled experiment:

- one K
- one reward function
- one KL coefficient
- one learning rate
- one budget

Watch:

- reward hacking
- answer-format collapse
- KL spikes
- parseability gains that do not improve true accuracy
- template/topic overfit

### Stage 7: Transfer And Writeup

Notebook:

- `notebooks/finchain_05_transfer_and_writeup.ipynb`

Evaluate:

- base
- SFT
- rejection SFT
- OPD
- GRPO

On:

- FinChain held-out split
- FinQA transfer subset

Final table:

```text
method | model | train examples | rollout tokens | GPU-hours | dollars | parseability | FinChain acc | FinQA acc | notes
```

## Study Guide Outline

The study guide should be implemented alongside notebooks, not after everything is done.

Chapters:

1. What post-training changes after pretraining
2. SFT: token-level imitation
3. DPO: pairwise preference learning with a reference policy
4. OPD: DPO-style learning on current-policy verified rollouts
5. PPO: why clipping and trust regions matter
6. GRPO: grouped rewards without a separate critic
7. RLHF versus RLVR
8. KL control and reward hacking
9. Verifier design for finance
10. Cost-aware experiment design
11. What the FinChain study taught

Each chapter should link to the relevant notebook and code path.

## Interview Narrative

Short version:

> I started with SFT to learn the training substrate, then moved to FinChain because it gave deterministic financial reasoning rewards. That let me compare rejection SFT, OPD, and GRPO under a fixed compute budget. The interesting part was not just final accuracy; it was how rollout difficulty, parseability, KL drift, and cost shaped which method was actually useful.

Longer version:

> The project deliberately separates method learning from domain realism. FinChain is the controlled verifier-rich environment where I can debug RLVR. FinQA is the transfer check that tells me whether the behavior survives messier filing excerpts. That structure keeps the project practical while still finance-specific.

## Practical Next Commands

Local canary:

```powershell
$env:WANDB_MODE = "offline"
.venv\Scripts\python.exe -m finpost.training.train --config experiments/local_tiny_gpt2.yaml --device cpu
```

First implementation slice:

```powershell
python -m pytest tests
```

Then build the FinChain loader, verifier, and notebook before spending GPU money.
