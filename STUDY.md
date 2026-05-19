# finpost study guide: FinChain-first RLVR

Status: active study guide as of 2026-05-19.

This document is the human and agent learning map for `finpost`. The goal is not
to collect post-training acronyms. The goal is to build a small, falsifiable
research program that teaches the mechanics behind modern post-training roles:
SFT, DPO, OPD, GRPO, RLVR, KL control, verifier design, cost accounting, and
eventually multi-GPU sharding.

The short version:

> Use FinChain as a controlled wind tunnel for verifiable financial reasoning,
> then use FinQA/TAT-QA/FinanceBench-style tasks as transfer and realism checks.

That is a hypothesis, not a religion. FinChain is promising because it gives
cheap executable rewards. It is risky because template-generated symbolic tasks
can make models look more financially competent than they are. The study is
designed around that tension.

## 1. The motivating problem

Post-training is increasingly about turning a base model into a policy that does
something specific under constraints. The hard part is not just "make accuracy go
up." The hard part is deciding what signal is trustworthy enough to optimize.

In finance, this matters immediately. Consider a model answering:

> Revenue increased from 12.4B to 14.1B. What was the year-over-year growth rate?

A good answer is not merely fluent. It must:

1. select the correct numbers,
2. apply the correct formula,
3. compute `(14.1 - 12.4) / 12.4`,
4. format the final answer in a parseable way,
5. avoid inventing a business story that the data did not support.

This is exactly where many generic post-training examples are too soft. If a
language-model judge says "looks good," you still may not know whether the
arithmetic is true. If a static preference dataset says response A is better
than response B, you may still not know whether your current policy is failing
in the same way. If a leaderboard reports one aggregate accuracy number, you may
not know whether the model learned finance, learned a template, or learned a
parser trick.

The repo's north star is therefore:

> Build a verifier-driven post-training loop where every model update can be
> explained in terms of examples, rollouts, rewards, KL drift, and cost.

## 2. Honest verdict on FinChain

FinChain is the right next direction if the target is learning RLVR mechanics.
It is not automatically the right final benchmark for finance.

Why it is strong:

- It provides symbolic financial reasoning tasks with executable verification.
- It makes step-level or chain-level diagnostics possible, not just final-answer
  scoring.
- It is cheap enough to support rollouts, bucketing, rejection SFT, OPD, and
  GRPO without constantly spending on LLM judges.
- It creates a controlled environment where reward hacking and KL drift can be
  seen quickly.

Why it is weak:

- It may overrepresent tidy symbolic reasoning and underrepresent messy filings.
- It may reward template recognition rather than robust financial grounding.
- It may let the project become elegant but detached from real-world finance QA.
- It is new enough that treating it as "proven" would be false rigor.

The right language is:

> FinChain is the controlled RLVR laboratory. FinQA, TAT-QA, and FinanceBench are
> realism checks.

That framing is what makes the path less trodden. The conventional move would be
to chase a finance leaderboard, add a big model, and report final accuracy. The
more useful move is to build the reward laboratory first, stress it, and only
then ask whether the learned behavior transfers to real report excerpts.

## 3. Fundamental assumptions

The study rests on assumptions that must be tested, not waved through.

### Assumption A: verifiable rewards are more useful than judge rewards here

For financial arithmetic, deterministic verification should be the first reward
source. LLM-as-judge can help with semantic answerability or explanation quality,
but it should not be the primary truth source for numeric correctness.

Failure mode:

- The model learns to satisfy a judge's style preferences while still computing
  the wrong value.

Test:

- Corrupt numeric answers and chains; the verifier must reject them for the
  right reason.

### Assumption B: synthetic symbolic tasks can teach transferable structure

FinChain may teach formulas, chain discipline, and parseable final answers. But
transfer to FinQA is not guaranteed.

Failure mode:

- FinChain accuracy rises while FinQA accuracy stays flat or gets worse.

Test:

- Hold out templates/topics in FinChain, then run a small FinQA transfer eval
  after SFT, OPD, and GRPO.

### Assumption C: on-policy data is more informative than static pairs

DPO teaches a crucial pairwise preference objective, but fixed offline pairs can
be stale. OPD and GRPO are attractive because their training signal comes from
the current policy's own failures.

Failure mode:

- On-policy sampling becomes expensive noise if the verifier is weak or the model
  is too bad to generate useful variation.

Test:

- Track easy/ambiguous/hard buckets. Ambiguous prompts should be the main source
  of useful preference and RL signal.

### Assumption D: cost is part of the result

Accuracy without GPU-hours and rollout tokens is not enough. A method that gains
1 point at 10x the cost may be worse for this project than a method that gains
0.5 points cheaply and is easy to explain.

Failure mode:

- The project turns into a collection of expensive anecdotes.

Test:

- Every notebook writes or updates a cost ledger.

## 4. Conventional path versus the weird path

### The conventional path

1. Pick a popular benchmark.
2. Pick a model family.
3. Run SFT.
4. Run DPO or PPO because that is what the field talks about.
5. Report accuracy.

This is not useless. It teaches the outer shell of the workflow. But it often
skips the part interviewers care about when they probe: why this reward, why
this data distribution, why this update rule, why this compute budget, and what
failed?

### The weird but effective path

1. Start with a domain where the verifier is strong.
2. Treat the benchmark as a wind tunnel, not the destination.
3. Use small rollouts to map model uncertainty.
4. Spend extra samples only on ambiguous prompts.
5. Compare rejection SFT, OPD, and GRPO under the same rollout cache.
6. Track KL/reference drift and parseability alongside accuracy.
7. Only then test transfer on messier finance tasks.

This is less common because it is less glamorous. It does not start with the
largest model or the messiest dataset. It starts with measurement. That is the
advantage.

## 5. The opposite direction worth considering

The strongest critique of FinChain-first is that it may be too clean. If we took
one step in the opposite direction, we would do **evaluator-first finance**:

1. Do not train at first.
2. Build a brutal evaluation harness over FinQA, TAT-QA, and a small
   FinanceBench-style open-source subset.
3. Evaluate several base and instruct models.
4. Catalog failure modes by retrieval, formula selection, arithmetic, unit
   conversion, and final-answer formatting.
5. Only then choose a training dataset.

This would be slower emotionally because it delays the fun part. It may be more
scientifically honest if the central question is "what does finance reasoning
actually require?" rather than "how do I learn RLVR?"

My recommendation is a hybrid:

- Use FinChain to learn RLVR mechanics.
- Add a small evaluator-first realism gate before declaring any method win.

That means the project can say:

> I did not mistake a symbolic benchmark for finance. I used it as a controlled
> reward lab, then measured transfer.

## 6. Method ladder

### Base / few-shot

Purpose:

- Establish what the model already knows.

What it teaches:

- Prompt sensitivity, parseability, and whether the base model already has
  enough formula knowledge to make training worthwhile.

### SFT

Purpose:

- Teach format, domain vocabulary, and reasoning trace style.

Core idea:

- Minimize token-level cross entropy on gold responses.

Risk:

- SFT can teach answer shape without improving reasoning. Your earlier SFT
  ablation already showed the kind of signal to watch: parse score can stay
  stable while exact accuracy degrades.

### DPO

Purpose:

- Learn the pairwise preference objective and reference-policy framing.

Core idea:

- Increase the log-probability margin between chosen and rejected completions
  relative to a frozen reference model.

Risk:

- Fixed offline pairs can become stale. DPO is valuable for fundamentals, but it
  should not block OPD/GRPO on FinChain.

### Rejection SFT

Purpose:

- Test the cheapest verified self-training baseline.

Core idea:

- Sample completions, keep verified-correct outputs, train on them as SFT data.

Risk:

- It only learns from positives. It may miss information contained in near-misses.

### OPD

Purpose:

- Bridge DPO and RLVR.

Core idea:

- Sample from the current policy, verify completions, build chosen/rejected
  pairs, and apply a DPO-style loss with optional difficulty weights.

Why it is promising:

- It reuses DPO mechanics while making the data distribution on-policy.

### GRPO

Purpose:

- Run a direct grouped RLVR update.

Core idea:

- For each prompt, sample a group of completions, score each with the verifier,
  normalize rewards within the group, and update the policy with KL control.

Why it is promising:

- FinChain gives exactly the kind of grouped verified reward signal GRPO needs.

Risk:

- If the reward is too narrow, GRPO can optimize the parser instead of reasoning.

## 7. Notebook sequence

Existing notebooks to keep:

- `notebooks/sft_phase1_runpod_ablation_2000.ipynb`
- `notebooks/dpo_phase1_runpod.ipynb`

New RLVR notebooks:

1. `notebooks/finchain_00_dataset_and_verifier.ipynb`
2. `notebooks/finchain_00_model_bakeoff.ipynb`
3. `notebooks/finchain_01_sft_lora.ipynb`
4. `notebooks/finchain_02_rollouts_and_buckets.ipynb`
5. `notebooks/finchain_03_rejection_sft_and_opd.ipynb`
6. `notebooks/finchain_04_grpo.ipynb`
7. `notebooks/finchain_05_transfer_and_writeup.ipynb`
8. `notebooks/finchain_06_distributed_training_lab.ipynb`

Every notebook should have:

- a purpose cell,
- a hardware/cost pre-flight cell,
- a visible progress helper,
- a small dry run before the expensive run,
- a cost ledger update,
- representative examples,
- failure-mode notes,
- a final "what did we learn?" cell.

The notebooks are not just wrappers around scripts. They are the lab surface.
They should show enough intermediate state that a human can learn from each cell.

## 8. Function and infrastructure audit

Current SFT strengths:

- explicit masked cross entropy,
- packed examples,
- checkpointing with RNG state,
- cheap CPU canary path,
- visible wandb metrics,
- NaN guards.

Current SFT bottlenecks:

- document isolation currently uses a dense 4D attention mask,
- SFT DataLoader previously lacked configurable workers and pinned memory,
- no distributed launch path yet,
- no LoRA/QLoRA trainer path yet,
- no FlashAttention variable-length packed-attention path yet.

Current DPO strengths:

- chosen/rejected batches are concatenated into one policy forward and one
  reference forward,
- tokenized preference cache avoids repeated tokenizer work,
- DPO has DataLoader worker and pinned-memory knobs.

Current DPO bottlenecks:

- policy and reference both live on the same device, which is simple but memory
  expensive,
- reference logits are recomputed every step rather than precomputed,
- no distributed path,
- no adapter-only DPO path yet.

Immediate safe improvement already made:

- SFT now exposes DataLoader worker and pinned-memory knobs and uses non-blocking
  tensor transfers when moving batches to the training device.

Deferred improvements:

- precompute DPO reference log-probabilities,
- LoRA/QLoRA support for 3B/4B FinChain SFT and DPO/OPD,
- Accelerate/FSDP launch path,
- FlashAttention 2 variable-length packing,
- vLLM or another continuous-batching path for high-throughput rollouts.

## 9. Multi-GPU learning path

Do not start by trying to make the whole project distributed.

The learning ladder should be:

1. single-GPU LoRA/QLoRA on Qwen3-4B,
2. distributed toy notebook that proves DDP/FSDP concepts,
3. DDP for replicated adapter training if memory already fits,
4. FSDP or DeepSpeed ZeRO when memory does not fit or optimizer state dominates,
5. multi-GPU rollout generation when sampling throughput becomes the bottleneck.

Interview-relevant terms to learn:

- rank,
- world size,
- process group,
- all-reduce,
- distributed sampler,
- gradient synchronization,
- parameter sharding,
- optimizer-state sharding,
- activation checkpointing,
- NCCL,
- sharded checkpoints.

The main conceptual distinction:

- **DDP** replicates the model on every GPU and synchronizes gradients.
- **FSDP/ZeRO** shard model states across GPUs so each GPU stores less.
- **Tensor/pipeline parallelism** split the model's computation itself and are
  usually unnecessary for this project until models get much larger.

## 10. Platform stance

RunPod is still useful, but it should not be the only mental model.

Use RunPod for:

- cheap single-GPU notebooks,
- A40/L40S/RTX 6000 Ada LoRA experiments,
- quick Jupyter iteration,
- small cluster tests if availability is acceptable.

Use Lambda when:

- you want a more conventional VM/cloud experience,
- you can tolerate higher cost or capacity friction,
- you need stable multi-GPU boxes or 1-click clusters.

Use Modal when:

- the workload is job-shaped rather than notebook-shaped,
- you want repeatable batch rollouts,
- startup/container ergonomics matter more than a persistent interactive pod.

Use Vast.ai when:

- price is the dominant constraint,
- you are willing to filter hard for host reliability,
- you can tolerate more platform variability.

The practical rule:

> Notebook learning on RunPod; reproducible batch rollouts on Modal or a stable
> VM; serious multi-GPU only when the single-GPU result has earned the spend.

## 11. Evidence and sources

Primary/current sources used for the current direction:

- FinChain arXiv: https://arxiv.org/abs/2506.02515
- FinQA site: https://finqasite.github.io/
- TAT-QA repository: https://github.com/NExTplusplus/TAT-QA
- FinanceBench repository: https://github.com/patronus-ai/financebench
- PyTorch DistributedDataParallel docs: https://docs.pytorch.org/docs/main/generated/torch.nn.parallel.DistributedDataParallel.html
- Hugging Face Accelerate FSDP guide: https://huggingface.co/docs/accelerate/main/en/usage_guides/fsdp
- DeepSpeed ZeRO tutorial: https://www.deepspeed.ai/tutorials/zero/
- RunPod pods and instant clusters docs: https://docs.runpod.io/
- Lambda GPU cloud pricing: https://lambda.ai/service/gpu-cloud/pricing
- Modal GPU docs: https://modal.com/docs/reference/modal.gpu
- Vast.ai docs/site: https://vast.ai/

## 12. The interview story

The strongest version:

> I built a post-training lab around financial symbolic reasoning because finance
> gives objective correctness signals. I started with SFT and DPO to learn the
> fundamentals, then moved to FinChain because executable chains let me build an
> RLVR loop with deterministic rewards. The core comparison was not just
> accuracy: I compared SFT, rejection SFT, OPD, and GRPO by accuracy, parseability,
> KL drift, rollout tokens, GPU-hours, and transfer to messier finance tasks. I
> also learned where the approach can fail: template overfit, reward hacking, and
> synthetic-to-real transfer.

The sentence that keeps it honest:

> FinChain is my wind tunnel, not my destination.
