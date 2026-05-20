# Plan: FinChain-First RLVR Study

Status: Active direction as of 2026-05-19.

## North Star

Build a notebook-first, compute-aware post-training study for financial symbolic reasoning:

> Starting from a small-model SFT foundation, use FinChain's executable financial reasoning chains to compare SFT, rejection SFT, OPD, and GRPO under fixed rollout and training budgets, then test transfer on FinQA.

The point is not to train the largest possible model. The point is to learn the stack well enough to explain every artifact: data, verifier, rollout cache, preference construction, KL control, failure modes, and cost.

## Current Read

- SFT is completed enough to become a documented baseline and canary path.
- DPO scaffolding is useful and should stay, but a full finance DPO arc is not the highest-leverage next move.
- FinQA is still interesting, but it is not the easiest primary benchmark for RLVR because verification is mostly final-answer numeric matching over messy real excerpts.
- FinChain is the better next substrate because it makes the reward signal cheaper, denser, and more falsifiable.

## Hard Decisions

### Q-I: Primary Finance Benchmark

**Decision:** Use FinChain as the primary training and evaluation substrate for the next phase.

**Why:** FinChain gives executable symbolic chains and deterministic verification. That is exactly what OPD and GRPO need. FinQA stays as a transfer benchmark.

### Q-J: Method Priority

**Decision:** Do not run every method as a peer. Use a ladder:

1. Base / few-shot
2. SFT
3. Rejection SFT
4. OPD
5. GRPO
6. DPO as a fundamentals/comparator artifact only

**Why:** SFT and DPO teach the mechanics. OPD and GRPO are the highest-signal RLVR-shaped methods for this domain. A giant method zoo will dilute learning and make interviews less crisp.

### Q-K: Model

**Decision:** Default to `Qwen/Qwen2.5-1.5B` for the next FinChain loop.

**Scale-up candidate:** `Qwen/Qwen3-4B-Base` after the 1.5B loop is interpretable.

**Canary:** keep `Qwen/Qwen2.5-0.5B` for local smoke tests and cheap notebook iteration.

### Q-L: Hardware

**Decision:** Start on one 48GB GPU. A40 is acceptable; L40S, RTX 6000 Ada, L40, or A6000 are also viable. Use LoRA/QLoRA for all 3B/4B training.

**Cluster rule:** only move to 2x/4x A100 or H100 after the single-GPU run is reproducible and the reason for scale is specific, such as rollout throughput, larger K, or faster GRPO iteration.

## Phase 0: Close The SFT Foundation

Goal: freeze the current SFT work as a baseline artifact.

Deliverables:

- Result table for the completed 0.5B SFT runs
- Short writeup explaining the ablation insight: combined learner performed best; beyond roughly 500 steps, parse score stayed stable while exact accuracy degraded, suggesting overfit or format/reasoning drift at this scale
- Clear statement that 0.5B is now infrastructure/canary, not the serious finance model
- CPU/local canary command remains maintained

Exit gate:

- A new reader can understand what was learned from SFT without rerunning the whole notebook.

## Phase 1: FinChain Data And Verifier Harness

Goal: make FinChain loadable, inspectable, and verifiable inside notebooks and scripts.

Deliverables:

- `src/finpost/data/finchain.py`
- `src/finpost/evals/finchain_metrics.py`
- `src/finpost/posttraining/verifier.py` extensions for FinChain answer and chain checks
- Notebook: `notebooks/finchain_00_dataset_and_verifier.ipynb`
- CLI: `scripts/run_finchain_eval.py`

Acceptance checks:

- Load a small FinChain split locally.
- Render examples with prompt, chain, final answer, and topic metadata.
- Verify gold answers at or near 100%.
- Verify intentionally corrupted answers fail for the expected reason.

## Phase 2: Model Bake-Off

Goal: choose a serious model empirically before training.

Default candidates:

- `Qwen/Qwen2.5-1.5B`
- `Qwen/Qwen3-4B-Base`
- Optional reference: `Phi-3.5-mini-instruct` or `Llama-3.2-3B-Instruct` for inference-only comparison

Metrics:

- zero-shot and few-shot FinChain accuracy
- parseability
- average output tokens
- inference tokens per second
- failure modes by template/topic
- memory footprint on the chosen GPU

Exit gate:

- Pick one main model and one canary model. Do not proceed with a model zoo.

## Phase 3: FinChain SFT

Goal: train one finance SFT baseline that teaches format, formula use, and chain style.

Training posture:

- LoRA or QLoRA
- no broad sweep
- short context first
- fixed held-out eval
- save adapter, metrics, config, and notebook outputs

Notebook:

- `notebooks/finchain_01_sft_lora.ipynb`

Exit gate:

- SFT improves parseability and/or accuracy over base without obvious overfit.
- Loss, accuracy, parseability, and examples tell the same story.

## Phase 4: Rollout Cache And Cost Ledger

Goal: build the reusable substrate for OPD and GRPO.

Deliverables:

- `src/finpost/posttraining/rollout_schema.py`
- `src/finpost/posttraining/rollout_cache.py`
- `src/finpost/posttraining/sampler.py`
- `src/finpost/posttraining/bucket.py`
- `src/finpost/posttraining/cost_ledger.py`
- Notebook: `notebooks/finchain_02_rollouts_and_buckets.ipynb`

Default rollout policy:

- sample K=4 completions per prompt
- verify each completion
- compute `p_correct`
- bucket prompts as easy, ambiguous, or hard
- draw extra samples only for ambiguous prompts

Exit gate:

- Rollouts are reproducible by model revision, prompt revision, and sampling hash.
- Cost ledger reports rollout tokens, verifier calls, GPU-hours, estimated dollars, and parseability.

## Phase 5: Rejection SFT And OPD

Goal: compare the simplest verified self-training baseline against preference learning over on-policy samples.

Deliverables:

- `src/finpost/posttraining/rejection_sft.py`
- `src/finpost/posttraining/preference.py`
- `src/finpost/posttraining/opd.py`
- Notebook: `notebooks/finchain_03_rejection_sft_and_opd.ipynb`

Comparisons:

- SFT
- rejection SFT from verified-correct rollouts
- uniform OPD
- adaptive OPD with higher weight and extra sampling for ambiguous prompts

Exit gate:

- OPD result is reported with accuracy, parseability, KL/reference drift proxy, and cost.

## Phase 6: GRPO

Goal: run one controlled RLVR experiment using grouped verified rewards.

Deliverables:

- `src/finpost/posttraining/grpo.py`
- Notebook: `notebooks/finchain_04_grpo.ipynb`

Default constraints:

- one model
- one K value
- one reward function
- one KL coefficient schedule
- one training budget
- no grid until the first run is interpretable

Exit gate:

- GRPO is compared against SFT, rejection SFT, and OPD under the same evaluation harness and cost ledger.
- The writeup names whether GRPO improved reasoning, mostly improved format, or reward-hacked the verifier.

## Phase 7: Transfer And Interview Artifact

Goal: prove the result is not just benchmark overfitting and package the learning.

Deliverables:

- FinQA transfer eval notebook
- final comparison notebook
- study guide covering SFT, DPO, OPD, PPO, GRPO, RLHF, RLVR, KL control, verifier design, and reward hacking
- distributed-training guide covering DDP, FSDP, DeepSpeed ZeRO, sharded checkpoints, platform choices, and when rollout parallelism beats full distributed training
- README result table

Transfer checks:

- FinQA final-answer accuracy
- parseability
- topic/template failure analysis
- examples where FinChain helped
- examples where FinChain overfit or failed to transfer

## Hardware Policy

### Minimum Effective Dose

Use a single 48GB GPU:

- A40: acceptable and familiar
- L40S or RTX 6000 Ada: often better if price is close
- A6000/L40: viable alternatives

Use LoRA/QLoRA for 3B/4B models. Keep full fine-tuning out of the main path unless a smaller model or a very constrained context makes it cheap.

### When To Use A Cluster

Use 2x/4x A100 or H100 only when:

- single-GPU results are reproducible
- the bottleneck is measured
- the cluster experiment answers a new question, such as "does larger K improve GRPO per dollar?" or "does rollout throughput dominate training cost?"

Do not use a cluster to compensate for unclear evaluation or uncontrolled experiments.

## Cut List

Cut for now:

- More 0.5B scale-up experiments beyond canaries
- Full finance DPO arc before OPD/GRPO
- 7B training before a clean 3B/4B result
- PPO implementation before GRPO unless the user specifically wants theory-first implementation practice
- LLM-as-judge as primary reward
- Teacher-generated SEC filing data before the FinChain verifier loop works

Keep:

- notebook-first execution
- cheap local canaries
- FinChain verifier and rollout cache
- cost ledger
- SFT baseline
- rejection SFT
- adaptive OPD
- one controlled GRPO run
- FinQA transfer eval

## Active Workstreams

- `.scratch/finchain-rlvr-posttraining/PRD.md` - active PRD for the FinChain RLVR pivot
- `.scratch/phase1-dpo-comparison/PRD.md` - keep as DPO fundamentals/comparator work
- `.scratch/phase1-compute-aware-post-training/PRD.md` - reuse concepts for rollout cache, bucketing, and cost ledger
- `.scratch/phase1-grpo-research/PRD.md` - superseded in direction by the FinChain GRPO slice, but still useful background
- `STUDY.md` / `STUDY.html` - human-readable learning map and honest FinChain critique
- `docs/distributed-training-and-platforms.md` - multi-GPU and platform learning map
