# FinChain-First RLVR Post-Training Study

- **Status:** In Progress
- **Created:** 2026-05-19
- **Owner:** shann
- **Estimated time:** 2-4 focused weeks for the first complete study loop
- **Depends on:** phase1-sft-trainer, phase1-base-vs-sft-eval, phase1-compute-aware-post-training

## Goal

Build the active FinChain-first study flow for learning and demonstrating post-training skill: dataset/eval harness, finance SFT baseline, rollout cache, rejection SFT, OPD, one GRPO run, cost ledger, and FinQA transfer check.

This workstream is optimized for interview-quality learning artifacts: notebooks, result tables, failure examples, and a study guide that connects theory to code.

## Scope

**In scope:**

- FinChain loader and prompt formatting
- FinChain verifier for final answers and symbolic chains where available
- model bake-off over a small subset
- LoRA/QLoRA SFT on one selected 3B/4B model
- rollout cache keyed by model revision, prompt revision, and sampling parameters
- difficulty bucketing by verified rollout success rate
- rejection SFT from verified-correct generations
- OPD pair construction and adaptive weights
- one controlled GRPO run with KL tracking
- cost ledger for rollout tokens, verifier calls, GPU-hours, dollars, parseability, and accuracy
- FinQA transfer evaluation after the FinChain loop is complete
- notebooks that preserve the lab-like execution flow

**Out of scope:**

- full fine-tuning 3B/4B models
- 7B+ models before the 4B workflow is clean
- PPO implementation before GRPO
- broad method/model sweeps
- LLM-as-judge as primary reward
- teacher-generated SEC filing dataset

## Deliverables

Code:

- `src/finpost/data/finchain.py`
- `src/finpost/evals/finchain_metrics.py`
- `src/finpost/posttraining/rollout_schema.py`
- `src/finpost/posttraining/rollout_cache.py`
- `src/finpost/posttraining/sampler.py`
- `src/finpost/posttraining/verifier.py`
- `src/finpost/posttraining/bucket.py`
- `src/finpost/posttraining/preference.py`
- `src/finpost/posttraining/rejection_sft.py`
- `src/finpost/posttraining/opd.py`
- `src/finpost/posttraining/grpo.py`
- `src/finpost/posttraining/cost_ledger.py`

Scripts:

- `scripts/run_finchain_eval.py`
- `scripts/run_finchain_rollouts.py`
- `scripts/build_rejection_sft.py`
- `scripts/build_opd_pairs.py`
- `scripts/build_cost_report.py`

Configs:

- `experiments/finchain/model_bakeoff.yaml`
- `experiments/finchain/qwen3_4b_lora_sft.yaml`
- `experiments/finchain/rejection_sft.yaml`
- `experiments/finchain/opd_uniform.yaml`
- `experiments/finchain/opd_adaptive.yaml`
- `experiments/finchain/grpo.yaml`
- `experiments/finchain/finqa_transfer.yaml`

Notebooks:

- `notebooks/finchain_00_dataset_and_verifier.ipynb`
- `notebooks/finchain_00_model_bakeoff.ipynb`
- `notebooks/finchain_01_sft_lora.ipynb`
- `notebooks/finchain_02_rollouts_and_buckets.ipynb`
- `notebooks/finchain_03_rejection_sft_and_opd.ipynb`
- `notebooks/finchain_04_grpo.ipynb`
- `notebooks/finchain_05_transfer_and_writeup.ipynb`
- `notebooks/finchain_06_distributed_training_lab.ipynb`

Docs:

- `docs/runbooks/finchain-rlvr-study-flow.md`
- `docs/adr/0002-finchain-first-rlvr-roadmap.md`
- `STUDY.md`
- `STUDY.html`
- `docs/finchain-rlvr-professor-study.html`
- `docs/distributed-training-and-platforms.md`
- `docs/distributed-training-and-platforms.html`
- final study guide section or standalone doc after results exist

## Acceptance Criteria

- `python -m pytest tests` passes before GPU work starts.
- A local FinChain loader smoke test can load examples and render prompt/answer/chain fields.
- Gold FinChain examples verify at or near 100% on the selected local subset.
- Corrupted final answers fail the verifier with a useful reason code.
- Model bake-off produces a CSV or JSON table with accuracy, parseability, average output tokens, and runtime notes.
- SFT baseline stores config, adapter/checkpoint metadata, metrics, and representative generations.
- Rollout cache can be rerun without regenerating existing samples when the model and sampling hash match.
- Cost ledger reports rollout tokens, verifier calls, training tokens, GPU-hours, estimated dollars, and accuracy per cost unit.
- OPD uses current-policy rollouts and records chosen/rejected pair counts by difficulty bucket.
- GRPO records group rewards, KL/reference drift metric, parseability, and failure examples.
- Final comparison includes at least SFT, rejection SFT, OPD, and GRPO on FinChain, plus a FinQA transfer check.
- Distributed-training lab can explain rank/world size, DDP versus FSDP/ZeRO, and why rollout parallelism may be the first useful multi-GPU scaling move.

## Method Priority

Most promising without skipping fundamentals:

1. SFT: required anchor and format teacher.
2. Rejection SFT: cheapest verified self-training baseline.
3. Adaptive OPD: highest-leverage bridge from DPO mechanics to on-policy verifier learning.
4. GRPO: headline RLVR method.

DPO remains worth keeping because it teaches a core post-training primitive, but it should not block the FinChain RLVR path.

## Hardware Recommendation

Default:

- one 48GB GPU on RunPod or similar
- use LoRA/QLoRA
- start with `Qwen/Qwen3-4B-Base`
- fall back to `Qwen/Qwen2.5-3B-Base` if tooling friction is high

Cluster only after:

- single-GPU study is reproducible
- the bottleneck is measured
- the scaling question is explicit

## Risks

- **Template overfit:** FinChain may reward symbolic pattern matching. Mitigate with held-out templates/topics and FinQA transfer.
- **Reward hacking:** model may learn parseable shortcuts. Mitigate with chain checks, corrupted-answer tests, and failure examples.
- **Method sprawl:** too many methods dilute the learning artifact. Mitigate by keeping DPO optional and running only one GRPO configuration first.
- **GPU spend drift:** notebook convenience can hide cost. Mitigate with a cost ledger from the rollout stage onward.

## Notes / Open Questions

- Decide whether the first FinChain split should hold out templates, topics, or both.
- Decide whether GRPO reward should be binary correctness first or shaped with parseability/chain-validity terms.
- Decide whether FinQA transfer should be a small curated subset first or the full validation split.
