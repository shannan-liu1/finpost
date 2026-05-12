# Phase 1.5 Compute-Aware Post-Training on Small Reasoning Models

- **Status:** Not Started (drafted 2026-05-11)
- **Created:** 2026-05-11
- **Owner:** Shannan
- **Estimated time:** ~2–3 weeks after the Phase 1 SFT baseline lands
- **Depends on:** [`phase1-sft-trainer`](../phase1-sft-trainer/PRD.md), [`phase1-training-runbook`](../phase1-training-runbook/PRD.md), [`phase1-base-vs-sft-eval`](../phase1-base-vs-sft-eval/PRD.md)

## Goal

Deliver a post-training pipeline for small reasoning models that treats compute as a first-class experimental variable. The pipeline samples rollouts from the best Supervised Fine-Tuning (SFT) checkpoint, verifies each completion with the cheapest sufficient verifier, buckets prompts by difficulty, and then allocates **dense pairwise supervision (On-Policy Distillation, OPD) to high-disagreement prompts** while spending little or no compute on prompts where the model is already saturated or hopelessly wrong.

The headline claim under test:

> **Verifier-weighted, adaptive-compute On-Policy Distillation improves GSM8K and MATH accuracy per GPU-hour over uniform SFT, rejection SFT, and uniform OPD on `Qwen/Qwen2.5-0.5B`.**

This workstream slots in between the existing Phase 1 SFT trainer and the downstream Direct Preference Optimization (DPO) and Group Relative Policy Optimization (GRPO) workstreams. SFT produces the policy this pipeline starts from. DPO and GRPO consume the rollout, verifier, bucketing, and cost-ledger infrastructure built here.

### Why now

A previous estimate priced an SFT run at ~$100/A100-hour-class spend. The llm.c reproduction of GPT-2 124M on 10B FineWeb tokens — about 90 minutes on 8×A100, roughly $20, 29.9 HellaSwag — shows that the right framing is *clean systems on small models*, not a bigger model. The transferable lessons from llm.c that this workstream adopts:

1. Measure the whole training loop: tokens/sec, GPU-hours, dollars, verifier calls, wall-clock — alongside accuracy.
2. Separate CPU-bound (tokenization, rollout post-processing, verifier execution) from GPU-bound (training) stages. Precompute and cache everything that the GPU does not need to recompute.
3. Treat small models as laboratories. The contribution is not "made GPT-2/Qwen 0.5B good at math"; the contribution is "showed when dense preference-style supervision beats sparse signal under controlled compute".
4. Benchmark against simple baselines: same base model, same evaluation harness, only the post-training method varies.

## Scope

**In scope:**

- A 3,000-step combined SFT baseline on `Qwen/Qwen2.5-0.5B` with eval and checkpoint cadence at 500 steps, plus the two specialist arms (`gsm8k_only`, `math_only`).
- A rollout module that, given a checkpoint and a prompt set, generates K completions per prompt, caches them on disk, and never regenerates unless `(model_revision, sampling_params)` changes.
- A verifier module that grades each completion using the cheapest sufficient method (exact answer parser → symbolic / numeric equivalence; LLM-as-judge is explicitly disallowed for numerical correctness in this workstream).
- A difficulty-bucketing module that turns rollout grades into `(prompt_id, bucket, p_correct, n_samples)` records.
- An adaptive-sampling module that issues additional rollout work only on ambiguous prompts.
- A preference-pair builder that emits `(prompt, chosen, rejected, bucket, train_weight, source_checkpoint, source_revision)` records from cached rollouts.
- An On-Policy Distillation trainer (OPD) that consumes those preference pairs through a Direct-Preference-Optimization-style pairwise loss. Numerical parity against the offline DPO loss is required on uniform inputs.
- A cost ledger written at the run level and at the experiment level. Required fields: rollout tokens, verifier calls, training tokens, GPU-hours, USD cost, accuracy, pass@K, accuracy/$, accuracy/GPU-hour.
- A five-method comparison report at fixed dollar and GPU-hour budgets.
- A short writeup that names which method wins per axis.

**Out of scope (lives elsewhere):**

- The DPO comparison study itself. [`phase1-dpo-comparison`](../phase1-dpo-comparison/PRD.md) builds its own **fixed offline** preference dataset, runs DPO against it, and compares Base vs. SFT vs. SFT+DPO. This workstream and the DPO workstream share the verifier ladder and the DPO-style pairwise loss math (with a `1e-5` parity test enforced on uniform inputs) but each owns its own rollout cache and preference dataset. The split is intentional: offline-DPO vs. on-policy-OPD is itself a comparison axis.
- The GRPO online reinforcement-learning loop. Reward functions and grouped advantages live in [`phase1-grpo-research`](../phase1-grpo-research/PRD.md). This workstream supplies the rollout cache and verifier that GRPO will reuse.
- 10-K / filing domain data. Phase 2 ([`phase2-filing-distillation-dataset`](../phase2-filing-distillation-dataset/PRD.md)) consumes the compute-aware contracts produced here; it does not modify them.
- Multi-GPU and distributed training. The pipeline targets a single GPU (Colab T4 or a single rented A10 / L4 / A100). Multi-GPU is a follow-up if cost-per-experiment becomes the bottleneck.
- Larger base models. Qwen2.5-0.5B is the only Phase 1 substrate (decision Q-G in PLAN.md).
- LoRA / QLoRA. Phase 1 uses full fine-tuning; QLoRA arrives in Phase 2.

## Background — the cost philosophy

This workstream treats compute as three buckets and instruments each one.

### 1. Generation compute (the largest expected cost)

Rollouts dominate post-training cost when K is large. Cost is approximately:

```
rollout_cost ≈ num_prompts × samples_per_prompt × avg_output_tokens × tok_per_$ rate
```

The naive K=16 or K=32 everywhere is wasteful. The adaptive recipe used here:

```
Stage A: sample K=4 for all prompts             (cheap pass)
Stage B: assign bucket from p_correct           (free)
Stage C: sample K_extra only on ambiguous       (concentrated spend)
Stage D: train, weighted by bucket              (training spend)
```

Easy prompts (p_correct ≥ 0.8): no extra rollouts, low or zero training weight.
Ambiguous prompts (0.2 ≤ p_correct ≤ 0.8): extra K=12 → 28, full training weight.
Hard prompts (p_correct < 0.2): no extra rollouts; either deferred to a curriculum stage or weighted at 0.5.

### 2. Training compute

Kept cheap by sticking to the Phase 1 substrate: Qwen2.5-0.5B, short context (512 for GSM8K, 1024 only when needed), bf16, gradient accumulation, packed sequences, a fixed evaluation cadence, aggressive early stopping. The OPD trainer reuses the existing optimizer/scheduler/checkpoint stack from `phase1-sft-trainer`.

### 3. Verifier compute

Ladder of verifiers, cheapest first. For GSM8K and MATH, the exact-answer parser plus a symbolic/numeric equivalence check resolves the overwhelming majority of completions for free. No LLM-as-judge call is permitted on numerical correctness in this workstream.

## Stages (sliced for execution)

### Stage 0 — 3K-step SFT comparison surface

**Behaviour contract:** Three SFT runs on `Qwen/Qwen2.5-0.5B` with the existing Phase 1 trainer, each for 3,000 optimizer steps, with `eval_every_n_steps=500` and `checkpoint_every_n_steps=500`. Best checkpoint is selected by combined validation loss, not final step. Artifacts:

- `experiments/compute_aware/sft_gsm8k_only_3k.yaml`
- `experiments/compute_aware/sft_math_only_3k.yaml`
- `experiments/compute_aware/sft_combined_3k.yaml`

Per-checkpoint evaluation curves are produced (accuracy vs. step, accuracy vs. training tokens, accuracy vs. GPU-time).

**Why three:** the specialist arms are the cleanest comparison surface for the later compute-aware methods. They also confirm whether combined training transfers across GSM8K and MATH on this substrate.

**Eval mechanism:** per-checkpoint accuracy is produced by invoking the CLI built in [`phase1-base-vs-sft-eval`](../phase1-base-vs-sft-eval/PRD.md) (`python -m finpost.evals.eval_exact`) on each saved checkpoint, then aggregating the resulting `accuracy_summary.json` files into `eval_curve.json`. The same CLI is reused at Stage 5 to evaluate every post-training arm against the base model. Centralising the eval primitive ensures every method comparison is measured on the same instrument.

**Decision rule:**
- train loss falling and eval accuracy rising at step 3,000 → schedule one 5,000-step run on the winning arm.
- train loss falling and eval accuracy flat or dropping → stop, log as overfit or data-mix issue.
- specialist arm beats combined on its own test set but combined beats specialist averaged → keep combined as the policy substrate for Stage 1.

### Stage 1 — Cheap initial rollout and bucketing

**Behaviour contract:** From the chosen SFT checkpoint, sample K=4 completions for every prompt in a held-out training-prompt set (not test). Run the exact-answer verifier on every completion. Produce a parquet/jsonl file with `(prompt_id, sample_idx, completion, parsed_answer, is_correct, model_revision, sampling_params_hash)`. Aggregate into `(prompt_id, p_correct, bucket, n_samples)`.

**Cost target:** total rollout tokens at this stage `≤ 4 × num_prompts × p99_output_tokens`. Wall-clock `≤ 30 minutes on a single A100` (≤ 15 minutes on H100). Cost `≤ $1`.

### Stage 2 — Adaptive sampling on ambiguous prompts

**Behaviour contract:** For every prompt in the `ambiguous` bucket from Stage 1, generate `K_extra` additional completions (default `K_extra=12`, configurable up to 28). Append to the rollout cache; the cache lookup key `(model_revision, prompt_id, sampling_params_hash)` must be stable, idempotent, and resumable.

**Cost target:** total rollout tokens at this stage `≤ K_extra × num_ambiguous × p99_output_tokens`. Confirm in the cost ledger that `ambiguous` is a small fraction of `num_prompts` (typically 20–40%).

### Stage 3 — Preference-pair construction

**Behaviour contract:** From the rollout cache, build preference pairs for any prompt where at least one correct and one incorrect completion exists. Each pair carries `(prompt, chosen, rejected, bucket, train_weight, source_checkpoint, source_revision, sample_idx_chosen, sample_idx_rejected)`. The pair distribution per prompt is configurable (e.g. "all pairs", "best-vs-worst", "random one pair per prompt").

This stage emits the on-policy preference dataset that the OPD trainer consumes. It is **not** shared with [`phase1-dpo-comparison`](../phase1-dpo-comparison/PRD.md), which builds its own fixed offline preference dataset — keeping offline DPO and on-policy OPD as separate comparison axes. The two pipelines share the verifier and the pairwise-loss math, not the data. All-correct and all-incorrect prompts produce zero pairs and are bucketed for separate handling (Q-B in PLAN.md is resolved by construction in both pipelines independently).

### Stage 4 — On-Policy Distillation trainer (OPD)

**Behaviour contract:** Implement the OPD trainer with a Direct-Preference-Optimization-style pairwise loss:

```
loss = -log_sigmoid(beta * ((logp_pi(chosen|x) - logp_ref(chosen|x))
                            - (logp_pi(rejected|x) - logp_ref(rejected|x))))
```

The reference model is the SFT checkpoint, frozen with `requires_grad=False`. The policy is the SFT checkpoint, training. The training loop reuses the existing optimizer/scheduler/checkpoint stack from `phase1-sft-trainer`.

**Numerical parity gate:** on a fixed batch with uniform `train_weight=1.0`, the OPD per-example loss must match the offline DPO loss from [`phase1-dpo-comparison`](../phase1-dpo-comparison/PRD.md) within `1e-5`. This guarantees that any measured difference between OPD and DPO is attributable to the rollout policy / data distribution, not loss-function drift.

**Bucket weighting:** the per-example loss is multiplied by `train_weight`, which defaults to `{easy: 0.25, ambiguous: 1.0, hard: 0.5}` and is overridable per experiment.

### Stage 5 — Five-method comparison at fixed budgets

For the same SFT-best checkpoint, run:

- **A. uniform SFT.** The Stage 0 baseline. Cost baseline for accuracy/$ comparisons.
- **B. rejection SFT.** SFT trained only on completions the verifier accepts (chosen-only).
- **C. uniform OPD.** OPD trained on all preference pairs with `train_weight=1.0`.
- **D. verifier-weighted OPD.** OPD with bucket-derived `train_weight` (default schedule).
- **E. adaptive-compute OPD.** Verifier-weighted OPD plus Stage 2 adaptive rollouts. The headline arm.

Each method runs at two preset budgets — a **small budget** (≈30 minutes on a single A100, ~$1 at spot pricing) and a **medium budget** (≈2 hours on a single A100, ~$4) — so the cost-vs-accuracy curves are directly comparable. A single H100 is an acceptable substitute for either; same dollar envelope, ~2× faster wall-clock.

The combined ten-run comparison should fit comfortably under **$25 total spend**, matching the llm.c GPT-2 124M reproduction spirit ($20 on 8×A100 for 90 minutes). The arithmetic justifying this target:

- Qwen 0.5B post-training data is GSM8K + MATH train ≈ 15K examples, several orders of magnitude smaller than the 10B FineWeb tokens used by llm.c.
- 3K SFT optimizer steps at batch 32, seq 512 ≈ 50M training tokens — ~25–30 minutes on a single A100 at ~30K tokens/sec.
- Initial rollout (K=4, ~5K prompts, ~512 output tokens) ≈ 10M output tokens — ~30 minutes on A100 with batched inference.
- Adaptive rollout (K_extra=12 on ~30% ambiguous) ≈ 9M output tokens — another ~30 minutes.

If the actual cost ledger from Stage 1 reports numbers materially above this envelope (>$5 per method at the medium budget), pause and root-cause before scaling up.

### Stage 6 — Cost ledger and writeup

Per-run cost ledger (one row per method per budget):

| method | budget | base | rollout tokens | verifier calls | train tokens | GPU-hours | $ | GSM8K acc | MATH acc | pass@4 | accuracy/$ | accuracy/GPU-hour |

A one-page writeup names the winner per axis, lists the failure cases, and identifies which prompts the adaptive-compute method spent on that the uniform method missed.

## Deliverables

```
.scratch/phase1-compute-aware-post-training/
├── PRD.md                                       # this document
└── issues/
    ├── 01-3k-step-sft-comparison-surface.md     # Stage 0
    ├── 02-rollout-and-cache.md                  # Stage 1 + Stage 2 plumbing
    ├── 03-verifier-and-bucketing.md             # cheapest-first verifier ladder
    ├── 04-preference-pair-builder.md            # Stage 3
    ├── 05-opd-trainer-and-parity.md             # Stage 4
    ├── 06-five-method-comparison.md             # Stage 5
    └── 07-cost-ledger-and-writeup.md            # Stage 6

experiments/compute_aware/
├── sft_gsm8k_only_3k.yaml
├── sft_math_only_3k.yaml
├── sft_combined_3k.yaml
├── opd_uniform.yaml
├── opd_verifier_weighted.yaml
└── opd_adaptive.yaml

src/finpost/postraining/
├── rollout.py        # cached batched sampling
├── verifier.py       # exact -> symbolic -> numeric ladder
├── bucket.py         # difficulty assignment
├── preference.py     # preference-pair builder
├── opd.py            # OPD trainer (DPO-style pairwise loss + train_weight)
└── cost_ledger.py    # per-run accounting

scripts/
├── run_rollout.py    # CLI: checkpoint + prompts -> rollout cache
├── run_bucketing.py  # CLI: rollout cache -> bucket assignments
├── run_opd.py        # CLI: config -> OPD training run
└── build_cost_report.py

tests/
├── test_rollout_cache.py
├── test_verifier.py
├── test_bucket.py
├── test_preference.py
├── test_opd_loss.py     # parity vs DPO reference
└── test_cost_ledger.py

docs/primers/
└── compute-aware-post-training.md   # explainer linked from CONTEXT.md glossary
```

## Acceptance criteria

Each criterion is a command, file existence check, or observable numeric outcome.

1. `python -m finpost.training.train --config experiments/compute_aware/sft_combined_3k.yaml --max-steps 3000` runs end-to-end, writes checkpoints at steps 500, 1000, 1500, 2000, 2500, 3000, and emits a per-checkpoint evaluation curve under `results/<run_name>/eval_curve.json`.
2. The Stage 0 acceptance report names the best checkpoint by combined validation accuracy (not final step) for each of the three SFT arms and records it in `results/compute_aware/stage0_summary.md`.
3. `python scripts/run_rollout.py --checkpoint <best_sft> --prompts data/processed/train_prompts.jsonl --k 4` writes a deterministic rollout file. Re-running with the same `(checkpoint, prompts, sampling_params)` reuses the cache and performs zero new generations.
4. `python scripts/run_bucketing.py --rollouts <cache>` emits a bucket assignment file with the three buckets and reports their counts.
5. `python scripts/run_rollout.py --checkpoint <best_sft> --prompts <ambiguous_subset> --k 12 --append` appends to the cache without regenerating any earlier samples.
6. `pytest tests/test_opd_loss.py -k parity` passes: OPD per-example loss matches the DPO reference within `1e-5` on a fixed batch with `train_weight=1.0`.
7. Numerical correctness verifier never invokes an LLM-as-judge call. `grep -R "anthropic\|openai" src/finpost/postraining/verifier.py` returns no matches.
8. The cost ledger emitted by `python scripts/build_cost_report.py --run-glob 'results/compute_aware/*'` contains rows for at least methods A, B, C, D, E at the small budget, with all of: rollout tokens, verifier calls, training tokens, GPU-hours, USD cost, GSM8K accuracy, MATH accuracy, accuracy per dollar, accuracy per GPU-hour.
9. The headline writeup (`results/compute_aware/writeup.md`) explicitly states whether method E (adaptive-compute OPD) beat methods A–D on accuracy per GPU-hour at both budgets, and on absolute accuracy at the medium budget, with bootstrapped 95% confidence intervals.
10. Every spend-bearing run launched as part of Stages 0–5 carries a completed `.scratch/templates/cost-gate-checklist.md` under `results/<run_name>/cost_gate.md`. No spend-bearing run is launched without the checklist's "Owner decision: approve" line filled in.

## Cost philosophy — operating rules

These are durable rules that apply to every experiment in this workstream:

1. **Cache everything.** Prompts, generations, parsed answers, verifier results, tokenized datasets, evaluation outputs. Cache key includes `(model_revision, sampling_params_hash, verifier_version)`. Never regenerate unless the model changed.
2. **Adaptive sampling, not uniform K.** Spend extra rollout tokens only where uncertainty is high.
3. **Separate rollout from training.** Rollouts may run on cheaper or quantized inference. Training runs in bf16 full-precision. Document the rollout precision in the cost ledger so accuracy regressions from quantized rollout are detectable.
4. **Cheapest verifier first.** Exact parser → symbolic / numeric → small local model. LLM-as-judge is disallowed for numerical correctness; it is permitted only for the explanation-faithfulness check used downstream by Phase 2 filings.
5. **Cost as a first-class metric.** Every method comparison reports `accuracy / $` and `accuracy / GPU-hour` alongside accuracy. A method that wins on accuracy but loses on `accuracy / $` at the medium budget is described as such in the writeup and is not allowed to be the headline result without explicit justification.
6. **Default GPU is a single A100 or H100 spot instance.** Post-training Qwen 0.5B on ≈15K math examples has several orders of magnitude less data than the 10B-token pretraining runs that finish in 10–90 minutes on these GPUs; a slower or smaller GPU would only inflate cost per experiment. The cost-gate checklist captures the per-run choice between A100 and H100; T4 / Colab is permitted only for sanity-check runs under 5 minutes where queue time would otherwise dominate.

## Decisions resolved by this workstream

- **PLAN.md Q-B** ("how to handle DPO prompts with all-correct or all-incorrect samples?"). Resolved by construction in Stage 3: all-correct and all-incorrect prompts produce zero preference pairs and are bucketed as `easy` or `hard` for separate handling (down-weighted SFT for easy; curriculum or zero-weight for hard).

## Open questions

- **TODO: think about the `train_weight` schedule.** No strong prior yet. Default `{easy: 0.25, ambiguous: 1.0, hard: 0.5}` is a placeholder chosen to embody the qualitative claim "spend dense supervision where the model is uncertain". Alternatives worth considering before locking it in:
  - Continuous weight as a function of `p_correct` rather than a three-step bucket (e.g. `train_weight = 1 - |2 * p_correct - 1|`, a triangle peaking at `p_correct = 0.5`).
  - Asymmetric weighting that penalises hard prompts more than easy ones (e.g. `{easy: 0.25, ambiguous: 1.0, hard: 0.0}`), on the hypothesis that hard prompts inject noise rather than signal.
  - Bucket-count-balanced weighting that scales weights to equalise total contribution per bucket.

  Resolve before Stage 5 launches.
- Whether to default to A100 or H100. Both fit the dollar envelope; H100 halves wall-clock. Decide after the Stage 0 SFT run reports actual tokens/sec on each. Default A100 until then.
- Whether the rollout precision should be `bf16` or `int8` quantized. Decided after Stage 1 reports actual rollout wall-clock on the chosen GPU; default `bf16` until evidence justifies the change.

## Notes

- The branch `claude/compute-aware-post-training-CF2mL` carries this workstream's design and the eventual implementation. PRD-only changes are committed there; new code follows in subsequent commits as Stage issues land.
- The OPD trainer reuses the optimizer, scheduler, and checkpointing components from `phase1-sft-trainer`. The new code is the pairwise loss, the rollout/verifier/bucketing pipeline, and the cost ledger — not a new training framework.
- "GPRO" in casual notes is normalized to "GRPO" everywhere in this workstream's documentation.

## Amendment 2026-05-11 — Stage 5 priority order and stop-point discipline

Stage 5's original ordering of methods (A through E) was a logical / pedagogical ordering, not a priority-of-execution ordering. The five-method comparison stays in scope, but is now executed in priority-of-information order so that if cost or time forces an early stop, the methods that were already run form a self-coherent published result rather than a half-built comparison.

### Execution order

1. **A — uniform Supervised Fine-Tuning** (Stage 0 three-arm reruns)
   - The anchor. Every other method's claim is "method X beats Supervised Fine-Tuning by some delta," which requires the Supervised Fine-Tuning number.
   - Note: this is the only stage that produces the gsm8k-only and math-only specialist arms; those are needed by Stage 5 method comparisons that require all three arms.

2. **C — uniform On-Policy Distillation** (`train_weight = 1.0` everywhere)
   - The simpler On-Policy Distillation variant; runs before the experimental method per the "simpler baseline first" discipline.
   - Tests: does On-Policy Distillation beat Supervised Fine-Tuning at all, irrespective of weighting?

3. **D — verifier-weighted On-Policy Distillation** (`train_weight = 1.0 + alpha * uncertainty` or the bucketed equivalent)
   - The headline experimental method.
   - Tests, in combination with C: does the per-example weighting do the work, or was uniform On-Policy Distillation already capturing it?

4. **B — rejection Supervised Fine-Tuning** (Supervised Fine-Tuning trained only on completions the verifier accepts)
   - Alternative-explanation check: does the pairwise loss add anything beyond just training on correct rollouts?

5. **E — adaptive-compute On-Policy Distillation** (D plus Stage 2 adaptive rollouts on ambiguous prompts)
   - Tests a separate hypothesis — compute allocation across prompts — on top of the data-weighting hypothesis established by D versus C.

### Natural stop points

| Stop after | Claim available | Methods executed |
|---|---|---|
| 3 | "Verifier-weighted On-Policy Distillation beats uniform On-Policy Distillation, both beat Supervised Fine-Tuning" (or the null/refutation version). The core hypothesis is fully attributed. | A, C, D |
| 4 | The above, plus: "and the pairwise loss is doing something rejection Supervised Fine-Tuning cannot replicate." | A, C, D, B |
| 5 | The above, plus: "and adaptive compute allocation is more efficient than uniform allocation per prompt." Full story. | A, C, D, B, E |

Stopping after position 1 or 2 is suboptimal — position 2 alone (Supervised Fine-Tuning + uniform On-Policy Distillation, no weighted On-Policy Distillation) cannot answer the headline question. Stopping after position 3 is the *minimum publishable / writeup-worthy unit* of this workstream.

### Cost-gate operating rule

Before launching each method (positions 2 through 5), inspect the cost ledger accumulated to that point. If actual spend is materially above the budget envelope sketched in the original Stage 5 section (more than approximately 1.5× the projected per-method cost), pause and root-cause before continuing. The cost-gate checklist already required per spend-bearing run still applies; this amendment adds the per-position pause-and-review step on top of it.

### What does not change

- The five methods themselves and their behavior contracts.
- The matched-compute discipline within each method's two budget tiers (small / medium).
- The acceptance criteria in the existing Acceptance criteria section. (Criterion 9 is now interpreted as "if method E was run, it explicitly states whether E beat methods A–D"; if E was not run, the writeup states that and explains the stop-point.)
- The dependency on [`phase1-base-vs-sft-eval`](../phase1-base-vs-sft-eval/PRD.md) for the eval mechanism.
