# OPD Study Guide

Status: pair-construction backend implemented; weighted trainer integration
still pending.

## What OPD Is Doing

On-Policy Distillation is the practical bridge from offline DPO to RLVR:

1. Sample multiple completions from the current policy.
2. Score each completion with the deterministic verifier.
3. Build chosen/rejected pairs from the same prompt group.
4. Train with a DPO-style pairwise loss.
5. Repeat after the policy changes if the study needs another on-policy round.

The key distinction from ordinary DPO is the data distribution. DPO can train
on fixed offline pairs. OPD pairs are produced by the policy being improved, so
they expose the model's current mistakes and near-misses.

## Backend Contract

The implementation lives in `src/finpost/posttraining/opd.py`.

- `OPDRollout`: one current-policy completion plus verifier reward.
- `build_opd_pairs`: groups rollouts by prompt and emits best-vs-worst pairs.
- `OPDPair.to_dpo_preference_example`: converts OPD pairs into the existing
  `DPOPreferenceExample` shape.

This deliberately reuses the DPO trainer. OPD should not fork the pairwise loss
or tokenizer/collator stack until there is evidence that the existing path is a
bottleneck.

## Adaptive Weighting

The first adaptive weighting rule matches the FinChain study flow:

```text
success_rate >= 0.8 -> easy       -> weight 0.25
success_rate <= 0.2 -> hard       -> weight 0.50
otherwise           -> ambiguous  -> weight 1.00
```

The reasoning is pragmatic. Easy prompts already work, so they should not
dominate updates. Completely hard prompts may be unlearnable at the current
model size or rollout budget. Ambiguous prompts are where the verifier shows
the policy can sometimes solve the task, so pairwise pressure is most useful.

## Efficiency Reasoning

OPD's cost center is rollout generation and scoring, not pair construction.
The backend therefore uses one pass over the rollout list, groups by prompt id,
and emits one best-vs-worst pair per prompt by default. That keeps pair count
bounded and avoids over-weighting prompts with many sampled completions.

Later upgrades can add all-positive-vs-all-negative pair expansion, but only
after measuring whether one pair per prompt is underusing the rollout cache.

## FinChain Starting Point

Start with binary final-answer verifier rewards:

```text
reward = 1.0 if parsed final answer is correct else 0.0
```

Report pair counts by bucket before training. If most prompts are all-correct
or all-wrong, OPD has little signal and the right next move is more sampling,
a stronger SFT checkpoint, or a different prompt split rather than forcing a
pairwise update.
