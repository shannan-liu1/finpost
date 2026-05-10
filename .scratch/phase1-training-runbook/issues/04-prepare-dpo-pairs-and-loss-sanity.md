# 04 - Prepare DPO pairs and numerical sanity checks

- **Status:** Not Started
- **Ready for agent:** no
- **Depends on:** 03-run-phase1-sft-ablations, phase1-dpo-comparison

## Goal

Create preference pairs from the best real Qwen Supervised Fine-Tuning checkpoint and verify Direct Preference Optimization objective numerics before any Direct Preference Optimization training runs.

## Scope

**In scope:**
- completion sampling from the selected Qwen Supervised Fine-Tuning checkpoint,
- final-answer grading,
- chosen/rejected pair construction,
- pair provenance metadata,
- all-correct/all-incorrect prompt accounting,
- local DPO loss parity against a reference implementation,
- handoff into `.scratch/phase1-dpo-comparison/`.

**Out of scope:**
- full DPO ablation matrix,
- GRPO or any online reinforcement-learning loop,
- finance/filing data.

## Expected commands

Pair generation command once implemented:

```bash
python scripts/build_dpo_pairs.py \
  --checkpoint results/checkpoints/<sft-run>/best.pt \
  --output data/processed/phase1_dpo_pairs.jsonl \
  --samples-per-prompt 8 \
  --temperature 0.8 \
  --seed <seed>
```

DPO reference install and tests once implemented:

```bash
pip install -e ".[dpo-reference]"
pytest tests/test_dpo.py tests/test_preference_data.py -v
```

## Acceptance criteria

- Preference dataset is generated with provenance metadata.
- Source checkpoint is a completed Qwen Supervised Fine-Tuning checkpoint, not TinyGPT and not an untrained base model.
- Preference-pair file contains prompt, chosen response, rejected response, source dataset ID, source checkpoint ID, grading result, and grading reason.
- Generated pairs exclude test-set prompts.
- All-correct and all-incorrect prompt rates are recorded.
- Open Q-B policy is resolved or explicitly recorded before training on the generated pairs.
- Local DPO loss matches reference loss within `1e-5` on a fixed batch.
- Reference model is frozen and receives no gradients.
- Generated artifacts are linked into `.scratch/phase1-dpo-comparison/`.

## What this validates

Pair preparation validates that the DPO signal comes from real SFT model behavior on held-out training prompts. Loss parity validates that the local implementation matches the reference math before any optimizer step can amplify an error.

## Required proof artifacts

- Source Qwen SFT checkpoint path.
- Generation command and seed.
- Preference-pair file path.
- Grading summary.
- All-correct/all-incorrect rates.
- DPO parity pytest summary.
- Fixed-batch parity value.

## Stop conditions

- Stop if no real Qwen Supervised Fine-Tuning checkpoint exists.
- Stop if test-set prompts leak into preference-pair generation.
- Stop if local DPO loss does not match the reference within tolerance.
- Stop if Q-B is unresolved and the generated data contains all-correct or all-incorrect groups that would be silently dropped.
