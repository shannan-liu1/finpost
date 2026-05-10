# 01 - Build preference pairs

- **Status:** Not Started
- **Ready for agent:** no
- **Depends on:** phase1-sft-trainer completion and first Qwen SFT checkpoint

## Goal

Sample completions from the SFT model and build chosen/rejected pairs for Direct Preference Optimization.

## Scope

**In scope:** prompt sampling, completion generation, programmatic final-answer grading, pair construction, pair metadata.

**Out of scope:** DPO loss implementation and DPO training.

## Acceptance criteria

- Preference-pair file contains prompt, chosen response, rejected response, source dataset id, source checkpoint id, grading result, and grading reason.
- Pair construction is deterministic for fixed seed and input completions.
- All-correct and all-incorrect prompt groups are counted separately and not silently discarded.
- No test-set prompts are used to build DPO training pairs.

## Notes / open questions

- Q-B decision required: drop all-correct/all-incorrect groups, resample, or create controlled corruptions.
