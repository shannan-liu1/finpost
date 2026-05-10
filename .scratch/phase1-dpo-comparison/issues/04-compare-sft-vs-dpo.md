# 04 - Compare SFT against SFT plus DPO

- **Status:** Not Started
- **Ready for agent:** no
- **Depends on:** 03-dpo-trainer-soft-launch

## Goal

Compare Base, SFT, and SFT + DPO using the same evaluation set and report whether DPO adds measurable value.

## Scope

**In scope:** final-answer accuracy, confidence intervals, response length, failure examples, summary table.

**Out of scope:** finance-domain evaluation and GRPO.

## Acceptance criteria

- Comparison uses the same held-out prompts for all models.
- Report includes Base, SFT, and SFT + DPO metrics.
- Bootstrap confidence intervals are included for each metric.
- Report includes at least 10 qualitative examples where SFT and DPO differ.
- Final conclusion states whether DPO improved, harmed, or was indistinguishable from SFT.

## Notes / open questions

- This issue is the decision gate before moving to finance-domain distillation.
