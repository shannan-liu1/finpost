# 04 - Assemble SFT and DPO data plan

- **Status:** Not Started
- **Ready for agent:** no
- **Depends on:** 03-teacher-generation-and-judge-rubric

## Goal

Define how accepted filing examples become SFT data and DPO preference pairs.

## Scope

**In scope:** SFT target construction, rejected-response construction, pair metadata, dataset splits, leakage policy.

**Out of scope:** training Phase 2 models.

## Acceptance criteria

- SFT example format matches the Phase 2 final example shape in `PLAN.md`.
- DPO pair construction includes at least two rejected-response sources: sampled weak model responses and controlled corruptions.
- Dataset split is held out by company or filing, not by random example only.
- Every generated example records teacher model, verifier version, judge rubric version, and source filing metadata.

## Notes / open questions

- This issue gates Phase 2 training. Do not train on filing examples until this plan is explicit.
