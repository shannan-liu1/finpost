# 02 - Filing example schema and verifier

- **Status:** Not Started
- **Ready for agent:** no
- **Depends on:** 01-open-source-dataset-survey

## Goal

Define the canonical Phase 2 filing-excerpt example schema and the programmatic verifier contract.

## Scope

**In scope:** schema for filing excerpt, question, task type, cited line items, computation, final answer, verification result.

**Out of scope:** teacher prompt design and model training.

## Acceptance criteria

- Schema distinguishes `extraction` and `reasoning` examples.
- Verifier checks that cited line items appear in the filing excerpt.
- Verifier checks extraction answers appear in cited line items.
- Verifier can re-execute arithmetic for supported reasoning examples.
- Verifier returns structured rejection reasons.

## Notes / open questions

- The verifier is the core guardrail. Teacher and judge outputs are never accepted without it.
