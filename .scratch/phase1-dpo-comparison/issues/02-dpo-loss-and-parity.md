# 02 - DPO loss and reference parity

- **Status:** Not Started
- **Ready for agent:** no
- **Depends on:** 01-build-preference-pairs

## Goal

Implement the Direct Preference Optimization loss from scratch and verify it against a reference implementation.

## Scope

**In scope:** chosen/rejected log-prob extraction, frozen reference model handling, beta parameter, numerical parity test.

**Out of scope:** full DPO trainer loop.

## Acceptance criteria

- `pytest tests/test_dpo.py -v` passes.
- Local DPO loss matches reference loss within `1e-5` on a fixed batch.
- Test covers beta sensitivity and confirms chosen/rejected swap changes the loss direction.
- Reference model is frozen and never receives gradients.

## Notes / open questions

- Use TRL as reference only; do not use TRL trainer as the implementation.
