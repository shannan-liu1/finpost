# 02 - Implement DPO loss and reference parity

- **Status:** Implemented; TRL numerical reference comparison still optional
- **Ready for agent:** yes
- **Depends on:** 01-build-preference-pairs

## Parent

`.scratch/phase1-dpo-comparison/PRD.md`

## What to build

Implement the Direct Preference Optimization loss from scratch, including
chosen/rejected log-prob extraction, prompt masking, beta scaling, and frozen
reference-model handling. TRL may be used only as a numerical parity reference.

## Acceptance criteria

- [x] `src/finpost/training/dpo.py` exposes a small tested interface for
      computing per-example and mean DPO loss.
- [x] The implementation computes sequence log-probabilities only over response
      tokens, not prompt or padding tokens.
- [x] The reference model is frozen, uses `torch.no_grad()` where appropriate,
      and never receives gradients.
- [x] Swapping chosen and rejected responses changes the loss direction.
- [x] Beta changes the loss as expected and is serialized in configs.
- [ ] Local loss matches a reference calculation within `1e-5` on a fixed tiny
      batch.

## Verification

- `uv --cache-dir .uv-cache run python -m pytest tests/test_dpo.py tests/test_preference_data.py`

## Blocked by

Needs the preference example shape from issue 01.
