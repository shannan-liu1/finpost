# 02 - Implement DPO loss and reference parity

- **Status:** Not Started
- **Ready for agent:** yes
- **Depends on:** 01-build-preference-pairs

## Parent

`.scratch/phase1-dpo-comparison/PRD.md`

## What to build

Implement the Direct Preference Optimization loss from scratch, including
chosen/rejected log-prob extraction, prompt masking, beta scaling, and frozen
reference-model handling. TRL may be used only as a numerical parity reference.

## Acceptance criteria

- [ ] `src/finpost/training/dpo.py` exposes a small tested interface for
      computing per-example and mean DPO loss.
- [ ] The implementation computes sequence log-probabilities only over response
      tokens, not prompt or padding tokens.
- [ ] The reference model is frozen, uses `torch.no_grad()` where appropriate,
      and never receives gradients.
- [ ] Swapping chosen and rejected responses changes the loss direction.
- [ ] Beta changes the loss as expected and is serialized in configs.
- [ ] Local loss matches a reference calculation within `1e-5` on a fixed tiny
      batch.

## Verification

- `.\.venv\Scripts\python.exe -m pytest tests/test_dpo.py`
- `.\.venv\Scripts\python.exe -m pytest tests/test_loss_dtype.py`

## Blocked by

Needs the preference example shape from issue 01.
