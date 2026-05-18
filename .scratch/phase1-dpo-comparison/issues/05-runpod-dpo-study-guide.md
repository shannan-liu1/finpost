# 05 - Create RunPod DPO operator guide and notebook

- **Status:** In Progress
- **Ready for agent:** yes
- **Depends on:** 01-build-preference-pairs, 03-dpo-trainer-soft-launch

## Parent

`.scratch/phase1-dpo-comparison/PRD.md`

## What to build

Create the operator-facing surface for the DPO study: a RunPod notebook that
executes pair generation, canary training, full DPO training, conversion, eval,
and packaging in order, plus a maintained HTML guide explaining the study,
GPU requirements, cost expectations, and stop/go gates.

## Acceptance criteria

- [ ] `notebooks/dpo_phase1_runpod.ipynb` runs cell-by-cell from a fresh
      RunPod checkout after `docs/runbooks/runpod-bootstrap.md`.
- [ ] `docs/dpo-study.html` names the exact run sequence and artifacts to save.
- [ ] GPU guidance distinguishes 48 GB A40 / RTX A6000 / RTX 6000 Ada from
      older 24 GB Quadro RTX 6000 cards.
- [ ] The notebook has a DPO canary cell that must pass before full training.
- [ ] The notebook packages DPO pair data, train logs, eval outputs, and the
      cost ledger for download or later analysis.

## Verification

- Notebook JSON parses.
- `git diff --check -- notebooks/dpo_phase1_runpod.ipynb docs/dpo-study.html`
- RunPod smoke path completes through the canary cell before full DPO.

## Blocked by

The static HTML guide can exist immediately. The executable notebook needs the
DPO CLI and pair builder from issues 01 and 03.

## Progress notes

- 2026-05-18: Added a draft RunPod operator notebook at
  `notebooks/dpo_phase1_runpod.ipynb`. It is intentionally fail-fast until the
  DPO pair builder and trainer entry points exist.
