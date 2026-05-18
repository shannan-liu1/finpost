# 04 - Compare SFT against SFT plus DPO

- **Status:** Not Started
- **Ready for agent:** yes
- **Depends on:** 03-dpo-trainer-soft-launch

## Parent

`.scratch/phase1-dpo-comparison/PRD.md`

## What to build

Run the final Phase 1 comparison between Base, SFT-best, and SFT+DPO on the
same exact-answer evaluation surface, then write the decision report that says
whether DPO adds value over SFT.

## Acceptance criteria

- [ ] Evaluation uses the same held-out prompts, seed, and sample counts for
      all compared checkpoints.
- [ ] Report includes Base, SFT-best, and SFT+DPO final-answer accuracy for
      GSM8K and MATH.
- [ ] Report includes bootstrap confidence intervals, response-length
      statistics, parse-success rates, and cost ledger rows.
- [ ] Report includes at least 10 qualitative examples where SFT and DPO differ.
- [ ] Final conclusion states whether DPO improved, harmed, or was
      statistically indistinguishable from SFT.
- [ ] Raw eval output and summary report are stored under `results/evals/`.

## Verification

- `python -m finpost.evals.eval_exact --checkpoints ... --sources gsm8k math ...`
- `accuracy_summary.csv`, `cost_summary.json`, and qualitative examples exist
  under the DPO comparison output directory.

## Blocked by

Requires a completed DPO checkpoint from issue 03.
