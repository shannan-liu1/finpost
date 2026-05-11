# 05. On-Policy Distillation trainer and DPO parity (Stage 4)

- **Status:** Not Started
- **Ready for agent:** No (gated on issue 04)
- **Created:** 2026-05-11
- **Estimated time:** ~3–4 days
- **Depends on:** issue 04 (preference dataset), reuses optimizer/scheduler/checkpoint stack from `phase1-sft-trainer`

## Goal

Implement the On-Policy Distillation (OPD) trainer. It consumes the preference dataset from issue 04 through a Direct-Preference-Optimization-style pairwise loss multiplied by `train_weight`, with a frozen reference model and the existing optimizer/scheduler/checkpoint stack.

## Scope

**In scope:**
- `src/finpost/postraining/opd.py`.
- Loss:
  ```
  loss = -log_sigmoid(beta * ((logp_pi(chosen|x) - logp_ref(chosen|x))
                              - (logp_pi(rejected|x) - logp_ref(rejected|x))))
  loss = train_weight * loss
  ```
- Reference model = SFT-best checkpoint, frozen.
- Policy model = SFT-best checkpoint, training.
- Numerical parity test against the offline DPO loss used by [`phase1-dpo-comparison`](../../phase1-dpo-comparison/PRD.md): per-example loss matches within `1e-5` when `train_weight=1.0`.
- CLI: `scripts/run_opd.py --config <yaml>`.

**Out of scope:**
- The full five-method comparison (issue 06).
- The cost-ledger report builder (issue 07).

## Acceptance criteria

1. `pytest tests/test_opd_loss.py -k parity` passes — OPD loss matches DPO reference within `1e-5` on a fixed batch with `train_weight=1.0`.
2. `python scripts/run_opd.py --config experiments/compute_aware/opd_uniform.yaml --max-steps 20` runs end-to-end on a small slice and writes a checkpoint plus a per-step loss log.
3. Frozen reference model: an in-loop assertion confirms `requires_grad=False` on every parameter of the reference model.
4. Bucket-weighted loss: a unit test with two pairs (`bucket=easy, weight=0.25` and `bucket=ambiguous, weight=1.0`) reports a weighted mean loss equal to `0.25 * loss_easy + 1.0 * loss_ambig` divided by `(0.25 + 1.0)`.
5. Resume determinism on a tiny model: same config + same seed run from scratch for N steps vs. run for N/2 steps then resume produces matching final loss within `torch.allclose(atol=1e-5)`.
