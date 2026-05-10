# Cost Gate Checklist (pre-run signoff)

Use this checklist before any spend-bearing remote run (baseline, ablations, DPO generation, or evaluation at scale).

## Run metadata

- Run name / ID:
- Date (UTC):
- Operator:
- Issue / ticket link:
- Run type: baseline / ablation / dpo_prep / eval

## 1) Objective gate

- Decision this run will inform:
- Hypothesis being tested:
- Required output artifact(s):
- What we will do if the run fails to answer the question:

## 2) Cost gate

- Provider:
- GPU type:
- Price per hour (quoted at launch time):
- Planned max runtime (hours):
- Hard cap for this run (USD):
- Cumulative spend before this run (USD):
- Global spend cap (USD):
- Remaining budget after this run if maxed (USD):

## 3) Technical readiness gate

- [ ] Local bootstrap passed (`./scripts/local_phase1_bootstrap.sh`)
- [ ] Local mini test passed (`./scripts/local_phase1_minitest.sh`)
- [ ] Config is pinned and committed
- [ ] Input data snapshot/version is recorded
- [ ] Output/checkpoint path exists and is writable
- [ ] Resume-from-checkpoint path validated

## 4) Stop conditions (hard kill)

- [ ] Stop if spend reaches run cap
- [ ] Stop if wall-clock reaches max runtime
- [ ] Stop if loss diverges / NaN appears
- [ ] Stop if no meaningful improvement by step/epoch threshold:
- [ ] Stop if throughput is below acceptable floor:

## 5) Success criteria gate

- Quantitative pass criteria:
- Minimum artifact set required to mark run complete:
- Required logging fields (run_id, seed, wall_clock, final_loss, notes):

## 6) Owner signoff

- Owner decision: approve / deny
- Approved run cap (USD):
- Notes / constraints:
- Owner name:
- Timestamp (UTC):

## 7) Post-run accounting

- Actual runtime (hours):
- Actual spend (USD):
- Outcome summary:
- Follow-up action:
