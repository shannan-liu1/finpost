# 05 - Decision gates and signoff checklist

- **Status:** Not Started
- **Ready for agent:** no
- **Depends on:** 01-confirm-preflight-and-keys, 02-execute-sft-smoke-and-baseline

## Goal

Capture human decisions required to proceed safely through expensive training and evaluation stages.

## Scope

**In scope:**
- unresolved Plan decisions Q-B and Q-C,
- spend and runtime guardrails,
- checkpoint retention,
- soft-launch go/no-go,
- Supervised Fine-Tuning to Direct Preference Optimization go/no-go,
- Phase 1 to Phase 2 go/no-go.

**Out of scope:**
- implementing trainer code,
- implementing evaluation code,
- making owner decisions without written approval.

## Decisions requiring owner input

1. **Q-B (DPO pair edge cases):** For prompts with all-correct or all-incorrect sampled completions, choose policy:
   - drop prompt,
   - synthesize rejected via controlled corruption,
   - resample additional completions.
2. **Q-C (minimum eval sample size):** confirm power-analysis threshold for detecting about 5 percentage-point improvements.
3. **Run budget guardrails:** max hours/run and max cumulative spend before manual review.
4. **Checkpoint retention policy:** confirm storage budget and pruning rules for long ablation sequences.
5. **Colab fallback policy:** confirm when to abandon free Colab for paid GPU. Default fallback triggers: repeated runtime resets, inability to obtain GPU, or out-of-memory after shrinking sequence length and batch size.
6. **Soft-launch evidence gate:** confirm that TinyGPT local canary and Qwen 20-step soft launch artifacts are sufficient before allowing full Qwen Supervised Fine-Tuning.
7. **SFT baseline go/no-go:** confirm whether the full Qwen SFT baseline is stable enough to start ablations.
8. **DPO go/no-go:** confirm whether pure SFT results are strong enough to justify DPO, or whether trainer/data issues need to be fixed first.
9. **Phase 2 go/no-go:** confirm that Phase 1 comparison evidence is sufficient to start filing-data work.

## Acceptance criteria

- Each decision has an explicit owner-approved answer in writing.
- Approved answers are linked back into the runbook and operational issues.
- Full Qwen Supervised Fine-Tuning, ablations, Direct Preference Optimization, and Phase 2 filing-data work each has a recorded go/no-go decision before spend-bearing execution.
- Each spend-bearing run has a completed `.scratch/templates/cost-gate-checklist.md`.
- Colab runs record GPU type, available VRAM, checkpoint persistence path, and reset/OOM incidents before fallback is approved.
- Direct Preference Optimization approval names the source Qwen Supervised Fine-Tuning checkpoint.

## What this validates

This issue validates governance, not model quality. It prevents accidental spend, premature DPO training, and premature Phase 2 work before Phase 1 has answered its own question.

## Cost gate template

Use `.scratch/templates/cost-gate-checklist.md` before each remote spend-bearing run.

Attach the completed checklist to issue comments for:

- Qwen soft launch, if remote,
- full Qwen baseline,
- ablation batches,
- DPO pair generation,
- DPO soft launch and baseline,
- evaluation at scale.

## Signoff ledger

```markdown
## Decision

- Decision ID:
- Owner:
- Date/time UTC:
- Approved answer:
- Evidence reviewed:
- Scope allowed:
- Budget cap:
- Stop conditions:
- Follow-up issue:
```
