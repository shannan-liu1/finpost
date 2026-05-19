# AGENTS.md

This file orients Codex agents working in this repo.

## Project Context

See [CONTEXT.md](./CONTEXT.md) for project intent, target capability, glossary, and explicit out-of-scope. All domain language used in tests, issues, and PRDs should match the `CONTEXT.md` glossary.

The active direction is FinChain-first RLVR:

- FinChain is the primary benchmark and verifier substrate.
- FinQA is a transfer check, not the main training surface.
- Qwen3-4B-Base with LoRA/QLoRA is the default serious model.
- Qwen2.5-0.5B remains the local canary and cheap trainer regression model.
- OPD and GRPO are the highest-signal methods for the next phase; DPO remains useful as a fundamentals/comparator artifact.
- Treat FinChain as a controlled RLVR laboratory, not as proof of broad finance competence. Transfer checks on FinQA/TAT-QA/FinanceBench-style tasks are required before making strong claims.

## Agent Skills

### Issue Tracker

Local markdown under `.scratch/<feature-slug>/`. One PRD per workstream, optional `issues/` subdirectory. Full conventions and template in [`.scratch/README.md`](./.scratch/README.md). See [`docs/agents/issue-tracker.md`](./docs/agents/issue-tracker.md).

### Triage Labels

Solo project. Only `ready-for-agent` is tracked, recorded as a line on each issue file. See [`docs/agents/triage-labels.md`](./docs/agents/triage-labels.md).

### Domain Docs

Single-context. `CONTEXT.md` at the root; architecture decisions live in `docs/adr/` when an ADR-worthy decision lands. See [`docs/agents/domain.md`](./docs/agents/domain.md).

## Operational Runbooks

- [`docs/runbooks/phase1-training-runbook.md`](./docs/runbooks/phase1-training-runbook.md) - Phase 1 gate-by-gate procedure (local bootstrap -> TinyGPT canary -> Qwen SFT -> DPO).
- [`docs/runbooks/runpod-bootstrap.md`](./docs/runbooks/runpod-bootstrap.md) - first-time setup on a fresh RunPod pod. Covers the working-tree restore, the editable-install PEP 660 hook failure (write `.pth` by hand if `pip show finpost` succeeds but `import finpost` fails), and the torch / CUDA driver mismatch (`torch>=2.5` can pull a CUDA 13 wheel on an A40/CUDA 12.x pod; downgrade to `torch==2.4.1+cu124` if needed).
- [`docs/runbooks/finchain-rlvr-study-flow.md`](./docs/runbooks/finchain-rlvr-study-flow.md) - active FinChain-first study flow for notebooks, model/GPU choices, OPD, GRPO, cost ledgers, and interview artifacts.
- [`STUDY.md`](./STUDY.md) and [`STUDY.html`](./STUDY.html) - professor-style motivation, assumptions, contrarian analysis, and the active study map.
- [`docs/distributed-training-and-platforms.md`](./docs/distributed-training-and-platforms.md) - DDP/FSDP/ZeRO, sharding vocabulary, platform recommendations, and the multi-GPU learning path.

## Known Accepted Behaviors

Behaviors that are reviewer-flagged or non-obvious but have been explicitly accepted by the owner. Do not re-litigate or "fix" without an explicit new instruction.

### `generated_tokens_decoded` Is Tokenizer-Dependent

The eval CLI reports two token counts in `accuracy_summary.json` and `cost_summary.json`:

- `generated_tokens` - rectangular position count (`new_token_ids.numel()`). Real forward passes. Use as the denominator for compute cost.
- `generated_tokens_decoded` - non-pad count via `_count_non_pad_tokens` in `src/finpost/evals/eval_exact.py`. Approximates content length.

The non-pad count has a known asymmetry the owner has accepted:

- When `pad_token_id != eos_token_id`: post-EOS positions are pad, the EOS token itself is not pad, so the count includes EOS.
- When `pad_token_id == eos_token_id` (our GPT-style default, where `tokenizer.pad_token = tokenizer.eos_token`): post-EOS positions and the EOS itself are all `pad_id`, so the count excludes EOS.

Net effect: in the pad-equals-eos case the metric undercounts emitted tokens by exactly 1 per early-stopped sequence. Magnitude on a typical Phase 1 run (~1000 early stops out of 200k-400k tokens) is less than 0.5%.

Accepted because: (1) the count aligns with what `tokenizer.decode(..., skip_special_tokens=True)` actually produces, which is what `details_*.csv` shows; (2) the error is bounded and one-directional; (3) the rectangular count `generated_tokens` remains exact and is what cost-per-token consumers should use anyway.

If a future requirement needs a tokenizer-independent "emitted length," the upgrade path is a per-row scan that finds the first EOS and counts through it. That is implementable but currently out of scope.
