# CLAUDE.md

This file orients Claude Code agents working in this repo.

## Project context

See [CONTEXT.md](./CONTEXT.md) for project intent, target capability, glossary, and explicit out-of-scope. All domain language used in tests, issues, and PRDs should match the `CONTEXT.md` glossary.

## Agent skills

### Issue tracker

Local markdown under `.scratch/<feature-slug>/`. One PRD per workstream, optional `issues/` subdirectory. Full conventions and template in [`.scratch/README.md`](./.scratch/README.md). See [`docs/agents/issue-tracker.md`](./docs/agents/issue-tracker.md).

### Triage labels

Solo project — only `ready-for-agent` is tracked, recorded as a line on each issue file. See [`docs/agents/triage-labels.md`](./docs/agents/triage-labels.md).

### Domain docs

Single-context. `CONTEXT.md` at the root; `docs/adr/` created lazily by `/grill-with-docs` when an ADR-worthy decision lands. See [`docs/agents/domain.md`](./docs/agents/domain.md).

## Operational runbooks

- [`docs/runbooks/phase1-training-runbook.md`](./docs/runbooks/phase1-training-runbook.md) — Phase 1 gate-by-gate procedure (local bootstrap → TinyGPT canary → Qwen SFT → DPO).
- [`docs/runbooks/runpod-bootstrap.md`](./docs/runbooks/runpod-bootstrap.md) — first-time setup on a fresh RunPod pod. Covers the working-tree restore, the editable-install PEP 660 hook failure (write `.pth` by hand if `pip show finpost` succeeds but `import finpost` fails), and the torch / CUDA driver mismatch (pyproject pins `torch>=2.5` → pip pulls CUDA 13 wheel → RunPod A40 driver is CUDA 12.x → downgrade to `torch==2.4.1+cu124`). Read before sending the user through pod setup or debugging pod-side import errors.

## Known accepted behaviors

Behaviors that are reviewer-flagged or non-obvious but have been explicitly accepted by the owner. Do not re-litigate or "fix" without an explicit new instruction.

### `generated_tokens_decoded` is tokenizer-dependent (eval cost accounting)

The eval CLI reports two token counts in `accuracy_summary.json` and `cost_summary.json`:

- `generated_tokens` — rectangular position count (`new_token_ids.numel()`). Real forward passes. Use as the denominator for compute cost.
- `generated_tokens_decoded` — non-pad count via `_count_non_pad_tokens` in `src/finpost/evals/eval_exact.py`. Approximates content length.

The non-pad count has a known asymmetry the owner has accepted:

- When `pad_token_id != eos_token_id`: post-EOS positions are pad, the EOS token itself is not pad, so the count includes EOS.
- When `pad_token_id == eos_token_id` (our GPT-style default — we set `tokenizer.pad_token = tokenizer.eos_token`): post-EOS positions and the EOS itself are all `pad_id`, so the count excludes EOS.

Net effect: in the pad-equals-eos case the metric undercounts emitted tokens by exactly 1 per early-stopped sequence. Magnitude on a typical Phase 1 run (~1000 early stops out of 200k–400k tokens) is <0.5%.

Accepted because: (1) the count aligns with what `tokenizer.decode(..., skip_special_tokens=True)` actually produces (which is what `details_*.csv` shows), so it matches user-visible content; (2) the error is bounded and one-directional; (3) the rectangular count `generated_tokens` remains exact and is what cost-per-token consumers should use anyway.

If a future requirement needs a tokenizer-independent "emitted length," the upgrade path is a per-row scan that finds the first EOS and counts through it — implementable but currently out of scope.
