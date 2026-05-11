# Workstreams — PRDs and implementation issues

This directory tracks discrete pieces of work for the finpost project. Each workstream lives in its own directory: a `PRD.md` scoping what gets built, plus an optional `issues/` subdirectory of implementation tickets.

Layout matches the local-markdown convention used by the engineering skills (`/to-prd`, `/to-issues`, `/triage`).

## Layout

```
.scratch/
├── <feature-slug>/
│   ├── PRD.md
│   └── issues/
│       ├── 01-<slug>.md
│       └── 02-<slug>.md
```

## Conventions

- **Directory name:** `<feature-slug>` — kebab-case, no number prefix. Order is implicit in the dependency graph and timestamps, not in the name.
- **One PRD per workstream.** A workstream is a piece of work that could be handed to a fresh engineer with no other context and executed.
- **Status lifecycle:** `Not Started` → `In Progress` → `Done` (or `Cancelled` / `Superseded by <slug>`). Recorded as a `Status:` line near the top of the PRD.
- **Acceptance criteria are falsifiable.** Each item should be a command, a file existence check, or a concrete observable outcome — not "looks good" or "works as expected."
- **Append-only history.** When a PRD changes substantively, leave the old text in place and add an "Amendment" section dated below. When a PRD is cancelled or superseded, mark the status and link forward.

## PRD template

```markdown
# <Title — what gets built, present tense>

- **Status:** Not Started | In Progress | Done | Cancelled | Superseded by <slug>
- **Created:** YYYY-MM-DD
- **Owner:** name
- **Estimated time:** rough range
- **Depends on:** <slug>, <slug>

## Goal

One or two sentences: what this workstream delivers and why it exists.

## Scope

**In scope:**
- bullet
- bullet

**Out of scope:**
- bullet (with one-sentence reason or pointer to where it lives instead)

## Deliverables

Concrete artifacts. File paths, function signatures, scripts. If a deliverable is a directory, list its contents.

## Acceptance criteria

Falsifiable checks. Each item is a command that should succeed or an observable outcome that should hold.

## Notes / open questions

Anything we know but haven't resolved. Not load-bearing — just memory.
```

## Workstreams

| Slug | Title | Status |
|------|-------|--------|
| [`repo-skeleton`](./repo-skeleton/PRD.md) | Repository skeleton and environment | Done |
| [`phase1-data-loading`](./phase1-data-loading/PRD.md) | Phase 1 data loading (GSM8K, MATH) | Done |
| [`sft-trainer-skeleton`](./sft-trainer-skeleton/PRD.md) | Supervised Fine-Tuning trainer skeleton | Done |
| [`phase1-sft-trainer`](./phase1-sft-trainer/PRD.md) | Phase 1 production SFT trainer on GSM8K + MATH | In Progress (drafting) |
| [`phase1-training-runbook`](./phase1-training-runbook/PRD.md) | Phase 1 command-by-command training runbook and execution issues | In Progress |
| [`phase1-base-vs-sft-eval`](./phase1-base-vs-sft-eval/PRD.md) | Phase 1 base-vs-SFT exact-answer evaluation harness (consumed by phase1-compute-aware-post-training Stage 0) | Not Started |
| [`phase1-compute-aware-post-training`](./phase1-compute-aware-post-training/PRD.md) | Phase 1.5 compute-aware post-training: rollouts, verifier, bucketing, On-Policy Distillation, cost ledger | Not Started |
| [`phase1-dpo-comparison`](./phase1-dpo-comparison/PRD.md) | Phase 1 DPO trainer and SFT comparison (fixed offline preference dataset; offline-DPO vs. on-policy-OPD is a deliberate split) | Not Started |
| [`phase1-grpo-research`](./phase1-grpo-research/PRD.md) | GRPO research track for verifiable numerical reasoning (consumes Phase 1.5 rollout cache and verifier) | Not Started |
| [`phase2-filing-distillation-dataset`](./phase2-filing-distillation-dataset/PRD.md) | Phase 2 filing distillation dataset | Not Started |
