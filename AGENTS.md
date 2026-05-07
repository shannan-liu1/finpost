# AGENTS.md

This file orients Codex agents working in this repo.

## Project context

See [CONTEXT.md](./CONTEXT.md) for project intent, target capability, glossary, and explicit out-of-scope. All domain language used in tests, issues, and PRDs should match the `CONTEXT.md` glossary.

## Agent skills

### Issue tracker

Local markdown under `.scratch/<feature-slug>/`. One PRD per workstream, optional `issues/` subdirectory. Full conventions and template in [`.scratch/README.md`](./.scratch/README.md). See [`docs/agents/issue-tracker.md`](./docs/agents/issue-tracker.md).

### Triage labels

Solo project — only `ready-for-agent` is tracked, recorded as a line on each issue file. See [`docs/agents/triage-labels.md`](./docs/agents/triage-labels.md).

### Domain docs

Single-context. `CONTEXT.md` at the root; `docs/adr/` created lazily by `/grill-with-docs` when an ADR-worthy decision lands. See [`docs/agents/domain.md`](./docs/agents/domain.md).
