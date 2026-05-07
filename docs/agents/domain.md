# Domain Docs

How the engineering skills should consume this repo's domain documentation when exploring the codebase.

## Layout

Single-context repo:

```
/
├── CONTEXT.md
├── docs/adr/      ← created lazily by /grill-with-docs
└── src/
```

## Before exploring, read these

- **`CONTEXT.md`** at the repo root — the project's domain glossary (Filing, Filing excerpt, Numerical extraction, Numerical reasoning, etc.) plus project intent and explicit out-of-scope.
- **`docs/adr/*.md`** if the directory exists — read ADRs that touch the area you're about to work in.

If `docs/adr/` doesn't exist yet, proceed silently. Don't flag its absence; don't suggest creating it upfront. The producer skill (`/grill-with-docs`) creates it lazily when the first ADR-worthy decision lands.

## Use the glossary's vocabulary

When your output names a domain concept (in an issue title, a refactor proposal, a hypothesis, a test name), use the term as defined in `CONTEXT.md`. Don't drift to synonyms. If the glossary says "Filing excerpt", don't write "filing snippet" or "passage".

If the concept you need isn't in the glossary yet, that's a signal — either you're inventing language the project doesn't use (reconsider) or there's a real gap (note it for `/grill-with-docs`).

## Flag ADR conflicts

If your output contradicts an existing ADR, surface it explicitly rather than silently overriding:

> _Contradicts ADR-NNNN (\<title\>) — but worth reopening because…_
