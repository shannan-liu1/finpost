# Triage Labels

This is a solo learning project. Most of the canonical triage state machine doesn't apply — there's no separate reporter / maintainer / AFK-agent split. Only one signal is tracked.

| Canonical role     | Used here? | Recorded as                                                                |
| ------------------ | ---------- | -------------------------------------------------------------------------- |
| `needs-triage`     | No         | —                                                                          |
| `needs-info`       | No         | —                                                                          |
| `ready-for-agent`  | **Yes**    | `- **Ready for agent:** yes` line near the top of the PRD or issue file    |
| `ready-for-human`  | No         | —                                                                          |
| `wontfix`          | No         | Use `Status: Cancelled` on the PRD instead (see issue-tracker.md)          |

When a skill mentions a triage role this repo doesn't track, skip that step. The `/triage` skill is largely inapplicable here.

## What `ready-for-agent` means

The issue is specced tightly enough that an autonomous Claude session could pick it up and finish it without further human input. Concretely:

- Acceptance criteria are falsifiable (commands, file checks, observable outcomes — not "looks good")
- All decisions that need human judgment have already been made and recorded
- The issue references the relevant `CONTEXT.md` terms and any applicable ADRs
- Out-of-scope is explicit, so the agent doesn't gold-plate

If any of those is missing, mark `Ready for agent: no` and resolve the gaps before flipping it.
