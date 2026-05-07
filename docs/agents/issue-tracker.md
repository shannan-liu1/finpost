# Issue tracker: Local Markdown

Issues and PRDs for this repo live as markdown files under `.scratch/`. Full conventions and the PRD template are in [`/.scratch/README.md`](../../.scratch/README.md).

## Layout

- One workstream per directory: `.scratch/<feature-slug>/`
- The PRD is `.scratch/<feature-slug>/PRD.md`
- Implementation issues are `.scratch/<feature-slug>/issues/<NN>-<slug>.md`, numbered from `01`
- Comments and conversation history append to the bottom of the file under a `## Comments` heading

## Status lines

Two state axes coexist on PRDs and issues:

- **Workstream status** (`- **Status:** Not Started | In Progress | Done | Cancelled | Superseded by <slug>`): the work-lifecycle state of the PRD itself.
- **Ready for agent** (`- **Ready for agent:** yes | no`): whether the issue is specced tightly enough for an autonomous Claude session to pick up and finish without further human input. See [triage-labels.md](./triage-labels.md).

## When a skill says "publish to the issue tracker"

Create a new file under `.scratch/<feature-slug>/`, creating the directory if needed. Use `PRD.md` for product requirements, `issues/<NN>-<slug>.md` for implementation tickets. Use the PRD template at the top of `.scratch/README.md` rather than the generic skill default — this repo's template is richer (Status, Created, Owner, Estimated time, Depends on, Goal, Scope, Deliverables, Acceptance criteria, Notes/open questions).

## When a skill says "fetch the relevant ticket"

Read the file at the referenced path. The user will normally pass the slug or path directly (e.g. `phase1-data-loading` or `.scratch/phase1-data-loading/PRD.md`).
