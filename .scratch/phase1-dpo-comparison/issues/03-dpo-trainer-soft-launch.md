# 03 - DPO trainer soft launch

- **Status:** Not Started
- **Ready for agent:** no
- **Depends on:** 02-dpo-loss-and-parity

## Goal

Run a short DPO training path through TinyGPT first, then Qwen 0.5B.

## Scope

**In scope:** DPO trainer loop, offline tracking, checkpointing, resume, short soft-launch configs.

**Out of scope:** full DPO ablation matrix.

## Acceptance criteria

- TinyGPT DPO soft launch runs end to end with offline tracking and checkpointing.
- Qwen 0.5B DPO soft launch runs after the SFT checkpoint exists.
- DPO checkpoints include policy model, optimizer state, scheduler state, step, config, and source SFT checkpoint metadata.
- Resume from DPO checkpoint reproduces the same continuation loss within tolerance.

## Notes / open questions

- DPO soft launch should reuse the SFT trainer infrastructure wherever possible instead of creating a separate logging/checkpointing stack.
