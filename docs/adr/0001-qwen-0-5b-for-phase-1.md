# ADR-0001 - Use Qwen2.5 0.5B As The Phase 1 Base Model

- **Status:** Accepted
- **Date:** 2026-05-08

## Context

Phase 1 is a learning-first post-training stack. The earlier plan used Gemma 3 1B as the base model. That is a reasonable scale target, but it raises the cost and iteration time before the trainer, data path, and evaluation harness have proven themselves.

The project now prioritizes a cheaper experimental substrate that teaches the same mechanics: supervised fine-tuning, masking, packing, checkpointing, Direct Preference Optimization, and later reinforcement-learning variants.

## Decision

Use `Qwen/Qwen2.5-0.5B` as the canonical Phase 1 base model for early Supervised Fine-Tuning and Direct Preference Optimization work.

Use `Qwen/Qwen2.5-0.5B-Instruct` only as a reference baseline when useful, not as the main training substrate.

The trainer must not hardcode Gemma-specific formatting. It should support a configurable prompt/response serialization, with a Qwen-compatible default for Phase 1.

## Consequences

- Local and low-cost cloud iteration becomes more realistic before scaling.
- The project keeps the pedagogical value of training a base model rather than merely polishing an instruction-tuned model.
- Phase 1 acceptance criteria should refer to Qwen 0.5B, not Gemma 1B.
- Any future scale-up should be justified by evidence from the Qwen 0.5B run: stable training, nontrivial evaluation lift, and reproducible checkpoints.

## References

- Qwen2.5-0.5B model card: https://huggingface.co/Qwen/Qwen2.5-0.5B
- Qwen2.5-0.5B-Instruct model card: https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct
