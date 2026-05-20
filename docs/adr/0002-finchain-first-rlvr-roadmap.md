# ADR 0002: FinChain-First RLVR Roadmap

- **Status:** Accepted
- **Date:** 2026-05-19
- **Owner:** shann

## Context

The repo started as a fundamentals-first post-training project: build SFT, DPO, evaluation, and later RL-style methods on small numerical reasoning tasks before applying the stack to finance. SFT has now produced enough signal to freeze the 0.5B phase as a learning artifact and infrastructure canary.

The next question is where to spend scarce GPU time. FinQA is finance-relevant and realistic, but it is mostly a final-answer benchmark over messy filing excerpts. That makes it useful for transfer evaluation, but less ideal as the first RLVR training substrate.

FinChain is a better next benchmark for the method-learning phase because it provides symbolic financial reasoning examples with executable chains. That enables cheaper deterministic verification, step-level diagnostics, and denser reward signals for OPD and GRPO.

## Decision

Use FinChain as the primary benchmark for the next post-training phase.

Use the following method ladder:

1. Base / few-shot evaluation
2. Supervised Fine-Tuning (SFT)
3. Rejection SFT
4. On-Policy Distillation (OPD)
5. Group Relative Policy Optimization (GRPO)
6. Direct Preference Optimization (DPO) as a fundamentals/comparator artifact, not the primary finance path

Use `Qwen/Qwen2.5-1.5B` as the default serious model for the next FinChain loop. Keep `Qwen/Qwen2.5-0.5B` as a canary and treat `Qwen/Qwen3-4B-Base` as a scale-up candidate after the 1.5B loop is interpretable.

Use one 48GB GPU as the default hardware target. Treat multi-GPU runs as a later scaling experiment only after the single-GPU workflow is reproducible.

## Consequences

Positive:

- The reward signal is programmatic and cheap enough for RLVR-style experimentation.
- OPD and GRPO become meaningful method comparisons rather than abstract implementations.
- The project becomes easier to explain in interviews: "I used symbolic financial reasoning to build a verifier-driven post-training loop."
- FinQA remains available as a realism and transfer check.

Negative:

- FinChain may overrepresent template-like symbolic reasoning and underrepresent messy filing grounding.
- A strong FinChain result does not automatically imply broad finance competence.
- The project must explicitly track reward hacking, parseability, KL drift, and template overfit.

Mitigations:

- Hold out FinChain templates/topics when possible.
- Run FinQA transfer after the FinChain method comparison.
- Report per-template failure modes, not just aggregate accuracy.
- Keep the cost ledger and KL/reference drift metrics in the main result table.

## Rejected Alternatives

### FinQA As Primary Training Surface

Rejected for the next phase. FinQA is still valuable, but it is less verifier-rich than FinChain. It should be used for transfer once the RLVR loop works.

### Continue Scaling 0.5B Experiments

Rejected as mainline work. The 0.5B model remains useful for local canaries and debugging, but it is too small to be the serious finance reasoning model.

### Run Every Method Across Every Model

Rejected. The project needs crisp learning and interview artifacts, not a model zoo. One main model, one canary, and a small method ladder are enough.

### PPO First

Rejected for implementation order. PPO is important theory for understanding RLHF, but GRPO is a smaller and more directly useful implementation target for grouped verified rewards in this repo.

## References

- FinChain arXiv: https://arxiv.org/abs/2506.02515
- Qwen2.5-1.5B model card: https://huggingface.co/Qwen/Qwen2.5-1.5B
- RunPod GPU catalog/pricing: https://www.runpod.io/pricing
