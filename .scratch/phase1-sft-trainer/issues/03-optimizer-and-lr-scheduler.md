# 03. Optimizer and learning-rate scheduler factories

- **Status:** Not Started
- **Created:** 2026-05-06
- **Estimated time:** ~1 hour
- **Depends on:** [`01-config-schema`](./01-config-schema.md)

## Goal

Two factory functions that produce the optimizer and the LR scheduler from a `Config`. Tiny module; tested by exercising the factories on known inputs and asserting expected behavior at known steps.

## Scope

**In scope:**
- `build_optimizer(model: nn.Module, lr: float, weight_decay: float) -> torch.optim.AdamW`:
  - Two parameter groups: one with `weight_decay > 0` for normal weights, one with `weight_decay = 0` for biases and `LayerNorm` / `RMSNorm` parameters. This is standard practice — applying weight decay to bias and norm parameters degrades training without benefit.
- `build_lr_scheduler(optimizer, total_steps: int, warmup_steps: int) -> torch.optim.lr_scheduler.LambdaLR`:
  - Cosine schedule with linear warmup. Linear from 0 to peak LR over `warmup_steps`, then cosine decay from peak to ~0 over the remaining `total_steps - warmup_steps`.

**Out of scope:**
- Paged 8-bit AdamW (decided against in Q-E).
- Other schedules (constant, linear-only, polynomial). YAGNI.
- Per-parameter LR overrides. YAGNI.

## Deliverables

```
src/finpost/training/optim.py     # build_optimizer + build_lr_scheduler
tests/test_optim.py
```

## Acceptance criteria

1. `pytest tests/test_optim.py -v` passes.
2. After `build_optimizer(model, lr=1e-4, weight_decay=0.1)`:
   - At least two `param_groups` exist.
   - Bias and `LayerNorm` / `RMSNorm` parameters are in the `weight_decay=0` group.
   - All other parameters are in the `weight_decay=0.1` group.
   - No parameter appears in both groups, no parameter is missing.
3. With `build_lr_scheduler(opt, total_steps=1000, warmup_steps=100)`:
   - At step 0 (before any `.step()`): LR == 0.
   - At step 100 (end of warmup): LR == peak LR.
   - At step 1000: LR ≈ 0 (within `1e-6`).
   - LR is monotonically non-decreasing during warmup and monotonically non-increasing during decay.

## Notes

- The bias/LayerNorm filter: walk `model.named_parameters()`, check name suffix (`'.bias'`) and `param.ndim == 1` (norm weights are 1D). The standard recipe.
- `LambdaLR` is the simplest path: define the lambda inline as `lambda step: warmup_factor(step) if step < warmup_steps else cosine_factor(step)`.
- This is a small module. Don't pull in transformers' `get_cosine_schedule_with_warmup` — implementing it ourselves is part of the learning, and the implementation is six lines.
