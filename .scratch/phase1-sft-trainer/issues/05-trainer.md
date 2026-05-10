# 05. Trainer (the main loop)

- **Status:** Not Started
- **Created:** 2026-05-06
- **Estimated time:** ~3 hours
- **Depends on:** [`01-config-schema`](./01-config-schema.md), [`02-dataset-and-packing-collator`](./02-dataset-and-packing-collator.md), [`03-optimizer-and-lr-scheduler`](./03-optimizer-and-lr-scheduler.md), [`04-checkpointing`](./04-checkpointing.md)

## Goal

The actual SFT training loop. Wires together the dataset, the optimizer, the scheduler, the checkpointer, the masked CE loss, and Weights & Biases. After this issue lands, a single `Trainer(config).train()` call runs an end-to-end SFT job.

The first successful run target is `sshleifer/tiny-gpt2` on CPU. Qwen 0.5B is the second target, after the TinyGPT path proves loss measurement, tracking, checkpointing, and resume.

## Scope

**In scope:**
- A `Trainer` class with one entrypoint: `train()`.
- `train()` does:
  1. Set all RNG seeds (`torch`, `torch.cuda`, `numpy`, `random`) from `config.data.seed`.
  2. Load model with `dtype=config.model.dtype` and `use_safetensors=True` per `SECURITY.md`.
  3. Build train and val loaders via `make_loaders(config, tokenizer)`.
  4. Build optimizer and scheduler via the factories from issue 03.
  5. If `config.checkpointing.resume_from` is set: load checkpoint, restore everything (model, optimizer, scheduler, RNGs, step), continue from there.
  6. Initialize Weights & Biases run with `config.logging`.
  7. Loop over batches:
     - Forward pass (`model(input_ids=...)`, no `labels=` kwarg — we compute loss ourselves).
     - `compute_masked_ce_loss(logits, labels)` from `finpost.training.sft`.
     - Backward.
     - On every `grad_accum_steps`: `clip_grad_norm_`, `optimizer.step()`, `scheduler.step()`, `optimizer.zero_grad()`. Increment global step.
     - Log to wandb every step: train loss, current LR, gradient norm.
     - Every `val_every_n_steps`: run validation pass (loss only, batched over val loader), log val loss.
     - Every `checkpoint_every_n_steps`: save checkpoint, apply retention policy.
  8. At loop exit (max_steps reached): final checkpoint, log final metrics, finalize wandb run.
- `Trainer.validate() -> float` helper used by the loop (and callable standalone for debugging).

**Out of scope:**
- Generation-based accuracy validation (decided in Q-B).
- Distributed training. Single GPU.
- Mixed precision via `torch.autocast`. The model is loaded with `dtype=bfloat16` on CUDA, which is sufficient. `autocast` adds complexity without benefit at this scale.
- Hyperparameter search hooks. YAGNI.

## Deliverables

```
src/finpost/training/trainer.py     # Trainer class
tests/test_trainer.py
```

## Acceptance criteria

1. `pytest tests/test_trainer.py -v` passes.
2. **Determinism:** `Trainer(config).train()` with `tiny-gpt2`, fixed seed, max_steps=20 → loss curve matches the same call run twice within `atol=1e-5` per step.
3. **Resume continuity:** train 20 steps end-to-end, capture loss at each step (run A). Train 10 steps, save checkpoint, fresh process, resume, train 10 more (run B). The 11th-through-20th-step losses match within `atol=1e-5`.
4. **wandb logging:** run B's wandb dashboard contains `train/loss`, `train/lr`, `train/grad_norm`, `val/loss` (verified by checking the wandb run record after the test).
5. **Gradient accumulation correctness:** running with `grad_accum_steps=4` and effective batch=32 produces ~equivalent loss curve to `grad_accum_steps=1` and per-device-batch=32 (within `atol=1e-3` due to cross-batch numerical noise).
6. **Local soft launch:** a 20-step `sshleifer/tiny-gpt2` run emits loss metrics, validation loss, checkpoint path, resume metadata, and offline tracking artifacts without requiring GPU or network during the run.

## Notes

- For tests, use `wandb mode="disabled"` so tests don't make network calls.
- Throughput logging: track `tokens_per_sec` as `(useful_tokens_in_batch / batch_wall_time)` averaged across 50 steps. Log every 50 steps, not every step.
- Gradient norm: capture from `torch.nn.utils.clip_grad_norm_`, which returns the pre-clip norm. Log it; useful for debugging exploding gradients.
- Validation: model in eval mode + `torch.no_grad()` for the val pass. Restore train mode after.
- Be explicit about what dtype the loss is in — accumulate train loss in fp32 even if the forward pass is bf16 (small but real numerical stability win).
