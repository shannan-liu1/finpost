# Phase 1 production Supervised Fine-Tuning trainer on GSM8K + MATH

- **Status:** In Progress (all design decisions resolved; issues cut for execution)
- **Created:** 2026-05-05
- **Owner:** Shannan
- **Estimated time:** ~1.5–2 weeks
- **Depends on:** [`phase1-data-loading`](../phase1-data-loading/PRD.md), [`sft-trainer-skeleton`](../sft-trainer-skeleton/PRD.md)

## Goal

Build the production trainer that consumes Phase 1 data (GSM8K + MATH) and produces ablation-ready Supervised Fine-Tuning runs on `Qwen/Qwen2.5-0.5B`. The output of this workstream is the *infrastructure* (configurable training loop with checkpointing, logging, reproducibility); a separate downstream workstream operates that infrastructure to produce the actual ablation matrix. Splitting these keeps each focused: this one is engineering, the next one is experimentation.

After this workstream lands, we can launch a single training run with one config file and one command, get a Weights & Biases dashboard with loss curves and gradient norms, save and resume checkpoints, and reproduce any run from the recorded config + git SHA + RNG seed.

## Current assumption check

What exists today:
- dataset loaders and normalized `Example` schema,
- final-answer parsing for GSM8K and MATH,
- prompt-token masking,
- explicit masked cross-entropy,
- one-batch smoke training primitive,
- validated YAML/Pydantic config schema,
- `PhasedSFTDataset` with deterministic stratified train/val splitting,
- `PackingCollator` with greedy sequence packing, prompt-label masking, position resets, document boundaries, and optional 4D cross-document attention isolation,
- `make_loaders(config, tokenizer)` for packed train/val DataLoaders.

What does **not** exist yet:
- optimizer/scheduler factories,
- checkpoint save/load,
- full trainer loop,
- `python -m finpost.training.train` entry point,
- a real TinyGPT or Qwen SFT soft-launch through the production path.

The immediate implementation sequence is therefore: optimizer/scheduler -> checkpointing -> trainer loop -> CLI/configs -> TinyGPT local soft launch -> Qwen 0.5B SFT baseline.

## Scope

**In scope:**

- A unified iterable dataset wrapping `load_gsm8k` and `load_math`, applying a configurable prompt/response serialization at training time. The Phase 1 default is Qwen-compatible and must not hardcode Gemma-specific tokens.
- A configurable training loop with: optimizer (AdamW or paged 8-bit AdamW), cosine LR schedule with linear warmup, gradient accumulation, gradient clipping, mixed precision (bf16 on CUDA).
- Periodic validation (loss only) on a held-out subset.
- Checkpoint save/load (model weights + optimizer state + scheduler state + step counter + RNG state) — atomic writes, last-N + best-by-val-loss retention.
- Weights & Biases logging: train loss per step, val loss every N steps, learning rate, gradient norm, throughput in tokens/sec.
- A YAML config file format, validated by Pydantic.
- A `python -m finpost.training.train --config <path>` entry point.
- Resume from any checkpoint and continue with identical RNG state.
- Reproducibility: seed everything (PyTorch, NumPy, Python random, CUDA), record the config + git SHA + library versions in the run directory.

**Out of scope:**

- Distributed / multi-GPU training. Phase 1 should start with the local TinyGPT canary and Qwen 0.5B soft launch; larger remote hardware is an operational choice only after those pass.
- Direct Preference Optimization. Separate workstream once SFT is solid.
- Generation-based evaluation during training (running test problems and grading the model's answers). Loss-based validation only here; the full eval harness is its own workstream.
- Hyperparameter search infrastructure (Optuna, Ray Tune, etc.). Phase 1 ablations are a small manually-launched matrix; we don't need a sweep framework.
- The actual ablation matrix runs and analysis. Separate downstream workstream consumes this trainer.
- LoRA / QLoRA. Phase 1 uses full fine-tuning. QLoRA arrives in the Phase 2 workstream.

## Deliverables

```
src/finpost/training/
├── dataset.py           # PhasedSFTDataset wrapping load_gsm8k + load_math + chat template
├── config.py            # Pydantic Config schema (mirrors YAML structure)
├── optim.py             # build_optimizer + build_lr_scheduler factories
├── checkpoint.py        # save_checkpoint, load_checkpoint, atomic-write helpers
├── trainer.py           # The Trainer class with the main training loop
├── train.py             # __main__ entry point: parse --config, launch Trainer
└── (existing) masking.py, sft.py — unchanged

experiments/
├── local_tiny_gpt2.yaml # CPU/local infra canary config
└── baseline.yaml        # Qwen 0.5B reference config used after local canary passes

tests/
├── test_dataset.py      # chat-template application, mixing, length stats
├── test_config.py       # YAML round-trip, validation of bad configs
├── test_optim.py        # LR schedule values at known steps
├── test_checkpoint.py   # save/load round-trip identity, atomic-write semantics
└── test_trainer.py      # tiny-model end-to-end with known seed produces deterministic loss curve
```

## Acceptance criteria

1. `WANDB_MODE=offline python -m finpost.training.train --config experiments/local_tiny_gpt2.yaml --device cpu --max-steps 20` runs to completion on a 4 GB local machine, logs train/val loss, and writes a checkpoint. PowerShell equivalent: set `$env:WANDB_MODE="offline"` before the command.
2. `python -m finpost.training.train --config experiments/baseline.yaml --max-steps 20` launches the Qwen 0.5B soft-launch path in the target environment, logs train/val loss, and writes a checkpoint.
3. Full Qwen 0.5B SFT baseline runs only after the TinyGPT local canary and Qwen 20-step soft launch both pass.
4. Checkpoint round-trip: load saved checkpoint, run one more `train_step` with same input. Resulting loss differs from the next-step loss in the original run by < 1e-6.
5. Resume determinism: same config + same seed run from scratch for N steps vs. run for N/2 steps then resume produces bit-identical model weights at step N (within `torch.allclose(atol=1e-5)`).
6. `pytest tests/test_dataset.py tests/test_config.py tests/test_optim.py tests/test_checkpoint.py tests/test_trainer.py -v` passes.
7. A run launched with the published baseline config produces a Weights & Biases run page or offline run directory with all expected curves (loss train/val, lr, grad_norm, tokens_per_sec).

## Open decisions (to be resolved via grilling)

| ID | Decision | Why it matters |
|----|----------|---------------|
| ~~Q-A~~ | ~~Sequence packing vs. padding~~ → **DECIDED 2026-05-06: packing.** Throughput gain (~3–15× depending on context budget) and production relevance outweigh the ~80–120 lines of added implementation complexity. Per-document loss masking via `mask_prompt_tokens` extension; cross-document attention isolation via 4D mask (or FlashAttention varlen if it gets integrated cleanly). | (resolved) |
| ~~Q-B~~ | ~~Validation cadence and metric scope~~ → **DECIDED 2026-05-06: loss-only validation, every 250 optimizer steps (configurable).** Generation-based accuracy is ~225× more expensive (one forward pass per generated token vs. one total for loss); belongs in the offline eval harness, not the training loop. Trainer reports loss; eval harness consumes saved checkpoints and reports accuracy. | (resolved) |
| ~~Q-C~~ | ~~Training budget unit: epochs or total steps~~ → **DECIDED 2026-05-06: total optimizer steps as the canonical unit.** Configs specify `max_steps`, `warmup_steps`, `val_every_n_steps`, `checkpoint_every_n_steps`. Equivalent epoch count printed at startup for intuition. Modern LLM SFT default; matches nanoGPT, OLMo, axolotl, HF Trainer's `max_steps` override. No `num_train_epochs` in our config — one source of truth. | (resolved) |
| ~~Q-D~~ | ~~Dataset mixing: GSM8K + MATH ratio and ordering~~ → **DECIDED 2026-05-06: random shuffle of combined GSM8K + MATH, no per-source weighting.** Headline approach is combined training, evaluated separately on each test set. Natural ~40/60 token-share toward MATH is acceptable (slight bias to the harder benchmark is what we want). Multi-task transfer ablation (per-dataset training as separate arms) is captured below as a post-headline optional study. | (resolved) |
| ~~Q-E~~ | ~~Optimizer: bf16 AdamW vs. paged 8-bit AdamW~~ → **DECIDED 2026-05-06: standard bf16 AdamW.** Use a target environment with enough headroom for full Qwen 0.5B fine-tuning; 8-bit solves a memory problem we do not need to introduce in Phase 1. Defer 8-bit AdamW to Phase 2 where QLoRA actually needs it. Cleaner ablation signal without quantization noise. | (resolved) |

## Resolved-by-default decisions

These I'll just pick unless you push back:

- **Config format:** YAML. Industry standard, human-readable, comments allowed. Validated into a Pydantic `Config` model so typos and bad values raise loudly.
- **Train/val split:** hold out 5% of the combined train set, stratified by source (GSM8K + MATH each contribute proportionally). Held-out set gets a fixed seed so it's reproducible across runs.
- **Checkpoint storage:** `results/checkpoints/<run_name>/step-<N>.pt` with `<run_name>` derived from config + timestamp. Keep last 3 checkpoints + the best by val loss; prune the rest to save disk.
- **Atomic writes:** write to `step-<N>.pt.tmp`, then `os.replace` to final name. Avoids corrupted checkpoints on crash mid-write.
- **Reproducibility:** seed `torch`, `torch.cuda`, `numpy`, `random`. Set `torch.use_deterministic_algorithms(False)` (true would slow training meaningfully and we don't need bit-exact reproducibility, only seed-stable-up-to-floating-point).
- **Logging:** Weights & Biases. Run-naming convention: `<dataset>-<lr>-<seed>-<timestamp>`.
- **Gradient clipping:** clip to global norm 1.0. Standard Phase 1 default.

## Future / optional follow-up

**Multi-task transfer ablation (post-headline, not core scope).** After the headline combined-training run is in hand, a worthwhile follow-up is to train two specialist arms (GSM8K-only and MATH-only) using the same trainer with different config files. Comparing all three on both test sets answers a real research question: does combined training beat specialist training, and by how much. Captured here so it's not lost; cut as its own workstream when/if we get to it. Cost depends on the target environment selected after the Qwen soft launch.

## Notes / open questions

- The `phase1-data-loading` workstream returns `list[Example]` (eager). For long training runs this could be re-loaded per epoch or cached in memory. Memory footprint of the full Phase 1 train set is roughly 7,473 (GSM8K) + 7,498 (MATH) ≈ 15K Pydantic Examples at maybe 1 KB each = ~15 MB. Negligible — load once, hold in memory.
- Prompt/response serialization is deferred to dataset-construction time, not loader time. Reason: the trainer might want to experiment with different prompt formats (`<bos>question\n`, Qwen ChatML-style messages, or a plain question/answer format) and the loader shouldn't be coupled to that decision.
- The `train.py` entry point is a separate file rather than `python -m finpost.training`; this is so a future `eval.py` can be a sibling and the package layout reads naturally.
- Local model ladder: `sshleifer/tiny-gpt2` is the infrastructure canary for a 4 GB local machine; `distilgpt2` is an optional heavier CPU sanity check; `Qwen/Qwen2.5-0.5B` is the actual Phase 1 model.

## Amendment 2026-05-10 — DataLoader iterator state out of scope for resume

Issue 05 (Trainer) ships with a resume mechanism that restores model, optimizer, scheduler, RNG state, and step counter from a checkpoint. It does NOT restore the DataLoader iterator's position. On a fresh-process resume, `iter(train_loader)` starts a new shuffle, which consumes RNG state and produces a different batch sequence than the original run was on at the checkpoint step.

Practical impact: a real resume will replay already-seen batches for the first portion of the post-resume epoch and may show a small (~sub-1% over a long run) loss blip in the first few steps. This is the same limitation that HuggingFace `Trainer`, PyTorch Lightning, and `accelerate` all have without a stateful sampler.

Issue 05's criterion 3 ("resume continuity, atol=1e-5 over steps 11..20") was originally written assuming bit-identical resume. The implementation provides bit-identical RESUME MECHANISM (params/opt/scheduler/RNG round-trip) but not bit-identical loss trajectory. The renamed test `test_resume_from_checkpoint_restores_training_mechanism` validates the mechanism by feeding run B the correct batches manually.

Stateful-sampler checkpointing is filed as a follow-up issue and explicitly out of scope for the Phase 1 closing milestone (issue 06). For Phase 1's use case (small datasets, mostly single-shot Colab runs), the loss blip is operationally invisible.
