# Phase 1 production Supervised Fine-Tuning trainer on GSM8K + MATH

- **Status:** In Progress (all design decisions resolved; issues cut for execution)
- **Created:** 2026-05-05
- **Owner:** Shannan
- **Estimated time:** ~1.5–2 weeks
- **Depends on:** [`phase1-data-loading`](../phase1-data-loading/PRD.md), [`sft-trainer-skeleton`](../sft-trainer-skeleton/PRD.md)

## Goal

Build the production trainer that consumes Phase 1 data (GSM8K + MATH) and produces ablation-ready Supervised Fine-Tuning runs on Gemma 3 1B. The output of this workstream is the *infrastructure* (configurable training loop with checkpointing, logging, reproducibility); a separate downstream workstream operates that infrastructure to produce the actual ablation matrix. Splitting these keeps each focused: this one is engineering, the next one is experimentation.

After this workstream lands, we can launch a single training run with one config file and one command, get a Weights & Biases dashboard with loss curves and gradient norms, save and resume checkpoints, and reproduce any run from the recorded config + git SHA + RNG seed.

## Scope

**In scope:**

- A unified iterable dataset wrapping `load_gsm8k` and `load_math`, applying the Gemma chat template at training time.
- A configurable training loop with: optimizer (AdamW or paged 8-bit AdamW), cosine LR schedule with linear warmup, gradient accumulation, gradient clipping, mixed precision (bf16 on CUDA).
- Periodic validation (loss only) on a held-out subset.
- Checkpoint save/load (model weights + optimizer state + scheduler state + step counter + RNG state) — atomic writes, last-N + best-by-val-loss retention.
- Weights & Biases logging: train loss per step, val loss every N steps, learning rate, gradient norm, throughput in tokens/sec.
- A YAML config file format, validated by Pydantic.
- A `python -m finpost.training.train --config <path>` entry point.
- Resume from any checkpoint and continue with identical RNG state.
- Reproducibility: seed everything (PyTorch, NumPy, Python random, CUDA), record the config + git SHA + library versions in the run directory.

**Out of scope:**

- Distributed / multi-GPU training. Single A100 80GB on Lambda only for Phase 1 — the project's compute plan.
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
└── baseline.yaml        # Reference config used in the acceptance criteria

tests/
├── test_dataset.py      # chat-template application, mixing, length stats
├── test_config.py       # YAML round-trip, validation of bad configs
├── test_optim.py        # LR schedule values at known steps
├── test_checkpoint.py   # save/load round-trip identity, atomic-write semantics
└── test_trainer.py      # tiny-model end-to-end with known seed produces deterministic loss curve
```

## Acceptance criteria

1. `python -m finpost.training.train --config experiments/baseline.yaml --tiny-model --device cpu --max-steps 20` runs to completion in under 2 minutes, prints loss decreasing from ≈10 to <8.
2. `python -m finpost.training.train --config experiments/baseline.yaml` (Gemma 3 1B on A100) launches a real training run, populates Weights & Biases dashboard, saves checkpoints, completes within the budgeted wall-clock time.
3. Checkpoint round-trip: load saved checkpoint, run one more `train_step` with same input. Resulting loss differs from the next-step loss in the original run by < 1e-6.
4. Resume determinism: same config + same seed run from scratch for N steps vs. run for N/2 steps then resume produces bit-identical model weights at step N (within `torch.allclose(atol=1e-5)`).
5. `pytest tests/test_dataset.py tests/test_config.py tests/test_optim.py tests/test_checkpoint.py tests/test_trainer.py -v` passes.
6. A run launched with the published baseline config produces a Weights & Biases run page with all expected curves (loss train/val, lr, grad_norm, tokens_per_sec).

## Open decisions (to be resolved via grilling)

| ID | Decision | Why it matters |
|----|----------|---------------|
| ~~Q-A~~ | ~~Sequence packing vs. padding~~ → **DECIDED 2026-05-06: packing.** Throughput gain (~3–15× depending on context budget) and production relevance outweigh the ~80–120 lines of added implementation complexity. Per-document loss masking via `mask_prompt_tokens` extension; cross-document attention isolation via 4D mask (or FlashAttention varlen if it gets integrated cleanly). | (resolved) |
| ~~Q-B~~ | ~~Validation cadence and metric scope~~ → **DECIDED 2026-05-06: loss-only validation, every 250 optimizer steps (configurable).** Generation-based accuracy is ~225× more expensive (one forward pass per generated token vs. one total for loss); belongs in the offline eval harness, not the training loop. Trainer reports loss; eval harness consumes saved checkpoints and reports accuracy. | (resolved) |
| ~~Q-C~~ | ~~Training budget unit: epochs or total steps~~ → **DECIDED 2026-05-06: total optimizer steps as the canonical unit.** Configs specify `max_steps`, `warmup_steps`, `val_every_n_steps`, `checkpoint_every_n_steps`. Equivalent epoch count printed at startup for intuition. Modern LLM SFT default; matches nanoGPT, OLMo, axolotl, HF Trainer's `max_steps` override. No `num_train_epochs` in our config — one source of truth. | (resolved) |
| ~~Q-D~~ | ~~Dataset mixing: GSM8K + MATH ratio and ordering~~ → **DECIDED 2026-05-06: random shuffle of combined GSM8K + MATH, no per-source weighting.** Headline approach is combined training, evaluated separately on each test set. Natural ~40/60 token-share toward MATH is acceptable (slight bias to the harder benchmark is what we want). Multi-task transfer ablation (per-dataset training as separate arms) is captured below as a post-headline optional study. | (resolved) |
| ~~Q-E~~ | ~~Optimizer: bf16 AdamW vs. paged 8-bit AdamW~~ → **DECIDED 2026-05-06: standard bf16 AdamW.** A100 80GB has enough headroom; 8-bit solves a memory problem we don't have in Phase 1. Defer 8-bit AdamW to Phase 2 where QLoRA actually needs it. Cleaner ablation signal without quantization noise. | (resolved) |

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

**Multi-task transfer ablation (post-headline, not core scope).** After the headline combined-training run is in hand, a worthwhile follow-up is to train two specialist arms (GSM8K-only and MATH-only) using the same trainer with different config files. Comparing all three on both test sets answers a real research question: does combined training beat specialist training, and by how much. Captured here so it's not lost; cut as its own workstream when/if we get to it. Cost is ~2 additional A100 runs (~$10–20).

## Notes / open questions

- The `phase1-data-loading` workstream returns `list[Example]` (eager). For long training runs this could be re-loaded per epoch or cached in memory. Memory footprint of the full Phase 1 train set is roughly 7,473 (GSM8K) + 7,498 (MATH) ≈ 15K Pydantic Examples at maybe 1 KB each = ~15 MB. Negligible — load once, hold in memory.
- Chat template application is deferred to dataset-construction time, not loader time. Reason: the trainer might want to experiment with different prompt formats (`<bos>question\n` vs. full Gemma `<start_of_turn>user...`) and the loader shouldn't be coupled to that decision.
- The `train.py` entry point is a separate file rather than `python -m finpost.training`; this is so a future `eval.py` can be a sibling and the package layout reads naturally.
