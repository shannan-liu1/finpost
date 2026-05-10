# 0003. Supervised Fine-Tuning trainer skeleton

- **Status:** Done (verified 2026-05-05)
- **Created:** 2026-05-05
- **Owner:** Shannan
- **Estimated time:** ~1.5 hours
- **Depends on:** [`repo-skeleton`](../repo-skeleton/PRD.md)

## Goal

Build the bones of the Supervised Fine-Tuning training loop: load Gemma 3 1B, prepare a single batch of fake `(prompt, response)` examples, apply prompt-token masking, compute cross-entropy loss, run one optimizer step, print the loss. No real data, no checkpointing, no logging integration. The deliverable is the smallest end-to-end training run that exercises every mechanical concept the real trainer will need: tokenization with a chat template, prompt-vs-response masking, mixed precision, the loss reduction, and the optimizer step.

After this PRD we will know that our understanding of masking, the loss shape, and the model interface is correct — before we wire any of it up to real data or real compute.

## Scope

**In scope:**
- A `mask_prompt_tokens(input_ids, prompt_lengths) -> labels` function that returns a `labels` tensor where positions corresponding to the prompt are set to `-100` (the cross-entropy ignore index) and positions corresponding to the response keep their token IDs.
- A `train_step(model, batch, optimizer)` function that does forward, loss, backward, optimizer step. Loss reduction is mean over non-ignored response tokens.
- A `scripts/sft_smoke.py` script that:
  - Loads Gemma 3 1B (or a small stand-in if the `--tiny-model` flag is passed — useful for CPU-only runs and CI).
  - Constructs three hand-written `(prompt, response)` examples directly in the script (no data files).
  - Tokenizes them with the Gemma chat template.
  - Calls `train_step` for 5 iterations.
  - Prints the loss after each step.
- A pytest test verifying:
  - `mask_prompt_tokens` correctly sets prompt positions to `-100` and leaves response positions unchanged for a few hand-constructed inputs.
  - The masked positions match what is intended for a known chat-template-formatted input.

**Out of scope:**
- Loading real datasets (PRD 0002 handles loading; integration is a later PRD).
- Gradient accumulation (covered in the full Supervised Fine-Tuning trainer PRD).
- Learning-rate scheduling.
- Checkpointing.
- Weights & Biases logging.
- Direct Preference Optimization (separate PRD).

## Deliverables

```
src/finpost/training/
├── __init__.py
├── masking.py             # mask_prompt_tokens(...)
└── sft.py                 # train_step(...) and supporting helpers

scripts/
└── sft_smoke.py           # end-to-end one-batch smoke run

tests/
└── test_masking.py
```

## Acceptance criteria

1. `python scripts/sft_smoke.py --tiny-model --device cpu` runs to completion in under 60 seconds.
2. `python scripts/sft_smoke.py --tiny-model --device cpu` prints 5 finite, non-`NaN` loss values.
3. The 5 printed loss values trend downward (strict monotonicity is *not* required — the run is too small for that — but the final loss should be lower than the first by some margin).
4. `pytest tests/test_masking.py -v` passes.
5. The masking test includes at least one assertion of the form: "given a prompt of length P and a response of length R, exactly P positions in `labels` are `-100` and exactly R positions match the response tokens."

## Notes / open questions

- The `--tiny-model` stand-in: candidate is `google/gemma-3-270m` (or whichever currently-smallest Gemma we can load). Confirm at execution time.
- Real Gemma 3 1B requires Hugging Face authentication and a meaningful download (~2 GB). The `--tiny-model` flag is the path that runs locally without that download.
- For CPU runs, mixed precision is a no-op. The smoke script should `--dtype` flag for explicit control; default to `float32` on CPU and `bfloat16` on CUDA.
- This PRD intentionally does *not* wire up an optimizer scheduler or gradient accumulation; those are mechanical additions handled in a later PRD when we run the real ablation matrix.

## Amendment 1 - 2026-05-09

This completed skeleton PRD is now historical. The current Phase 1 model decision is ADR-0001: `Qwen/Qwen2.5-0.5B` for the real SFT/DPO substrate, and `sshleifer/tiny-gpt2` for the local infrastructure canary.

Do not carry forward the old Gemma-specific smoke-script assumptions into the production trainer. The production path is defined by `.scratch/phase1-sft-trainer/PRD.md` and should use configurable prompt/response serialization.
