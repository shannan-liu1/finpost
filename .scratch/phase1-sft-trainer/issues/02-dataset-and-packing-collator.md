# 02. Dataset and packing collator

- **Status:** Not Started
- **Created:** 2026-05-06
- **Estimated time:** ~3 hours
- **Depends on:** [`01-config-schema`](./01-config-schema.md)

## Goal

Wire `load_gsm8k` and `load_math` into a torch-friendly dataset, hold out a stratified val split with a fixed seed, apply the Gemma chat template at iteration time, and produce **packed** batches (multiple examples per row, up to `max_seq_len`) with per-document loss masks and per-document attention isolation.

This is the load-bearing decision from Q-A and Q-D made concrete. Most of the engineering complexity in this PRD lives here.

## Scope

**In scope:**
- A `PhasedSFTDataset` (subclass of `torch.utils.data.Dataset`) that:
  - Loads combined GSM8K + MATH via existing loaders.
  - Holds out `val_split_pct` (default 5%) stratified by source, with `data.seed` controlling the held-out indices for reproducibility.
  - On `__getitem__`, applies the Gemma chat template (`<start_of_turn>user` / `<start_of_turn>model` wrappers) and tokenizes via the model's tokenizer, returning `(input_ids: tensor, prompt_length: int, source: str)`.
- A `PackingCollator(max_seq_len, eos_token_id, isolate_documents)` callable that:
  - Greedily packs incoming examples into rows up to `max_seq_len`. EOS token between consecutive examples within a row.
  - Builds `labels` per row: prompt positions (per packed example) → IGNORE_INDEX; response positions → target IDs.
  - Builds `position_ids` that reset at each document boundary (so RoPE sees each example as positions 0..L-1 internally).
  - If `isolate_documents=True`, builds a 4D attention mask blocking cross-document attention (positions in document A cannot attend to positions in document B).
  - Returns a dict: `{input_ids, labels, position_ids, attention_mask?}`.
- A `make_loaders(config, tokenizer) -> (train_loader, val_loader)` factory that returns torch `DataLoader`s for train (shuffled, packed) and val (no shuffle, packed for batch eval).

**Out of scope:**
- The chat template choice itself (use Gemma's default; if we ever experiment with custom templates, that's a separate workstream).
- Streaming / disk-based datasets (Phase 1 corpus fits in ~15 MB; load eagerly).

## Deliverables

```
src/finpost/training/dataset.py       # PhasedSFTDataset, PackingCollator, make_loaders
tests/test_dataset.py
```

## Acceptance criteria

1. `pytest tests/test_dataset.py -v` passes.
2. With a fixed `data.seed`, the val split is identical across runs (verify by collecting val IDs and comparing across two `make_loaders` calls).
3. Val and train sets are disjoint (no example appears in both).
4. Stratification: val proportion of GSM8K matches the config'd `val_split_pct` within ±1 percentage point; same for MATH.
5. Collator produces rows where every example's prompt positions are IGNORE_INDEX in `labels` and every response position equals the input token ID at that position.
6. Collator never produces a row exceeding `max_seq_len`.
7. With `isolate_documents=True`, the 4D attention mask zeroes attention from any position in document N to any position in document M (M ≠ N) — verify on a hand-constructed two-document row.
8. `position_ids` reset to 0 at each document boundary within a row.

## Notes

- The packing logic is the meatiest part of this issue. Recommended approach: a small `_pack_one_row(examples, max_seq_len)` helper that consumes from the front of a queue and returns `(packed_input_ids, doc_boundaries)`; build `labels`, `position_ids`, `attention_mask` from `doc_boundaries`.
- Cross-document attention isolation: PyTorch SDPA / FlashAttention accepts a 4D mask of shape `(batch, 1, seq_len, seq_len)`. Build it once per batch from `doc_boundaries`.
- For the smoke test in issue 06, packing is exercised end-to-end. Unit tests here use hand-constructed inputs.
- Reuse `mask_prompt_tokens` from `finpost.training.masking` as much as possible — the per-document case is just calling it inside a loop.
