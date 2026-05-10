# 07. Stateful sampler for bit-identical resume (follow-up)

- **Status:** Not Started
- **Created:** 2026-05-10
- **Estimated time:** ~2 hours
- **Depends on:** [`05-trainer`](./05-trainer.md), [`04-checkpointing`](./04-checkpointing.md)

## Goal

Make `Trainer.train()` produce a bit-identical loss curve across a save → fresh-process → resume cycle, even with a shuffling DataLoader. Closes the gap documented in the 2026-05-10 PRD amendment.

## Scope

**In scope:**
- A `StatefulShuffleSampler` (subclass of `torch.utils.data.Sampler`) that owns its own `torch.Generator` and exposes `state_dict()` / `load_state_dict()` to capture and restore the shuffled order plus the consumed-index pointer.
- Wire it into `make_loaders` so the trainer's `train_loader` uses it.
- `save_checkpoint` and `load_checkpoint` capture and restore the sampler's `state_dict` alongside model/optimizer/scheduler/RNG state.
- Update the previously-renamed `test_resume_from_checkpoint_restores_training_mechanism` to ALSO assert true fresh-process resume produces matching losses (rename or add a sibling test).

**Out of scope:**
- Distributed sampler (single GPU only in Phase 1).
- Resuming with a different shuffle seed than the original run.

## Deliverables

```
src/finpost/training/dataset.py        # add StatefulShuffleSampler, wire into make_loaders
src/finpost/training/checkpoint.py     # capture/restore sampler state
tests/test_trainer.py                  # add bit-identical fresh-process resume test
```

## Acceptance criteria

1. `pytest tests/test_trainer.py -v` passes including a new test that runs a true fresh-process resume (no manual batch slicing) and asserts steps N+1..2N losses match the uninterrupted run within atol=1e-5.
2. The `StatefulShuffleSampler` round-trips through `state_dict` / `load_state_dict` such that subsequent iteration produces the same indices as if the sampler had never been serialized.

## Notes

- Pattern: `StatefulShuffleSampler` holds a `torch.Generator`, calls `torch.randperm(len, generator=...)` once per epoch, stores the resulting permutation as a tensor field plus an integer pointer for the current position.
- `state_dict` saves `{"generator_state": gen.get_state(), "permutation": perm.clone(), "index": ptr}`. `load_state_dict` restores all three.
- This is a follow-up to Phase 1's closing milestone, NOT a blocker for it.
