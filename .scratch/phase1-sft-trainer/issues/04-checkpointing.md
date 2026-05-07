# 04. Checkpointing (atomic save/load + retention policy)

- **Status:** Not Started
- **Created:** 2026-05-06
- **Estimated time:** ~1.5 hours
- **Depends on:** [`01-config-schema`](./01-config-schema.md)

## Goal

Save and restore complete training state — model weights, optimizer state, scheduler state, step counter, RNG state, the run config — atomically (no half-written files on crash) and with a retention policy that keeps disk usage bounded.

## Scope

**In scope:**
- `save_checkpoint(directory, step, model, optimizer, scheduler, rng_states, config) -> Path`:
  - Writes to `directory / f"step-{step:08d}"` as a sub-directory.
  - Model weights → `model.safetensors` (per `SECURITY.md` — no pickle).
  - Optimizer + scheduler + RNG + step + config → `state.pt` (PyTorch's pickle is fine here; we wrote it).
  - Atomic semantics: write to a `<dir>.tmp/` first, then `os.replace` to final name. Mid-write crash leaves `<dir>.tmp/` (deletable on restart) and never a corrupt final.
- `load_checkpoint(path) -> CheckpointState` (NamedTuple of model_state_dict, optimizer_state_dict, scheduler_state_dict, step, rng_states, config).
- `apply_retention_policy(directory, last_n: int, best_so_far: Path | None) -> None`:
  - Keep the `last_n` newest checkpoint sub-directories.
  - Keep the one referenced by `best_so_far` (val-loss tracker, owned by the Trainer).
  - Delete everything else.
- A small CLI helper `python -m finpost.training.checkpoint --inspect <path>` that prints the checkpoint's step, config summary, and tensor shapes (useful for debugging).

**Out of scope:**
- Distributed checkpointing (single GPU only in Phase 1).
- Checkpoint conversion / format migration. YAGNI until we need it.

## Deliverables

```
src/finpost/training/checkpoint.py     # save_checkpoint, load_checkpoint, apply_retention_policy
tests/test_checkpoint.py
```

## Acceptance criteria

1. `pytest tests/test_checkpoint.py -v` passes.
2. **Round-trip identity:** save a tiny model + optimizer + scheduler + step → load it back → all state dictionaries are equal (compare via `torch.equal` per tensor, dict equality for the rest).
3. **Atomic write:** simulate a mid-write failure (raise during the model write step). Verify no final-named directory exists; only the `.tmp` directory is present and can be deleted cleanly.
4. **Retention policy:** create 5 mock checkpoint dirs (steps 100, 200, 300, 400, 500). Call `apply_retention_policy(last_n=2, best_so_far=path_to_step_300)`. Verify only step-400, step-500, step-300 remain (last 2 + best).
5. **Resume produces identical trajectory:** train tiny-gpt2 for 10 steps with seed S, snapshot loss after each step (call this trajectory A). Train for 5 steps, save checkpoint, fresh process, load checkpoint, train for 5 more (trajectory B). The 6th-through-10th-step losses in A and B match within `atol=1e-5`.

## Notes

- RNG state to capture: `torch.get_rng_state()`, `torch.cuda.get_rng_state_all()` (if CUDA available), `numpy.random.get_state()`, `random.getstate()`. Restore in the inverse order in `load_checkpoint`.
- The "atomic via temp directory + os.replace" pattern works on POSIX and on Windows (Python's `os.replace` is atomic on both for files and directories).
- `safetensors` write uses `safetensors.torch.save_file(model.state_dict(), path)`. Reading uses `safetensors.torch.load_file(path)`.
- Don't save the model's optimizer-attached references (state dicts only).
