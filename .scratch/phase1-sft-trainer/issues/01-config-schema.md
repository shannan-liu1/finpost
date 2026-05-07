# 01. Config schema (Pydantic + YAML)

- **Status:** Done (verified 2026-05-06; 17/17 tests pass)
- **Created:** 2026-05-06
- **Estimated time:** ~1 hour
- **Depends on:** none

## Goal

Define the structured config every Phase 1 SFT run reads. YAML on disk, Pydantic in memory. Validation lives in Pydantic; the YAML is just a serialization format.

## Scope

**In scope:**
- A nested Pydantic `Config` model, with sub-models per concern: `model`, `data`, `training`, `packing`, `logging`, `checkpointing`.
- `Config.from_yaml(path: str | Path) -> Config` classmethod.
- `Config.to_yaml(path: str | Path) -> None` round-trip method.
- Field-level validation (e.g. `max_steps > 0`, `lr > 0`, valid model id format).
- Frozen instances (`model_config = ConfigDict(frozen=True)` per nested model).

**Out of scope:**
- The YAML *file itself* (`experiments/baseline.yaml`) — issue 06 cuts that.
- Loading model/tokenizer from config — that's the Trainer's responsibility (issue 05).

## Deliverables

```
src/finpost/training/config.py     # Config + sub-models + from_yaml/to_yaml
tests/test_config.py
```

Suggested top-level fields:

```python
class Config(BaseModel):
    model: ModelConfig            # base_model_id, dtype, use_safetensors
    data: DataConfig              # sources (list[str]), val_split_pct, seed
    training: TrainingConfig      # max_steps, warmup_steps, lr, weight_decay,
                                  # grad_accum_steps, grad_clip,
                                  # val_every_n_steps, checkpoint_every_n_steps
    packing: PackingConfig        # max_seq_len, isolate_documents (bool)
    logging: LoggingConfig        # wandb_project, run_name (or auto-generated)
    checkpointing: CheckpointConfig  # save_dir, retention_last_n, retention_best_by
```

## Acceptance criteria

1. `pytest tests/test_config.py -v` passes.
2. `Config.from_yaml("experiments/baseline.yaml").to_yaml("/tmp/out.yaml")` then re-loaded equals the original.
3. Loading a YAML with a typo'd field name raises `ValidationError` with a useful message (Pydantic does this automatically with `extra="forbid"`).
4. Loading a YAML with `max_steps: -1` raises `ValidationError`.
5. Mutating a loaded `Config.training.lr` raises (frozen model).

## Notes

- Reuse the same Pydantic conventions established in `src/finpost/data/schema.py` (`frozen=True`, `extra="forbid"`).
- YAML library: `pyyaml` is already a transitive dep via Hugging Face / wandb; verify no new dep added.
- Don't over-engineer. Six sub-models max, ~10 fields each.
