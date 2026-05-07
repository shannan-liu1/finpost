"""Training run configuration: YAML on disk, Pydantic in memory.

Validation lives in Pydantic; the YAML is purely a serialization
format. A typo'd field name, an out-of-range value, or a missing
required field raises ``ValidationError`` at config-load time —
before any expensive training startup happens. The trainer never has
to defend against malformed configs because they can't get past
``Config.from_yaml``.

Six nested sub-models, one per concern:

  ``model``        — base model identifier and dtype
  ``data``         — which datasets to train on, val split
  ``training``     — optimizer, schedule, cadence
  ``packing``      — sequence-packing knobs
  ``logging``      — Weights & Biases run identity
  ``checkpointing`` — save directory, retention, resume

Each sub-model is frozen (``frozen=True``) and forbids extra fields
(``extra="forbid"``), matching the conventions in
``finpost.data.schema``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

# Convention applied to every sub-model in this file:
#   - frozen:        instance is immutable after construction
#   - extra=forbid:  unknown field names raise instead of silently dropping
# Identical pattern to finpost.data.schema.Example. Each sub-model
# declares this explicitly because Pydantic doesn't inherit ConfigDict.
_FROZEN_FORBID = ConfigDict(frozen=True, extra="forbid")


# Allowed dataset sources for Phase 1. Stored as strings (not enum
# objects) so the YAML serializes cleanly as a flat list of literals.
DatasetSource = Literal["gsm8k", "math"]


# Allowed model dtypes. Stored as strings; the trainer converts to
# torch.dtype via ``getattr(torch, dtype_str)`` at load time.
DType = Literal["bfloat16", "float32", "float16"]


class ModelConfig(BaseModel):
    model_config = _FROZEN_FORBID

    base_model_id: str = Field(
        ...,
        min_length=1,
        description="Hugging Face model identifier (e.g. 'google/gemma-3-1b-it').",
    )
    dtype: DType = Field(
        default="bfloat16",
        description="Compute dtype for the model.",
    )
    use_safetensors: bool = Field(
        default=True,
        description="Refuse pickle weights; required by SECURITY.md for the real Gemma model.",
    )


class DataConfig(BaseModel):
    model_config = _FROZEN_FORBID

    sources: list[DatasetSource] = Field(
        ...,
        min_length=1,
        description="Datasets to combine for training.",
    )
    val_split_pct: float = Field(
        default=5.0,
        ge=0.0,
        lt=100.0,
        description="Percentage of train held out (stratified by source) for validation.",
    )
    seed: int = Field(
        default=42,
        ge=0,
        description="Seed for val split, dataset shuffle, and RNG init.",
    )


class TrainingConfig(BaseModel):
    model_config = _FROZEN_FORBID

    max_steps: int = Field(
        ...,
        gt=0,
        description="Total optimizer steps (steps-primary; epochs derived for display only).",
    )
    warmup_steps: int = Field(
        default=100,
        ge=0,
        description="Linear warmup duration in optimizer steps. Must be < max_steps.",
    )
    lr: float = Field(
        ...,
        gt=0.0,
        description="Peak learning rate after warmup.",
    )
    weight_decay: float = Field(
        default=0.01,
        ge=0.0,
        description="AdamW weight decay applied to non-bias / non-norm params.",
    )
    grad_accum_steps: int = Field(
        default=1,
        ge=1,
        description="Micro-batches accumulated per optimizer step.",
    )
    grad_clip: float = Field(
        default=1.0,
        gt=0.0,
        description="Global gradient norm clip threshold.",
    )
    val_every_n_steps: int = Field(
        default=250,
        gt=0,
        description="Validation cadence in optimizer steps.",
    )
    checkpoint_every_n_steps: int = Field(
        default=500,
        gt=0,
        description="Checkpoint cadence in optimizer steps.",
    )
    per_device_batch_size: int = Field(
        default=8,
        ge=1,
        description="Packed rows per GPU forward pass.",
    )

    @model_validator(mode="after")
    def _warmup_must_be_less_than_max(self) -> "TrainingConfig":
        # Cross-field invariant: a cosine schedule with warmup_steps >=
        # max_steps means we'd never reach the decay phase. Almost
        # certainly a config bug. Catch at validation, not at run time.
        if self.warmup_steps >= self.max_steps:
            raise ValueError(
                f"warmup_steps ({self.warmup_steps}) must be < max_steps ({self.max_steps})"
            )
        return self


class PackingConfig(BaseModel):
    model_config = _FROZEN_FORBID

    max_seq_len: int = Field(
        default=4096,
        gt=0,
        description="Maximum tokens per packed row.",
    )
    isolate_documents: bool = Field(
        default=True,
        description="Build per-document attention masks so packed examples can't attend across boundaries.",
    )


class LoggingConfig(BaseModel):
    model_config = _FROZEN_FORBID

    wandb_project: str = Field(
        default="finpost-phase1",
        min_length=1,
        description="Weights & Biases project name.",
    )
    run_name: str | None = Field(
        default=None,
        description="If None, the trainer auto-generates: <model>-<lr>-<seed>-<timestamp>.",
    )


class CheckpointConfig(BaseModel):
    model_config = _FROZEN_FORBID

    save_dir: Path = Field(
        default=Path("results/checkpoints"),
        description="Root directory for run-specific subdirectories.",
    )
    retention_last_n: int = Field(
        default=3,
        ge=0,
        description="Number of most-recent checkpoints to keep on disk.",
    )
    resume_from: Path | None = Field(
        default=None,
        description="If set, load this checkpoint and continue training from its step.",
    )


class Config(BaseModel):
    """Top-level training run configuration."""

    model_config = _FROZEN_FORBID

    model: ModelConfig
    data: DataConfig
    training: TrainingConfig
    # Optional sub-configs default to their respective defaults if the
    # YAML omits them. default_factory rather than default= because
    # mutable defaults are an antipattern; default_factory lazily
    # constructs a fresh instance.
    packing: PackingConfig = Field(default_factory=PackingConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    checkpointing: CheckpointConfig = Field(default_factory=CheckpointConfig)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "Config":
        """Load and validate a YAML config from disk."""
        path = Path(path)
        with path.open("r", encoding="utf-8") as fp:
            raw = yaml.safe_load(fp)
        if not isinstance(raw, dict):
            raise ValueError(
                f"YAML root must be a mapping; got {type(raw).__name__} from {path}"
            )
        return cls.model_validate(raw)

    def to_yaml(self, path: str | Path) -> None:
        """Serialize to a YAML file. Round-trips with from_yaml."""
        path = Path(path)
        # mode="json" so Path objects serialize to strings and Literal
        # types serialize as plain strings. Without it, model_dump()
        # leaves Path() objects in the dict and yaml.safe_dump can't
        # handle them.
        as_dict = self.model_dump(mode="json")
        with path.open("w", encoding="utf-8") as fp:
            yaml.safe_dump(as_dict, fp, sort_keys=False, default_flow_style=False)
