"""Tests for the training run config schema.

Each test pins one promise the schema makes. Together they document
the contract that the trainer (and any future code that reads a
config) can rely on.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from finpost.training.config import (
    CheckpointConfig,
    Config,
    DataConfig,
    LoggingConfig,
    ModelConfig,
    PackingConfig,
    TrainingConfig,
)


def _minimal_valid_config() -> Config:
    """The smallest Config that validates: required fields only, defaults elsewhere."""
    return Config(
        model=ModelConfig(base_model_id="Qwen/Qwen2.5-0.5B"),
        data=DataConfig(sources=["gsm8k", "math"]),
        training=TrainingConfig(max_steps=1000, lr=2.0e-5),
    )


# -----------------------------------------------------------------------------
# Construction and defaults
# -----------------------------------------------------------------------------


def test_minimal_config_constructs_with_defaults_filled_in() -> None:
    cfg = _minimal_valid_config()
    # Required fields present
    assert cfg.model.base_model_id == "Qwen/Qwen2.5-0.5B"
    assert cfg.data.sources == ["gsm8k", "math"]
    assert cfg.training.max_steps == 1000
    assert cfg.training.lr == 2.0e-5
    # Defaults applied for everything else
    assert cfg.model.dtype == "bfloat16"
    assert cfg.data.val_split_pct == 5.0
    assert cfg.data.seed == 42
    assert cfg.training.warmup_steps == 100
    assert cfg.packing.max_seq_len == 4096
    assert cfg.packing.isolate_documents is True
    assert cfg.logging.wandb_project == "finpost-phase1"
    assert cfg.logging.run_name is None
    assert cfg.checkpointing.save_dir == Path("results/checkpoints")
    assert cfg.checkpointing.retention_last_n == 3
    assert cfg.checkpointing.resume_from is None


# -----------------------------------------------------------------------------
# YAML round-trip
# -----------------------------------------------------------------------------


def test_yaml_round_trip(tmp_path: Path) -> None:
    """Config -> YAML -> Config produces an equal Config."""
    cfg = _minimal_valid_config()
    yaml_path = tmp_path / "config.yaml"
    cfg.to_yaml(yaml_path)
    loaded = Config.from_yaml(yaml_path)
    assert loaded == cfg


def test_yaml_round_trip_preserves_overrides(tmp_path: Path) -> None:
    """Non-default values survive a YAML round-trip."""
    cfg = Config(
        model=ModelConfig(base_model_id="Qwen/Qwen2.5-0.5B-Instruct", dtype="float32"),
        data=DataConfig(sources=["math"], val_split_pct=10.0, seed=7),
        training=TrainingConfig(
            max_steps=2000,
            lr=1.0e-4,
            warmup_steps=50,
            grad_accum_steps=8,
            grad_clip=0.5,
        ),
        packing=PackingConfig(max_seq_len=2048, isolate_documents=False),
        logging=LoggingConfig(wandb_project="custom", run_name="my-run"),
        checkpointing=CheckpointConfig(
            save_dir=Path("/tmp/finpost-ckpts"),
            retention_last_n=5,
        ),
    )
    yaml_path = tmp_path / "custom.yaml"
    cfg.to_yaml(yaml_path)
    loaded = Config.from_yaml(yaml_path)
    assert loaded == cfg


def test_from_yaml_with_non_mapping_root_raises(tmp_path: Path) -> None:
    """A YAML file whose root is a list (or scalar) is a config error."""
    yaml_path = tmp_path / "bad.yaml"
    yaml_path.write_text("- this\n- is\n- a list\n")
    with pytest.raises(ValueError, match="must be a mapping"):
        Config.from_yaml(yaml_path)


# -----------------------------------------------------------------------------
# Required-field and extra-field enforcement
# -----------------------------------------------------------------------------


def test_required_field_missing_raises() -> None:
    """ModelConfig without base_model_id is invalid."""
    with pytest.raises(ValidationError):
        ModelConfig()  # type: ignore[call-arg]


def test_extra_field_raises_at_subconfig() -> None:
    """A typo in a sub-model field name should fail loudly, not be dropped."""
    with pytest.raises(ValidationError):
        ModelConfig(
            base_model_id="x",
            dtypee="bfloat16",  # typo  # type: ignore[call-arg]
        )


def test_extra_field_raises_at_top_level() -> None:
    """Top-level Config also forbids extras."""
    with pytest.raises(ValidationError):
        Config(  # type: ignore[call-arg]
            model=ModelConfig(base_model_id="x"),
            data=DataConfig(sources=["gsm8k"]),
            training=TrainingConfig(max_steps=1, lr=1e-5),
            unknown_section={"foo": "bar"},
        )


# -----------------------------------------------------------------------------
# Value constraints (Field(... ge=, gt=, lt=, ...))
# -----------------------------------------------------------------------------


def test_max_steps_must_be_positive() -> None:
    with pytest.raises(ValidationError):
        TrainingConfig(max_steps=0, lr=1e-5)


def test_lr_must_be_positive() -> None:
    with pytest.raises(ValidationError):
        TrainingConfig(max_steps=1000, lr=0.0)


def test_warmup_must_be_less_than_max_steps() -> None:
    """The cross-field model_validator catches this, not a per-field check."""
    with pytest.raises(ValidationError, match="warmup_steps"):
        TrainingConfig(max_steps=100, lr=1e-5, warmup_steps=200)


def test_warmup_equal_to_max_also_raises() -> None:
    """warmup == max_steps means we never reach the decay phase. Reject."""
    with pytest.raises(ValidationError, match="warmup_steps"):
        TrainingConfig(max_steps=100, lr=1e-5, warmup_steps=100)


def test_val_split_pct_bounds() -> None:
    with pytest.raises(ValidationError):
        DataConfig(sources=["gsm8k"], val_split_pct=-0.1)
    with pytest.raises(ValidationError):
        DataConfig(sources=["gsm8k"], val_split_pct=100.0)


def test_empty_sources_rejected() -> None:
    """min_length=1 means an empty list is invalid."""
    with pytest.raises(ValidationError):
        DataConfig(sources=[])


# -----------------------------------------------------------------------------
# Literal enforcement (enums)
# -----------------------------------------------------------------------------


def test_unknown_dataset_source_rejected() -> None:
    """The dataset source Literal enforces enum membership at validation time."""
    with pytest.raises(ValidationError):
        DataConfig(sources=["foo"])  # type: ignore[list-item]


def test_finchain_dataset_source_is_allowed() -> None:
    """FinChain can use the existing SFT trainer data config."""
    cfg = DataConfig(sources=["finchain"], val_split_pct=10.0)
    assert cfg.sources == ["finchain"]


def test_unknown_dtype_rejected() -> None:
    with pytest.raises(ValidationError):
        ModelConfig(base_model_id="x", dtype="float64")  # type: ignore[arg-type]


# -----------------------------------------------------------------------------
# Immutability (frozen=True)
# -----------------------------------------------------------------------------


def test_top_level_config_is_frozen() -> None:
    cfg = _minimal_valid_config()
    with pytest.raises(ValidationError):
        cfg.model = ModelConfig(base_model_id="other")  # type: ignore[misc]


def test_subconfig_is_frozen() -> None:
    cfg = _minimal_valid_config()
    with pytest.raises(ValidationError):
        cfg.training.lr = 5.0e-5  # type: ignore[misc]
