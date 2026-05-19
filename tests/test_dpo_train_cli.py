"""Behavior tests for the DPO trainer CLI support code."""

from __future__ import annotations

import yaml


def _raw_config(tmp_path) -> dict:
    pairs = tmp_path / "pairs.jsonl"
    pairs.write_text(
        '{"prompt":"Q?","chosen":"A","rejected":"B","source":"gsm8k"}\n',
        encoding="utf-8",
    )
    return {
        "model": {
            "policy_checkpoint": "results/checkpoints/sft-hf",
            "reference_checkpoint": "results/checkpoints/sft-hf",
            "dtype": "float32",
            "use_safetensors": True,
        },
        "data": {
            "pairs_path": str(pairs),
            "manifest_path": str(tmp_path / "manifest.json"),
            "seed": 7,
        },
        "dpo": {"beta": 0.1},
        "training": {
            "max_steps": 100,
            "warmup_steps": 10,
            "lr": 5e-6,
            "weight_decay": 0.01,
            "grad_accum_steps": 2,
            "grad_clip": 1.0,
            "checkpoint_every_n_steps": 25,
            "per_device_pair_batch_size": 2,
        },
        "packing": {"max_seq_len": 128},
        "logging": {
            "wandb_project": "finpost-test",
            "run_name": "dpo-test",
        },
        "checkpointing": {
            "save_dir": str(tmp_path / "checkpoints"),
            "retention_last_n": 2,
            "resume_from": None,
        },
    }


def test_resolve_config_applies_cli_overrides_and_keeps_warmup_valid(tmp_path) -> None:
    """Short canary overrides should not fail because production warmup is longer."""
    from finpost.training.dpo_train import resolve_config

    path = tmp_path / "dpo.yaml"
    raw = _raw_config(tmp_path)
    raw["training"]["warmup_steps"] = 100
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")

    config = resolve_config(
        config_path=path,
        max_steps_override=50,
        resume_from_override=tmp_path / "step-00000010",
    )

    assert config.training.max_steps == 50
    assert config.training.warmup_steps == 49
    assert config.checkpointing.resume_from == tmp_path / "step-00000010"


def test_dpo_checkpoint_round_trips_model_optimizer_scheduler_and_config(tmp_path) -> None:
    """DPO checkpoints stay convertible while avoiding SFT Config validation."""
    import torch

    from finpost.training.dpo_train import (
        DPOConfig,
        load_dpo_checkpoint,
        save_dpo_checkpoint,
    )
    from finpost.training.optim import build_lr_scheduler, build_optimizer

    config = DPOConfig.model_validate(_raw_config(tmp_path))
    model = torch.nn.Linear(3, 2)
    optimizer = build_optimizer(model, lr=config.training.lr, weight_decay=0.0)
    scheduler = build_lr_scheduler(
        optimizer,
        total_steps=config.training.max_steps,
        warmup_steps=config.training.warmup_steps,
    )
    rng_states = {
        "torch": torch.get_rng_state(),
        "torch_cuda": [],
        "numpy": None,
        "python": None,
    }

    checkpoint_dir = save_dpo_checkpoint(
        directory=tmp_path / "checkpoints",
        step=12,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        rng_states=rng_states,
        config=config,
    )
    state = load_dpo_checkpoint(checkpoint_dir)

    assert (checkpoint_dir / "model.safetensors").exists()
    assert (checkpoint_dir / "state.pt").exists()
    assert state.step == 12
    assert state.config.model.policy_checkpoint == config.model.policy_checkpoint
    assert set(state.model_state_dict) == set(model.state_dict())
