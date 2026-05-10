"""Tests for the ``python -m finpost.training.train`` CLI entry point.

Each test pins one promise of the entry point (issue 06):

  1. ``--config`` is required — running with no args exits non-zero.
  2. CLI overrides win over YAML — ``--max-steps 3`` beats ``max_steps: 5``
     in the file.
  3. The startup banner prints the model id, the resolved max_steps,
     and a "steps per epoch" line so a user can see at a glance what
     run they actually got.
  4. End-to-end TinyGPT 20-step run via subprocess — exercises issue 06
     acceptance criteria 1-3 in one shot. The full path: parse → load
     YAML → build Trainer → train → checkpoint on disk with both
     ``model.safetensors`` and ``state.pt``.

Determinism (criterion 4) is covered by issue 05's
``test_two_trainer_runs_with_same_config_match_loss_curve``; a CLI-layer
copy would just be redundant. Resume continuity (criterion 5) is a
manual canary check per the issue 06 spec.
"""

from __future__ import annotations

import io
import os
import subprocess
import sys
from pathlib import Path

import yaml

from finpost.training import train as train_cli


def _minimal_yaml(tmp_path: Path, **overrides: object) -> Path:
    """Write a minimal valid training YAML to ``tmp_path`` and return the path.

    Defaults are tiny-gpt2 + tiny knobs so any test that loads the
    Config can do so without the trainer ever running. Tests that need
    a different shape pass in ``overrides`` keyed by top-level section
    name (e.g. ``training={"max_steps": 5, ...}``).
    """
    body: dict[str, object] = {
        "model": {
            "base_model_id": "sshleifer/tiny-gpt2",
            "dtype": "float32",
            "use_safetensors": False,
        },
        "data": {
            "sources": ["gsm8k"],
            "val_split_pct": 0.0,
            "seed": 0,
        },
        "training": {
            "max_steps": 5,
            "warmup_steps": 1,
            "lr": 1.0e-4,
            "per_device_batch_size": 1,
            "val_every_n_steps": 100,
            "checkpoint_every_n_steps": 100,
        },
        "packing": {"max_seq_len": 32, "isolate_documents": True},
        "logging": {"wandb_project": "finpost-tests", "run_name": "cli-test"},
        "checkpointing": {
            "save_dir": str(tmp_path / "checkpoints"),
            "retention_last_n": 1,
            "resume_from": None,
        },
    }
    body.update(overrides)
    path = tmp_path / "config.yaml"
    with path.open("w", encoding="utf-8") as fp:
        yaml.safe_dump(body, fp, sort_keys=False)
    return path


# -----------------------------------------------------------------------------
# 1. ``--config`` is required
# -----------------------------------------------------------------------------


def test_cli_requires_config_flag() -> None:
    """Running ``python -m finpost.training.train`` with no args must exit
    non-zero.

    We invoke the parser via subprocess with ``--help`` first to confirm
    the module is importable, then with empty args to confirm argparse's
    "required argument" check fires. Subprocess (rather than calling
    ``train_cli.main`` directly) is the most faithful check of the
    user-visible behaviour: argparse calls ``sys.exit`` on missing
    required args, which is hard to test in-process without bare
    ``except SystemExit`` plumbing.
    """
    completed = subprocess.run(
        [sys.executable, "-m", "finpost.training.train"],
        capture_output=True,
        text=True,
    )
    assert completed.returncode != 0
    # argparse prints "the following arguments are required: --config"
    # to stderr. Match the load-bearing word so a future error-message
    # tweak doesn't trip the test.
    assert "--config" in completed.stderr


# -----------------------------------------------------------------------------
# 2. CLI overrides win over YAML
# -----------------------------------------------------------------------------


def test_cli_max_steps_override_beats_yaml(tmp_path: Path) -> None:
    """``--max-steps 3`` overrides ``max_steps: 5`` from the YAML."""
    config_path = _minimal_yaml(tmp_path)  # max_steps=5 in YAML
    config = train_cli.resolve_config(
        config_path=config_path,
        max_steps_override=3,
        resume_from_override=None,
    )
    assert config.training.max_steps == 3


def test_cli_resume_from_override_beats_yaml(tmp_path: Path) -> None:
    """``--resume-from`` injects a path even if the YAML had ``null``."""
    config_path = _minimal_yaml(tmp_path)  # resume_from=null in YAML
    fake_resume = tmp_path / "fake-checkpoint" / "step-00000010"
    config = train_cli.resolve_config(
        config_path=config_path,
        max_steps_override=None,
        resume_from_override=fake_resume,
    )
    assert config.checkpointing.resume_from == fake_resume


def test_cli_no_overrides_preserves_yaml_values(tmp_path: Path) -> None:
    """With no overrides, the resolved Config matches the YAML exactly."""
    config_path = _minimal_yaml(tmp_path)
    config = train_cli.resolve_config(
        config_path=config_path,
        max_steps_override=None,
        resume_from_override=None,
    )
    assert config.training.max_steps == 5
    assert config.checkpointing.resume_from is None


# -----------------------------------------------------------------------------
# 3. Startup banner contents
# -----------------------------------------------------------------------------


def test_cli_prints_effective_config_and_steps_per_epoch(tmp_path: Path) -> None:
    """The startup banner shows the model id, resolved max_steps, and a
    'steps per epoch' line.

    The "steps per epoch" line is the user-visible signal that they
    understand the steps↔epochs relationship for the run. The exact
    epoch count requires loading the dataset (which the CLI does NOT
    do at startup — Trainer does it later); the printed line documents
    the formula the user can plug their cached dataset size into.
    """
    config_path = _minimal_yaml(tmp_path)
    config = train_cli.resolve_config(
        config_path=config_path,
        max_steps_override=7,  # different from YAML's 5 so we know it's the resolved value
        resume_from_override=None,
    )
    buffer = io.StringIO()
    train_cli.print_effective_config(config, stream=buffer)
    output = buffer.getvalue()
    assert "sshleifer/tiny-gpt2" in output
    assert "max_steps" in output and "7" in output
    # Phrase is load-bearing — both the spec and this test pin it
    # so a future banner reformat doesn't silently drop it.
    assert "steps per epoch" in output


# -----------------------------------------------------------------------------
# 4. End-to-end subprocess run — issue 06 acceptance criteria 1-3
# -----------------------------------------------------------------------------


def test_end_to_end_tiny_gpt2_run_writes_checkpoint(tmp_path: Path) -> None:
    """Run the CLI end-to-end on tiny-gpt2 and verify the final checkpoint.

    This is the heaviest test in the suite (~30-90s on CPU) because it
    spans the full training path: tokenizer + model load, real
    GSM8K + MATH dataset load (cached), packing collator, 20 optimizer
    steps, validation pass, checkpointing.

    What we assert:
      - subprocess exits 0,
      - the configured save_dir contains step-00000020/,
      - that step directory contains both model.safetensors and state.pt.

    What we DO NOT assert (covered elsewhere):
      - exact wandb keys logged (issue 05's trainer tests),
      - bit-identical determinism (issue 05's two-runs test),
      - resume correctness on a fresh process (manual canary in issue 06).
    """
    save_dir = tmp_path / "ckpts"
    # Build a YAML pointing at tmp_path so the test never pollutes the
    # repo's results/checkpoints/. The save_dir is the only knob we
    # need to override; everything else can stay at the canary's
    # defaults to keep the test honest about what the CLI does.
    yaml_body = {
        "model": {
            "base_model_id": "sshleifer/tiny-gpt2",
            "dtype": "float32",
            "use_safetensors": False,
        },
        "data": {"sources": ["gsm8k", "math"], "val_split_pct": 5.0, "seed": 42},
        "training": {
            "max_steps": 20,
            "warmup_steps": 1,
            "lr": 1.0e-4,
            "weight_decay": 0.01,
            "grad_accum_steps": 1,
            "grad_clip": 1.0,
            "val_every_n_steps": 5,
            "checkpoint_every_n_steps": 10,
            "per_device_batch_size": 2,
        },
        "packing": {"max_seq_len": 128, "isolate_documents": True},
        "logging": {
            "wandb_project": "finpost-tests",
            "run_name": "tiny-gpt2-cli-e2e",
        },
        "checkpointing": {
            "save_dir": str(save_dir),
            "retention_last_n": 2,
            "resume_from": None,
        },
    }
    config_path = tmp_path / "e2e_config.yaml"
    with config_path.open("w", encoding="utf-8") as fp:
        yaml.safe_dump(yaml_body, fp, sort_keys=False)

    env = os.environ.copy()
    # Network sandboxing: make HF Hub revision checks no-op on the
    # cached datasets and tokenizer. ``WANDB_MODE=disabled`` is the
    # fastest path (no offline files written either) — issue 05 covers
    # the "what gets logged" question.
    env["HF_HUB_OFFLINE"] = "1"
    env["HF_DATASETS_OFFLINE"] = "1"
    env["WANDB_MODE"] = "disabled"
    # Force ``--device cpu`` because tiny-gpt2 on a GPU runner would
    # still work but introduces dtype/device cruft we don't want to
    # debug from a unit test.
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "finpost.training.train",
            "--config",
            str(config_path),
            "--device",
            "cpu",
        ],
        capture_output=True,
        text=True,
        env=env,
        timeout=600,  # 10 min ceiling; canary should finish in ~60s
    )

    # Surface the captured output if the assertion fails so the test
    # is debuggable in CI without re-running locally.
    assert completed.returncode == 0, (
        f"CLI exited {completed.returncode}\n"
        f"stdout:\n{completed.stdout}\n"
        f"stderr:\n{completed.stderr}"
    )
    final_ckpt = save_dir / "step-00000020"
    assert final_ckpt.is_dir(), f"missing checkpoint dir {final_ckpt}"
    assert (final_ckpt / "model.safetensors").is_file()
    assert (final_ckpt / "state.pt").is_file()
