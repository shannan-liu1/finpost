"""Tests for atomic checkpoint save/load and retention policy.

Each test pins one invariant of the checkpoint module:

1. Round-trip identity: save then load yields equal state dicts,
   identical step, identical config, identical RNG states.
2. Atomic write: a mid-write failure leaves only the ``.tmp`` directory
   on disk and never a partially-written final directory.
3. Retention policy: keeps the ``last_n`` newest plus the ``best_so_far``
   directory; deletes everything else.
4. Resume identity: train N steps, save at step K, fresh model + load,
   train N-K more — the loss trajectory after step K matches the
   uninterrupted run within tight tolerance.
5. CLI inspection prints step + config + tensor shapes for the
   debugging affordance.
"""

from __future__ import annotations

import os
import random
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn as nn

from finpost.training.checkpoint import (
    CheckpointState,
    apply_retention_policy,
    load_checkpoint,
    save_checkpoint,
)
from finpost.training.config import (
    Config,
    DataConfig,
    ModelConfig,
    TrainingConfig,
)
from finpost.training.optim import build_lr_scheduler, build_optimizer


def _tiny_model() -> nn.Module:
    """A non-tied 2-layer model.

    Using a hand-built ``nn.Sequential`` avoids any tied-parameter
    ambiguity (Hugging Face GPT-2 ties ``wte`` and ``lm_head``).
    Keeping the round-trip test on a clean model lets us assert exact
    state-dict equality key-by-key.
    """
    return nn.Sequential(
        nn.Linear(8, 16),
        nn.LayerNorm(16),
        nn.Linear(16, 4),
    )


def _tiny_config() -> Config:
    """A minimal valid Config used as the ``config`` field at save time.

    The values are arbitrary; the test only cares that the same Config
    instance round-trips through save and load unchanged.
    """
    return Config(
        model=ModelConfig(base_model_id="sshleifer/tiny-gpt2", dtype="float32"),
        data=DataConfig(sources=["gsm8k"], val_split_pct=10.0, seed=7),
        training=TrainingConfig(
            max_steps=10, lr=1e-4, warmup_steps=2, per_device_batch_size=2
        ),
    )


def _capture_rng_states() -> dict:
    """Snapshot every RNG the trainer touches.

    The trainer (issue 05) is responsible for calling this kind of
    helper before saving and applying the saved values after loading.
    Tests need the same snapshot to assert round-trip identity, so we
    inline the same dict format here rather than exposing a public
    helper from the module — checkpoint.py's surface stays narrow.
    """
    return {
        "torch": torch.get_rng_state(),
        "torch_cuda": (
            torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []
        ),
        "numpy": np.random.get_state(),
        "python": random.getstate(),
    }


# -----------------------------------------------------------------------------
# Criterion 2: round-trip identity
# -----------------------------------------------------------------------------


def test_save_then_load_round_trips_all_state(tmp_path: Path) -> None:
    """Every field saved is recovered byte-identically on load."""
    model = _tiny_model()
    optimizer = build_optimizer(model, lr=1e-3, weight_decay=0.1)
    scheduler = build_lr_scheduler(optimizer, total_steps=100, warmup_steps=10)
    # Drive a few optimizer + scheduler steps so the saved state isn't
    # the default zeroed initial state — that would let a save/load
    # implementation that simply rebuilt fresh objects pass the test.
    for _ in range(3):
        for p in model.parameters():
            p.grad = torch.zeros_like(p)
        optimizer.step()
        scheduler.step()

    rng_states = _capture_rng_states()
    config = _tiny_config()

    saved_path = save_checkpoint(
        directory=tmp_path,
        step=42,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        rng_states=rng_states,
        config=config,
    )
    state = load_checkpoint(saved_path)

    assert isinstance(state, CheckpointState)
    assert state.step == 42
    assert state.config == config

    # Model state dict: every tensor matches by torch.equal.
    saved_sd = model.state_dict()
    assert set(state.model_state_dict.keys()) == set(saved_sd.keys())
    for key, tensor in saved_sd.items():
        assert torch.equal(state.model_state_dict[key], tensor), key

    # Optimizer state dict: dict-equal after structure check. The state
    # dict is a nested dict of ints -> dict of param-state tensors plus
    # a ``param_groups`` list of plain Python values.
    expected_opt_sd = optimizer.state_dict()
    assert state.optimizer_state_dict["param_groups"] == expected_opt_sd["param_groups"]
    assert set(state.optimizer_state_dict["state"].keys()) == set(
        expected_opt_sd["state"].keys()
    )
    for pid, pstate in expected_opt_sd["state"].items():
        for k, v in pstate.items():
            loaded = state.optimizer_state_dict["state"][pid][k]
            if isinstance(v, torch.Tensor):
                assert torch.equal(loaded, v), (pid, k)
            else:
                assert loaded == v, (pid, k)

    # Scheduler state dict round-trips by plain equality (LambdaLR's
    # state is a small dict of Python ints/floats; ``base_lrs`` is a
    # list of floats).
    expected_sched_sd = scheduler.state_dict()
    # Drop any non-serializable closure entry; LambdaLR stores
    # ``lr_lambdas`` as a list whose elements may be None when the
    # lambda was a function (not picklable). Compare what's present.
    for key, value in expected_sched_sd.items():
        if key == "lr_lambdas":
            # LambdaLR replaces unpicklable lambdas with None on
            # state_dict(); presence is enough.
            assert key in state.scheduler_state_dict
            continue
        assert state.scheduler_state_dict[key] == value, key

    # RNG states: torch is a tensor, numpy is a tuple containing arrays,
    # python is a tuple of opaque ints. Equality requires per-type care.
    assert torch.equal(state.rng_states["torch"], rng_states["torch"])
    # numpy.random.get_state() returns a tuple where index 1 is an
    # ndarray of the Mersenne Twister keys; np.array_equal handles it.
    expected_np = rng_states["numpy"]
    loaded_np = state.rng_states["numpy"]
    assert expected_np[0] == loaded_np[0]
    assert np.array_equal(expected_np[1], loaded_np[1])
    assert expected_np[2:] == loaded_np[2:]
    assert state.rng_states["python"] == rng_states["python"]
    assert state.rng_states["torch_cuda"] == rng_states["torch_cuda"]


# -----------------------------------------------------------------------------
# Criterion 3: atomic write — mid-write failure leaves only the .tmp directory
# -----------------------------------------------------------------------------


def test_failure_during_model_write_leaves_only_tmp_dir(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Mid-write crash never produces a final-named directory."""
    model = _tiny_model()
    optimizer = build_optimizer(model, lr=1e-3, weight_decay=0.1)
    scheduler = build_lr_scheduler(optimizer, total_steps=100, warmup_steps=10)
    rng_states = _capture_rng_states()
    config = _tiny_config()

    # Patch the symbol the way checkpoint.py looks it up: it imports
    # ``safetensors.torch`` as a module and calls ``save_model`` through
    # that module reference. Patching the module attribute makes the
    # patch take effect inside ``save_checkpoint``.
    from finpost.training import checkpoint as checkpoint_module

    def _explode(*_args, **_kwargs):
        raise RuntimeError("simulated mid-write failure")

    monkeypatch.setattr(
        checkpoint_module.safetensors.torch, "save_model", _explode
    )

    with pytest.raises(RuntimeError, match="simulated mid-write failure"):
        save_checkpoint(
            directory=tmp_path,
            step=42,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            rng_states=rng_states,
            config=config,
        )

    final_dir = tmp_path / "step-00000042"
    tmp_dir = tmp_path / "step-00000042.tmp"
    assert not final_dir.exists(), "final-named directory must not exist after failure"
    assert tmp_dir.exists(), "tmp directory must remain so caller can detect/clean"
    # And the tmp directory is cleanable: no zombie file locks etc.
    import shutil

    shutil.rmtree(tmp_dir)
    assert not tmp_dir.exists()


# -----------------------------------------------------------------------------
# Criterion 4: retention policy — keep last_n + best_so_far, delete the rest
# -----------------------------------------------------------------------------


def test_retention_policy_keeps_last_n_plus_best_so_far(tmp_path: Path) -> None:
    """Five mock checkpoints, last_n=2, best=step-300 → keep 300/400/500."""
    # Build mock checkpoint dirs. They don't need real contents for the
    # retention policy — it only inspects directory names.
    steps = [100, 200, 300, 400, 500]
    paths = {step: tmp_path / f"step-{step:08d}" for step in steps}
    for p in paths.values():
        p.mkdir()
        (p / "marker.txt").write_text("placeholder")

    apply_retention_policy(
        directory=tmp_path,
        last_n=2,
        best_so_far=paths[300],
    )

    surviving = {p.name for p in tmp_path.iterdir() if p.is_dir()}
    assert surviving == {"step-00000300", "step-00000400", "step-00000500"}


def test_retention_policy_with_no_best_keeps_only_last_n(tmp_path: Path) -> None:
    """``best_so_far=None`` is a clean no-op: only last_n survives."""
    steps = [100, 200, 300, 400, 500]
    for step in steps:
        (tmp_path / f"step-{step:08d}").mkdir()

    apply_retention_policy(directory=tmp_path, last_n=2, best_so_far=None)

    surviving = {p.name for p in tmp_path.iterdir() if p.is_dir()}
    assert surviving == {"step-00000400", "step-00000500"}


def test_retention_policy_ignores_non_step_directories(tmp_path: Path) -> None:
    """Foreign directories (e.g. ``logs/``) are left alone."""
    (tmp_path / "step-00000100").mkdir()
    (tmp_path / "step-00000200").mkdir()
    (tmp_path / "logs").mkdir()
    (tmp_path / "logs" / "stuff.txt").write_text("hello")

    apply_retention_policy(directory=tmp_path, last_n=1, best_so_far=None)

    surviving = {p.name for p in tmp_path.iterdir() if p.is_dir()}
    assert surviving == {"step-00000200", "logs"}
    # And the unrelated content inside ``logs/`` is intact.
    assert (tmp_path / "logs" / "stuff.txt").read_text() == "hello"


# -----------------------------------------------------------------------------
# Criterion 5: resume produces identical trajectory
# -----------------------------------------------------------------------------


def _build_tiny_gpt2(seed: int):
    """Construct tiny-gpt2 with all dropout disabled.

    Dropout-off matters for this test because we need step k of run A
    (steps 6..10 with no interruption) to produce the same loss as step
    k of run B (load checkpoint, then steps 6..10). Dropout is stochastic
    per forward; with it on, getting bit-identical losses across A and B
    requires capturing and restoring the dropout RNG at exactly the right
    moments. Disabling dropout sidesteps all of that — the model is still
    trainable; we're just removing the only source of nondeterminism in
    the forward pass that would force tighter RNG bookkeeping in this
    test. The trainer (issue 05) handles RNG capture/restore for the
    real run; this test isolates the checkpoint round-trip.
    """
    from transformers import AutoModelForCausalLM

    torch.manual_seed(seed)
    model = AutoModelForCausalLM.from_pretrained(
        "sshleifer/tiny-gpt2",
        attn_pdrop=0.0,
        embd_pdrop=0.0,
        resid_pdrop=0.0,
    )
    model.train()
    return model


def _synthetic_batch(vocab_size: int, seed: int):
    """Deterministic synthetic ``(input_ids, labels)`` for a fixed batch.

    Reused across A and B so the input data isn't a confound — both
    trajectories see the exact same tensors at every step. We don't
    pull from real GSM8K/MATH because that drags in tokenizer + loader
    machinery that's irrelevant to checkpointing determinism.
    """
    g = torch.Generator().manual_seed(seed)
    input_ids = torch.randint(0, vocab_size, (2, 16), generator=g)
    labels = input_ids.clone()
    return input_ids, labels


def test_resume_produces_identical_loss_trajectory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A 10-step run and a 5+resume+5 run match on losses 6..10."""
    # Force the offline path. The model is already cached at
    # ~/.cache/huggingface/hub/; without these, the loader can stall on
    # network probes in CI.
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("HF_DATASETS_OFFLINE", "1")

    from finpost.training.sft import train_step

    SEED = 1234
    VOCAB = 50257  # tiny-gpt2 GPT-2 vocab size

    # Trajectory A: uninterrupted 10 steps. We deliberately drive the
    # scheduler in lockstep with the optimizer (one ``sched.step()`` per
    # training step). This is the realistic shape of the trainer's loop
    # and the exact pattern checkpoint resume must preserve: at the
    # moment of save, the scheduler's ``last_epoch`` is whatever it has
    # been advanced to, and on resume the loaded scheduler must continue
    # from there (NOT restart at 0, which would zero the LR via the
    # warmup branch and silently no-op the next optimizer step).
    torch.manual_seed(SEED)
    model_a = _build_tiny_gpt2(seed=SEED)
    opt_a = build_optimizer(model_a, lr=1e-3, weight_decay=0.0)
    sched_a = build_lr_scheduler(opt_a, total_steps=10, warmup_steps=1)
    losses_a: list[float] = []
    for step in range(10):
        input_ids, labels = _synthetic_batch(VOCAB, seed=SEED + step)
        losses_a.append(train_step(model_a, input_ids, labels, opt_a))
        sched_a.step()

    # Trajectory B: 5 steps, save, fresh model + load, 5 more.
    torch.manual_seed(SEED)
    model_b = _build_tiny_gpt2(seed=SEED)
    opt_b = build_optimizer(model_b, lr=1e-3, weight_decay=0.0)
    sched_b = build_lr_scheduler(opt_b, total_steps=10, warmup_steps=1)
    losses_b: list[float] = []
    for step in range(5):
        input_ids, labels = _synthetic_batch(VOCAB, seed=SEED + step)
        losses_b.append(train_step(model_b, input_ids, labels, opt_b))
        sched_b.step()

    rng_states = _capture_rng_states()
    saved_path = save_checkpoint(
        directory=tmp_path,
        step=5,
        model=model_b,
        optimizer=opt_b,
        scheduler=sched_b,
        rng_states=rng_states,
        config=_tiny_config(),
    )

    # "Fresh process" simulated: throw away every live training object
    # and rebuild from scratch. ``del`` + reassign lets the same names
    # be reused so the second half reads naturally.
    del model_b, opt_b, sched_b

    model_b = _build_tiny_gpt2(seed=SEED + 999)  # different seed: forces load to overwrite
    opt_b = build_optimizer(model_b, lr=1e-3, weight_decay=0.0)
    sched_b = build_lr_scheduler(opt_b, total_steps=10, warmup_steps=1)

    state = load_checkpoint(saved_path)
    # Apply the loaded state. ``strict=False`` is required because
    # ``safetensors.torch.save_model`` drops one tensor of any tied pair
    # (GPT-2 ties wte<->lm_head); the dropped key shows up as
    # ``missing`` and is harmless because the fresh model has the tying
    # already in place.
    model_b.load_state_dict(state.model_state_dict, strict=False)
    opt_b.load_state_dict(state.optimizer_state_dict)
    sched_b.load_state_dict(state.scheduler_state_dict)
    torch.set_rng_state(state.rng_states["torch"])
    if torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state.rng_states["torch_cuda"])
    np.random.set_state(state.rng_states["numpy"])
    random.setstate(state.rng_states["python"])

    for step in range(5, 10):
        input_ids, labels = _synthetic_batch(VOCAB, seed=SEED + step)
        losses_b.append(train_step(model_b, input_ids, labels, opt_b))
        sched_b.step()

    # Steps 6..10 (indices 5..9) must match. Steps 1..5 are not asserted
    # — they're identical by construction (same code, same seeds).
    for i in range(5, 10):
        assert abs(losses_a[i] - losses_b[i]) < 1e-5, (
            f"step {i}: A={losses_a[i]} B={losses_b[i]}"
        )


# -----------------------------------------------------------------------------
# CLI helper: ``python -m finpost.training.checkpoint --inspect <path>``
# -----------------------------------------------------------------------------


def test_inspect_cli_prints_step_config_and_tensor_shapes(tmp_path: Path) -> None:
    """Smoke-check the debugging affordance: step, config, shape summary."""
    model = _tiny_model()
    optimizer = build_optimizer(model, lr=1e-3, weight_decay=0.1)
    scheduler = build_lr_scheduler(optimizer, total_steps=100, warmup_steps=10)
    rng_states = _capture_rng_states()
    config = _tiny_config()

    saved_path = save_checkpoint(
        directory=tmp_path,
        step=42,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        rng_states=rng_states,
        config=config,
    )

    # Subprocess so we exercise the actual ``__main__`` entry point,
    # not just the inspect function in isolation. Using sys.executable
    # ensures we hit the same Python the test runner is using.
    result = subprocess.run(
        [sys.executable, "-m", "finpost.training.checkpoint", "--inspect", str(saved_path)],
        capture_output=True,
        text=True,
        check=True,
        env={**os.environ, "PYTHONIOENCODING": "utf-8"},
    )
    out = result.stdout

    # Step appears in some recognizable form.
    assert "42" in out
    # The base model id from config appears (tells the user what was
    # being trained).
    assert "sshleifer/tiny-gpt2" in out
    # The first ``Linear(8, 16)`` weight is shape (16, 8); the inspect
    # helper renders shapes as plain tuples, so the exact substring is
    # what we look for.
    assert "(16, 8)" in out
