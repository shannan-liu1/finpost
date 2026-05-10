"""Tests for the end-to-end SFT Trainer.

Each test pins one of the issue-05 acceptance criteria:

    1. Smoke / soft launch        — full loop runs, every required key
                                    is logged, a checkpoint exists.
    2. wandb logging keys         — train/loss, train/lr, train/grad_norm,
                                    val/loss all appear in stubbed log.
    3. Determinism                — two ``Trainer(config).train()`` calls
                                    on the same config produce identical
                                    loss curves within atol=1e-5.
    4. Resume continuity          — train 20 steps end-to-end vs.
                                    train 10 → save → fresh resume → 10.
                                    Steps 11..20 match within atol=1e-5.
    5. Gradient-accumulation
       correctness                — grad_accum=N, batch=B/N matches
                                    grad_accum=1, batch=B within atol=1e-3.

All tests run on tiny-gpt2 with dropout disabled and a synthetic
DataLoader. The real ``make_loaders`` (which loads GSM8K) is patched
out — too slow for unit tests and would require network on first run.
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

from finpost.training import trainer as trainer_module
from finpost.training.checkpoint import load_checkpoint
from finpost.training.config import (
    CheckpointConfig,
    Config,
    DataConfig,
    LoggingConfig,
    ModelConfig,
    PackingConfig,
    TrainingConfig,
)
from finpost.training.trainer import Trainer

# tiny-gpt2: ~1MB GPT-2 with reduced layers/dim. The same model the
# checkpoint and smoke tests already use; it's cached by the dev env so
# loading is instant once the package is installed.
_TINY_MODEL = "sshleifer/tiny-gpt2"
# tiny-gpt2's vocab matches the full GPT-2 BPE vocabulary.
_VOCAB_SIZE = 50257
# tiny-gpt2's max position embeddings cap.
_MAX_POS = 1024


# -----------------------------------------------------------------------------
# Shared fixtures and helpers
# -----------------------------------------------------------------------------


def _make_config(
    *,
    tmp_path: Path,
    max_steps: int = 4,
    warmup_steps: int = 1,
    grad_accum_steps: int = 1,
    per_device_batch_size: int = 2,
    val_every_n_steps: int = 2,
    checkpoint_every_n_steps: int = 2,
    seed: int = 0,
    resume_from: Path | None = None,
) -> Config:
    """Build a Config tailored for fast trainer unit tests.

    Defaults:
      - tiny-gpt2 in float32 (CPU-friendly; bf16 CPU is slow and
        introduces dtype noise that breaks determinism asserts).
      - use_safetensors=False because tiny-gpt2 ships as legacy .bin
        only. The real Phase 1 Qwen model has safetensors and we use
        them; this is a tiny-gpt2-only quirk.
      - Tiny max_seq_len so the synthetic batches the test builds fit
        well under any model position cap.
      - All cadences (val, checkpoint) parameterised so individual
        tests can opt into "never trigger this" via large numbers.
    """
    return Config(
        model=ModelConfig(
            base_model_id=_TINY_MODEL,
            dtype="float32",
            use_safetensors=False,
        ),
        data=DataConfig(
            sources=["gsm8k"],
            val_split_pct=0.0,
            seed=seed,
        ),
        training=TrainingConfig(
            max_steps=max_steps,
            warmup_steps=warmup_steps,
            lr=1e-4,
            grad_accum_steps=grad_accum_steps,
            per_device_batch_size=per_device_batch_size,
            val_every_n_steps=val_every_n_steps,
            checkpoint_every_n_steps=checkpoint_every_n_steps,
        ),
        packing=PackingConfig(max_seq_len=64, isolate_documents=True),
        logging=LoggingConfig(wandb_project="finpost-tests", run_name="test"),
        checkpointing=CheckpointConfig(
            save_dir=tmp_path / "checkpoints",
            retention_last_n=3,
            resume_from=resume_from,
        ),
    )


def _causal_attention_mask(batch_size: int, seq_len: int) -> torch.Tensor:
    """4D causal mask matching ``PackingCollator`` output shape and dtype.

    Real collator output is (B, 1, S, S) int64 with the lower-triangular
    causal pattern AND document-isolation pattern baked in. For the
    test loader we have one document per row, so the mask is just the
    causal triangle. dtype=int64 mirrors the collator exactly so the
    Trainer's ``.bool()`` cast is exercised.
    """
    causal = torch.tril(torch.ones((seq_len, seq_len), dtype=torch.long))
    return causal.unsqueeze(0).unsqueeze(0).expand(batch_size, 1, seq_len, seq_len).contiguous()


def _synthetic_batch(
    *,
    batch_size: int,
    seq_len: int,
    prompt_len: int,
    seed: int,
) -> dict[str, Any]:
    """One synthetic packed batch, in the exact shape the collator emits.

    We construct the batch by hand instead of going through the real
    collator because:
      - the real collator requires a tokenizer + dataset, both of which
        drag in network access on first run;
      - tests need bit-deterministic input across runs, which is easier
        to guarantee when we own the random seed directly.
    """
    g = torch.Generator().manual_seed(seed)
    input_ids = torch.randint(0, _VOCAB_SIZE, (batch_size, seq_len), generator=g)

    # labels = input_ids with the prompt portion masked. Mirrors what
    # ``mask_prompt_tokens`` does inside the real collator.
    labels = input_ids.clone()
    labels[:, :prompt_len] = -100

    # position_ids: 0..seq_len-1, restarting at 0 per row. With one
    # document per row, this is just arange(seq_len) broadcast.
    base = torch.arange(seq_len, dtype=torch.long).unsqueeze(0)
    position_ids = base.expand(batch_size, seq_len).contiguous()

    boundary = {
        "start": 0,
        "end": seq_len,
        "prompt_length": prompt_len,
        "source": "gsm8k",
        "example_id": None,
    }
    return {
        "input_ids": input_ids,
        "labels": labels,
        "attention_mask": _causal_attention_mask(batch_size, seq_len),
        "position_ids": position_ids,
        # ``document_boundaries`` is a list of dicts per row in the real
        # collator; the trainer never reads it, but ship the right
        # shape for completeness.
        "document_boundaries": [[boundary] for _ in range(batch_size)],
    }


class _SyntheticDataset(torch.utils.data.Dataset):
    """Indexed wrapper so we can plug into ``DataLoader`` without shuffling.

    Returns each pre-built batch dict at index i. With ``batch_size=1``
    on the DataLoader and a custom ``collate_fn`` that just unwraps the
    single-element list, each iteration step yields exactly one batch
    in the collator's emit shape.
    """

    def __init__(self, batches: list[dict[str, Any]]) -> None:
        self.batches = batches

    def __len__(self) -> int:
        return len(self.batches)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        return self.batches[idx]


def _identity_collate(samples: list[dict[str, Any]]) -> dict[str, Any]:
    """Return the single sample as-is. Pre-batched by ``_synthetic_batch``."""
    # DataLoader always wraps in a list; we sized batch=1 on the loader
    # so this list is length 1. Returning ``samples[0]`` hands the
    # already-shaped batch dict straight through.
    assert len(samples) == 1
    return samples[0]


def _make_synthetic_loaders(
    *,
    train_batches: list[dict[str, Any]],
    val_batches: list[dict[str, Any]],
) -> tuple[DataLoader, DataLoader]:
    """Build a (train, val) loader pair from explicit per-step batches.

    Lets each test specify exactly which batch the trainer sees at
    each step, which is what makes determinism / resume / grad-accum
    asserts possible.
    """
    train = DataLoader(
        _SyntheticDataset(train_batches),
        batch_size=1,
        shuffle=False,
        collate_fn=_identity_collate,
    )
    val = DataLoader(
        _SyntheticDataset(val_batches),
        batch_size=1,
        shuffle=False,
        collate_fn=_identity_collate,
    )
    return train, val


def _patch_loaders(
    monkeypatch: pytest.MonkeyPatch,
    *,
    train_batches: list[dict[str, Any]],
    val_batches: list[dict[str, Any]],
) -> None:
    """Replace ``trainer_module.make_loaders`` with the synthetic builder.

    Patch target is the ``make_loaders`` symbol bound INSIDE the trainer
    module — that's the reference the Trainer actually calls. Patching
    ``finpost.training.dataset.make_loaders`` would not take effect
    because the trainer imports it ``from ... import make_loaders``.
    """

    def fake(_config, _tokenizer):
        return _make_synthetic_loaders(
            train_batches=train_batches,
            val_batches=val_batches,
        )

    monkeypatch.setattr(trainer_module, "make_loaders", fake)


def _patch_tiny_model_load(monkeypatch: pytest.MonkeyPatch) -> None:
    """Force tiny-gpt2 to load with all dropout disabled.

    tiny-gpt2 has attn_pdrop / embd_pdrop / resid_pdrop > 0 by default.
    Dropout is the only source of nondeterminism in the forward pass on
    CPU; tests that assert bit-identical loss curves break with it on.
    The checkpoint test uses the same trick — see
    ``tests/test_checkpoint.py::_build_tiny_gpt2``.

    We patch ``AutoModelForCausalLM.from_pretrained`` at the trainer's
    binding so dropout config gets injected on every model load
    triggered by ``Trainer._setup``.
    """
    original = trainer_module.AutoModelForCausalLM.from_pretrained

    def from_pretrained_no_dropout(name: str, **kwargs):
        kwargs.setdefault("attn_pdrop", 0.0)
        kwargs.setdefault("embd_pdrop", 0.0)
        kwargs.setdefault("resid_pdrop", 0.0)
        return original(name, **kwargs)

    monkeypatch.setattr(
        trainer_module.AutoModelForCausalLM,
        "from_pretrained",
        from_pretrained_no_dropout,
    )


def _disable_wandb(monkeypatch: pytest.MonkeyPatch) -> list[dict[str, Any]]:
    """Stub wandb.init/log/finish; return a list that captures log calls.

    Returning the list as the test handle keeps each test free to
    inspect what was logged without holding a reference to the patched
    object. ``WANDB_MODE=disabled`` is also set as a belt for the
    real wandb client should the stub miss anything.

    Calling this multiple times in the same test is intentional and
    safe: each call re-stubs the same module attributes (last-write-
    wins until pytest tears down the monkeypatch fixture), and returns
    a fresh list — which is exactly what tests that simulate "two
    separate processes" need.
    """
    monkeypatch.setenv("WANDB_MODE", "disabled")
    log_calls: list[dict[str, Any]] = []

    def fake_init(**_kwargs):
        # Return a sentinel object; nothing in the trainer dereferences
        # the return value of init beyond the global wandb namespace.
        return object()

    def fake_log(data, *_args, **_kwargs):
        log_calls.append(data)

    def fake_finish(*_args, **_kwargs):
        return None

    monkeypatch.setattr(trainer_module.wandb, "init", fake_init)
    monkeypatch.setattr(trainer_module.wandb, "log", fake_log)
    monkeypatch.setattr(trainer_module.wandb, "finish", fake_finish)
    return log_calls


def _three_train_batches() -> list[dict[str, Any]]:
    """A reusable list of 3 deterministic micro-batches.

    The Trainer's loop cycles the iterator when it exhausts, so 3 is
    plenty even for max_steps=20 — the loader simply restarts. We use
    distinct seeds per batch so successive batches contain different
    tokens (otherwise gradient direction is identical every step and
    the loss curve is uninteresting).
    """
    return [
        _synthetic_batch(batch_size=2, seq_len=16, prompt_len=4, seed=100),
        _synthetic_batch(batch_size=2, seq_len=16, prompt_len=4, seed=101),
        _synthetic_batch(batch_size=2, seq_len=16, prompt_len=4, seed=102),
    ]


def _two_val_batches() -> list[dict[str, Any]]:
    """Two deterministic val batches; small enough for fast eval."""
    return [
        _synthetic_batch(batch_size=2, seq_len=16, prompt_len=4, seed=200),
        _synthetic_batch(batch_size=2, seq_len=16, prompt_len=4, seed=201),
    ]


# -----------------------------------------------------------------------------
# Criterion 6: smoke / soft launch
# -----------------------------------------------------------------------------


def test_smoke_run_emits_required_metrics_and_writes_checkpoint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A 20-step run produces every required log key and a checkpoint on disk.

    Mirrors criterion 6 of the issue: "a 20-step ``sshleifer/tiny-gpt2``
    run emits loss metrics, validation loss, checkpoint path, resume
    metadata, and offline tracking artifacts without requiring GPU or
    network during the run."

    Offline tracking is asserted via the ``WANDB_MODE=disabled`` env
    var and the stubbed wandb.* functions in ``_disable_wandb`` —
    nothing in the test makes a network call. GPU absence is implicit:
    the trainer device-selects to CPU when ``torch.cuda.is_available``
    is False, which is the case on the CI runner.
    """
    log_calls = _disable_wandb(monkeypatch)
    _patch_tiny_model_load(monkeypatch)
    _patch_loaders(
        monkeypatch,
        train_batches=_three_train_batches(),
        val_batches=_two_val_batches(),
    )

    # 20 optimizer steps, with val and checkpoint cadences chosen to
    # fire several times during the run so each log key is exercised.
    config = _make_config(
        tmp_path=tmp_path,
        max_steps=20,
        warmup_steps=2,
        val_every_n_steps=5,        # val at steps 5, 10, 15, 20
        checkpoint_every_n_steps=10,  # checkpoints at steps 10 and 20
    )

    Trainer(config).train()

    # Combine all logged dicts into a single key set; we don't care
    # which step each key landed at, only that each key appeared at
    # least once during the run.
    logged_keys: set[str] = set()
    for call in log_calls:
        logged_keys.update(call.keys())

    # The four headline metrics required by the issue. ``train/*`` get
    # logged every optimizer step; ``val/loss`` only at val cadence.
    assert "train/loss" in logged_keys
    assert "train/lr" in logged_keys
    assert "train/grad_norm" in logged_keys
    assert "val/loss" in logged_keys

    # Checkpoint cadence is every 10 steps and max_steps=20 →
    # checkpoints at steps 10 and 20. Both should survive retention
    # (last_n=3 leaves room).
    save_dir = tmp_path / "checkpoints"
    surviving = sorted(p.name for p in save_dir.iterdir() if p.is_dir())
    assert "step-00000010" in surviving
    assert "step-00000020" in surviving

    # Train-loss curve length is one entry per optimizer step.
    train_losses = _capture_train_losses_from_log(log_calls)
    assert len(train_losses) == 20

    # Resume metadata sanity check: the saved checkpoint can be loaded
    # back and contains the step counter, the validated config, and a
    # non-empty model state dict. This is the metadata an operator
    # would consult to confirm "yes, I can resume from here".
    final_state = load_checkpoint(save_dir / "step-00000020")
    assert final_state.step == 20
    assert final_state.config.training.max_steps == 20
    assert len(final_state.model_state_dict) > 0


# -----------------------------------------------------------------------------
# Criterion 4: wandb logging — every required key appears at least once
# -----------------------------------------------------------------------------


def test_wandb_log_includes_each_required_metric_key(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Stubbed wandb.log is called with each required metric key."""
    log_calls = _disable_wandb(monkeypatch)
    _patch_tiny_model_load(monkeypatch)
    _patch_loaders(
        monkeypatch,
        train_batches=_three_train_batches(),
        val_batches=_two_val_batches(),
    )

    # 6 steps so the val cadence (every 3) fires at least once and we
    # verify the val/loss key shows up in addition to the train ones.
    config = _make_config(
        tmp_path=tmp_path,
        max_steps=6,
        warmup_steps=1,
        val_every_n_steps=3,
        checkpoint_every_n_steps=10,  # large enough to never fire mid-run
    )

    Trainer(config).train()

    # Walk every recorded log call and unify their keys. The trainer
    # may issue separate calls for the train metrics dict and the
    # val/loss dict; that's fine — the spec only requires each key
    # appear, not that they coexist in one call.
    seen_keys: set[str] = set()
    for call in log_calls:
        seen_keys.update(call.keys())

    for required in ("train/loss", "train/lr", "train/grad_norm", "val/loss"):
        assert required in seen_keys, f"missing required wandb log key: {required}"


# -----------------------------------------------------------------------------
# Criterion 2: determinism — two ``Trainer(config).train()`` calls match
# -----------------------------------------------------------------------------


def _capture_train_losses_from_log(log_calls: list[dict[str, Any]]) -> list[float]:
    """Pull the per-step ``train/loss`` floats out of the wandb log list.

    The trainer logs train metrics in one dict per optimizer step; we
    extract just the loss field in step order. The list itself is
    already in chronological order because the stub appends.
    """
    losses: list[float] = []
    for call in log_calls:
        if "train/loss" in call:
            losses.append(float(call["train/loss"]))
    return losses


def test_two_trainer_runs_with_same_config_match_loss_curve(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Same config, fresh Trainer twice → bit-identical loss curve."""
    _patch_tiny_model_load(monkeypatch)

    train_batches = _three_train_batches()
    val_batches = _two_val_batches()

    log_calls_a = _disable_wandb(monkeypatch)
    _patch_loaders(monkeypatch, train_batches=train_batches, val_batches=val_batches)
    config_a = _make_config(
        tmp_path=tmp_path / "run_a",
        max_steps=20,
        warmup_steps=2,
        val_every_n_steps=100,    # never triggers
        checkpoint_every_n_steps=100,  # never triggers mid-run
        seed=42,
    )
    Trainer(config_a).train()
    losses_a = _capture_train_losses_from_log(log_calls_a)

    # Re-stub wandb so the second run gets a fresh log list. Loaders
    # are re-patched too because monkeypatch undoes per-call overrides
    # only at fixture teardown — but the SECOND _disable_wandb does
    # overwrite the first stub since it's the same attribute.
    log_calls_b = _disable_wandb(monkeypatch)
    _patch_loaders(monkeypatch, train_batches=train_batches, val_batches=val_batches)
    config_b = _make_config(
        tmp_path=tmp_path / "run_b",
        max_steps=20,
        warmup_steps=2,
        val_every_n_steps=100,
        checkpoint_every_n_steps=100,
        seed=42,
    )
    Trainer(config_b).train()
    losses_b = _capture_train_losses_from_log(log_calls_b)

    assert len(losses_a) == 20
    assert len(losses_b) == 20
    for i, (a, b) in enumerate(zip(losses_a, losses_b, strict=True)):
        assert abs(a - b) < 1e-5, f"step {i}: a={a} b={b}"


# -----------------------------------------------------------------------------
# Criterion 3: resume continuity — A run vs split B run match on overlap window
# -----------------------------------------------------------------------------


def test_resume_from_checkpoint_matches_uninterrupted_loss_trajectory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """20-step run vs (10 + save + fresh + 10) → losses 11..20 match.

    Implementation notes:

    1. Run A trains 20 steps end-to-end with a checkpoint saved at step
       10 (cadence=10). Run B is a fresh Trainer that ``resume_from``s
       A's step-10 checkpoint and trains to step 20. We compare losses
       11..20.

    2. The synthetic loader is built with 20 distinct batches (one per
       step) so neither run cycles through the same batch twice. Run B
       is given just the LAST 10 batches (indices 10..19). This sidesteps
       a known limitation of single-process resume: the DataLoader
       iterator state is not part of the checkpoint, so a fresh
       ``iter(train_loader)`` starts at batch 0 regardless of where
       training left off. By feeding run B exactly the batches it
       SHOULD see at steps 11..20, the test isolates the resume
       mechanism from loader-state restoration (which is out of scope
       for this issue and a known feature gap).

    3. Cosine schedule horizon: both runs configure ``max_steps=20``,
       so the LR at any step 11..20 is identical between the two.
    """
    _patch_tiny_model_load(monkeypatch)

    # 20 distinct batches — one per training step. Distinct seeds keep
    # successive batches genuinely different so the loss curve actually
    # moves rather than oscillating around a fixed-input minimum.
    all_batches = [
        _synthetic_batch(batch_size=2, seq_len=16, prompt_len=4, seed=400 + i)
        for i in range(20)
    ]
    val_batches = _two_val_batches()

    # Run A: 20 uninterrupted steps with a checkpoint at step 10.
    log_calls_a = _disable_wandb(monkeypatch)
    _patch_loaders(monkeypatch, train_batches=all_batches, val_batches=val_batches)
    save_dir_a = tmp_path / "run_a" / "checkpoints"
    config_a = _make_config(
        tmp_path=tmp_path / "run_a",
        max_steps=20,
        warmup_steps=2,
        val_every_n_steps=100,
        checkpoint_every_n_steps=10,  # save at step 10 (and final at exit)
        seed=42,
    )
    Trainer(config_a).train()
    losses_a = _capture_train_losses_from_log(log_calls_a)
    assert len(losses_a) == 20, f"run A produced {len(losses_a)} losses, expected 20"

    checkpoint_path = save_dir_a / "step-00000010"
    assert checkpoint_path.is_dir(), f"missing step-10 checkpoint at {checkpoint_path}"

    # Run B: feed only batches 10..19 so the loader gives the SAME data
    # run A saw at steps 11..20. See note (2) above for why.
    log_calls_b = _disable_wandb(monkeypatch)
    _patch_loaders(monkeypatch, train_batches=all_batches[10:], val_batches=val_batches)
    config_b = _make_config(
        tmp_path=tmp_path / "run_b",
        max_steps=20,
        warmup_steps=2,
        val_every_n_steps=100,
        checkpoint_every_n_steps=100,  # do not save mid-run
        seed=42,
        resume_from=checkpoint_path,
    )
    Trainer(config_b).train()
    losses_b = _capture_train_losses_from_log(log_calls_b)

    # Resume should produce 10 loss values (steps 11..20).
    assert len(losses_b) == 10, f"expected 10 post-resume losses, got {len(losses_b)}"

    # The acceptance criterion: steps 11..20 of A match the resumed run.
    for i in range(10):
        assert abs(losses_a[10 + i] - losses_b[i]) < 1e-5, (
            f"step {11 + i}: a={losses_a[10 + i]} b={losses_b[i]}"
        )


# -----------------------------------------------------------------------------
# Criterion 5: gradient accumulation correctness
# -----------------------------------------------------------------------------


def test_gradient_accumulation_matches_full_batch_loss_curve(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """grad_accum=2 + batch=2 ≈ grad_accum=1 + batch=4 within atol=1e-3.

    Setup: split the same 4-row tensor two ways. Run "full" sees one
    4-row batch per step. Run "accum" sees two 2-row sub-batches per
    optimizer step. The resulting effective gradient has the same
    expectation; the per-step LOSSES (averaged inside the trainer) are
    not bit-identical because mean-of-mean ≠ mean over different batch
    splits, but they match to ~1e-3 on a small tiny-gpt2.
    """
    _patch_tiny_model_load(monkeypatch)

    # Build a per-step pool of 4-row batches. The "accum" run will
    # split each into two 2-row halves so the SAME tokens drive both
    # runs in the same order — that's what keeps the loss curves
    # close.
    big_batches = [
        _synthetic_batch(batch_size=4, seq_len=16, prompt_len=4, seed=300),
        _synthetic_batch(batch_size=4, seq_len=16, prompt_len=4, seed=301),
        _synthetic_batch(batch_size=4, seq_len=16, prompt_len=4, seed=302),
        _synthetic_batch(batch_size=4, seq_len=16, prompt_len=4, seed=303),
    ]

    def split_batch(batch: dict[str, Any]) -> list[dict[str, Any]]:
        # Slice each tensor down the batch axis. document_boundaries
        # is per-row, so it splits identically.
        return [
            {
                "input_ids": batch["input_ids"][i : i + 2],
                "labels": batch["labels"][i : i + 2],
                "attention_mask": batch["attention_mask"][i : i + 2],
                "position_ids": batch["position_ids"][i : i + 2],
                "document_boundaries": batch["document_boundaries"][i : i + 2],
            }
            for i in (0, 2)
        ]

    # Full run: one 4-row batch per optimizer step.
    log_calls_full = _disable_wandb(monkeypatch)
    _patch_loaders(monkeypatch, train_batches=big_batches, val_batches=_two_val_batches())
    config_full = _make_config(
        tmp_path=tmp_path / "full",
        max_steps=4,
        warmup_steps=1,
        grad_accum_steps=1,
        per_device_batch_size=4,
        val_every_n_steps=100,
        checkpoint_every_n_steps=100,
        seed=42,
    )
    Trainer(config_full).train()
    losses_full = _capture_train_losses_from_log(log_calls_full)

    # Accum run: same tokens, but each 4-row batch is delivered as two
    # 2-row micro-batches. grad_accum=2 means one optimizer step per
    # pair. Net effect: one optimizer step per 4 rows, same as full.
    accum_batches: list[dict[str, Any]] = []
    for big in big_batches:
        accum_batches.extend(split_batch(big))

    log_calls_accum = _disable_wandb(monkeypatch)
    _patch_loaders(monkeypatch, train_batches=accum_batches, val_batches=_two_val_batches())
    config_accum = _make_config(
        tmp_path=tmp_path / "accum",
        max_steps=4,
        warmup_steps=1,
        grad_accum_steps=2,
        per_device_batch_size=2,
        val_every_n_steps=100,
        checkpoint_every_n_steps=100,
        seed=42,
    )
    Trainer(config_accum).train()
    losses_accum = _capture_train_losses_from_log(log_calls_accum)

    assert len(losses_full) == 4
    assert len(losses_accum) == 4
    for i, (full, accum) in enumerate(zip(losses_full, losses_accum, strict=True)):
        assert abs(full - accum) < 1e-3, f"step {i}: full={full} accum={accum}"


# -----------------------------------------------------------------------------
# Defensive: validate() before train() raises
# -----------------------------------------------------------------------------


def test_validate_before_train_raises(tmp_path: Path) -> None:
    """Calling validate() on a Trainer that hasn't trained yet errors loudly."""
    config = _make_config(tmp_path=tmp_path, max_steps=2, warmup_steps=1)
    trainer = Trainer(config)
    with pytest.raises(RuntimeError, match="train"):
        trainer.validate()


# -----------------------------------------------------------------------------
# Defensive: rng helpers round-trip
# -----------------------------------------------------------------------------


def test_rng_capture_and_apply_round_trip() -> None:
    """``_capture_rng_states`` then ``_apply_rng_states`` restores draws.

    Sanity test on the helpers the trainer uses for resume. After
    capturing, drawing from each RNG, applying, and re-drawing, we
    should see the same sequence both times.
    """
    # Snapshot, then advance every RNG.
    state0 = trainer_module._capture_rng_states()

    drawn_torch = torch.rand(3)
    drawn_numpy = np.random.rand(3)
    drawn_python = [random.random() for _ in range(3)]

    # Restore and re-draw — we should see the SAME values.
    trainer_module._apply_rng_states(state0)
    again_torch = torch.rand(3)
    again_numpy = np.random.rand(3)
    again_python = [random.random() for _ in range(3)]

    assert torch.equal(drawn_torch, again_torch)
    assert np.allclose(drawn_numpy, again_numpy)
    assert drawn_python == again_python
