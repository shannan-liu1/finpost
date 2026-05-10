"""Atomic checkpoint save/load and bounded retention.

Three operations and a CLI affordance, kept small on purpose:

  ``save_checkpoint``       — write everything needed to resume a run
                              into ``directory / step-XXXXXXXX/``
                              atomically. Mid-write crashes leave a
                              ``.tmp`` suffix and never a half-written
                              final.
  ``load_checkpoint``       — read a checkpoint directory back into a
                              ``CheckpointState`` namedtuple. Pure
                              read, no global side effects: the caller
                              decides when to apply RNG / load the
                              state dicts into a model and optimizer.
  ``apply_retention_policy`` — keep the ``last_n`` newest plus an
                              optional ``best_so_far``; delete the
                              rest. Lets training run for a long time
                              without unbounded disk growth.

Atomicity recipe: write to ``<dir>.tmp/`` first, then ``os.replace``
to the final name. ``os.replace`` is the only rename primitive that's
atomic on both POSIX and Windows; ``os.rename`` differs across platforms
when the target exists. If the process dies between mkdir and replace,
the surviving ``.tmp`` is the marker that the previous attempt failed
and is safe to remove.

Why model weights live in ``model.safetensors`` and everything else in
``state.pt``: the ``safetensors`` format refuses to carry arbitrary
Python objects (no pickle), which is what makes it the supply-chain-
safe choice for weights — see ``SECURITY.md``. The optimizer state,
scheduler state, RNG snapshots, step counter, and Pydantic config are
all small Python structures whose author is the trainer itself, so
PyTorch's pickle-based ``torch.save`` is fine for them.
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import Any, NamedTuple

import safetensors.torch
import torch

from finpost.training.config import Config

# Subdirectory naming. Eight zero-padded digits because no Phase 1 run
# is going to need more than ~10^8 steps; padding makes alphabetical
# sort (which the filesystem gives us for free) match numerical order.
_STEP_PREFIX = "step-"
_STEP_DIGITS = 8

_MODEL_FILENAME = "model.safetensors"
_STATE_FILENAME = "state.pt"


class CheckpointState(NamedTuple):
    """Read-only result of ``load_checkpoint``.

    The trainer is responsible for applying these fields back onto live
    objects: ``model.load_state_dict(state.model_state_dict)``,
    ``optimizer.load_state_dict(state.optimizer_state_dict)``,
    ``scheduler.load_state_dict(state.scheduler_state_dict)``, and
    restoring each entry of ``rng_states`` to its respective generator.
    Keeping that out of ``load_checkpoint`` itself avoids hidden global
    side effects in what should be a pure read.
    """

    model_state_dict: dict[str, torch.Tensor]
    optimizer_state_dict: dict[str, Any]
    scheduler_state_dict: dict[str, Any]
    step: int
    rng_states: dict[str, Any]
    config: Config


def save_checkpoint(
    directory: Path,
    step: int,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    rng_states: dict[str, Any],
    config: Config,
) -> Path:
    """Atomically write a complete training-state snapshot.

    Steps:

    1. Create ``<directory>/step-XXXXXXXX.tmp/``.
    2. Write model weights to ``model.safetensors`` inside it.
    3. Write everything else (optimizer, scheduler, RNG, step, config)
       to ``state.pt`` inside it via ``torch.save``.
    4. ``os.replace`` the temp directory to its final name.

    If any step before the ``os.replace`` raises, the final-named
    directory is never created. The ``.tmp`` directory may persist;
    callers can clean it up at next startup.

    Parameters
    ----------
    directory
        Parent directory under which the per-step subdirectory is
        created. Must exist; this function does not ``mkdir -p`` it.
    step
        The optimizer-step number to encode into the subdirectory name.
    model
        Module whose parameters and buffers are to be saved. We use
        ``safetensors.torch.save_model``, which handles weight tying
        (e.g. GPT-2's tied ``wte``/``lm_head``) by writing one of the
        shared tensors and dropping the alias; ``save_file`` would
        raise on tied storages instead.
    optimizer
        Optimizer whose ``state_dict()`` is captured.
    scheduler
        Scheduler whose ``state_dict()`` is captured. ``LambdaLR``'s
        state contains ``lr_lambdas`` as a list of ``None`` placeholders
        for any lambda that wasn't a picklable function — that's fine,
        because the trainer reconstructs the schedule with
        ``build_lr_scheduler`` before calling ``load_state_dict``.
    rng_states
        Pre-captured RNG state dict. The caller is responsible for
        snapshotting immediately before invoking ``save_checkpoint`` so
        the snapshot reflects the moment-of-save state. Expected keys
        (the trainer's capture function and the test helpers all build
        this exact shape):

          ``torch``       — ``torch.get_rng_state()``
          ``torch_cuda``  — ``torch.cuda.get_rng_state_all()`` or ``[]``
          ``numpy``       — ``numpy.random.get_state()``
          ``python``      — ``random.getstate()``
    config
        The validated ``Config`` for this run. Stored as a JSON dict
        (Pydantic ``model_dump(mode="json")``) inside ``state.pt``;
        ``load_checkpoint`` revalidates it on read.

    Returns
    -------
    Path to the final, atomically-renamed ``step-XXXXXXXX/`` directory.
    """
    directory = Path(directory)
    final_dir = directory / f"{_STEP_PREFIX}{step:0{_STEP_DIGITS}d}"
    tmp_dir = directory / f"{_STEP_PREFIX}{step:0{_STEP_DIGITS}d}.tmp"

    # If a previous attempt crashed, ``mkdir`` will raise FileExistsError
    # below — that's the intended signal. The cleanup is the caller's
    # responsibility and lives outside this function on purpose.
    tmp_dir.mkdir(parents=True)

    # Weights first because they're the biggest and most likely to fail
    # on disk-full. Failing here means we don't even start writing
    # state.pt, which keeps the failure footprint minimal.
    safetensors.torch.save_model(model, str(tmp_dir / _MODEL_FILENAME))

    payload = {
        "step": step,
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "rng_states": rng_states,
        # JSON-mode dump: Path objects become strings, Literal types
        # become plain strings — the same form ``Config.from_yaml``
        # validates from. We store a dict (not the Pydantic instance)
        # so ``load_checkpoint`` doesn't need the exact same Pydantic
        # version as the writer.
        "config": config.model_dump(mode="json"),
    }
    torch.save(payload, tmp_dir / _STATE_FILENAME)

    # The atomic flip: after this returns, an outside observer sees
    # either the old state (no final_dir) or the new state (final_dir
    # present and complete). Never an in-between.
    os.replace(tmp_dir, final_dir)
    return final_dir


def load_checkpoint(path: Path) -> CheckpointState:
    """Read a checkpoint directory into a ``CheckpointState``.

    Pure read: this function does not apply any state to live objects.
    The caller (the trainer) decides when to call
    ``model.load_state_dict``, ``optimizer.load_state_dict``,
    ``scheduler.load_state_dict``, and the RNG setters.

    Parameters
    ----------
    path
        A ``step-XXXXXXXX/`` directory created by ``save_checkpoint``.

    Returns
    -------
    A ``CheckpointState`` whose fields mirror what ``save_checkpoint``
    received, plus the model state dict reconstituted from
    ``model.safetensors``.
    """
    path = Path(path)
    model_state_dict = safetensors.torch.load_file(str(path / _MODEL_FILENAME))

    # weights_only=False is required because ``state.pt`` carries
    # non-tensor Python objects (the Pydantic config dict, numpy RNG
    # state, Python random.getstate() tuple). PyTorch 2.6+ defaulted
    # weights_only to True for security; this file is one we wrote
    # ourselves, so opting out is safe and necessary.
    payload = torch.load(path / _STATE_FILENAME, weights_only=False)

    return CheckpointState(
        model_state_dict=model_state_dict,
        optimizer_state_dict=payload["optimizer"],
        scheduler_state_dict=payload["scheduler"],
        step=payload["step"],
        rng_states=payload["rng_states"],
        config=Config.model_validate(payload["config"]),
    )


def apply_retention_policy(
    directory: Path,
    last_n: int,
    best_so_far: Path | None,
) -> None:
    """Delete checkpoint subdirectories outside the retention window.

    Keeps the ``last_n`` newest ``step-*`` subdirectories (by step
    number, parsed from the name — not filesystem mtime, which can be
    perturbed by file copies) plus the directory referenced by
    ``best_so_far`` if any. Everything else is removed.

    Parameters
    ----------
    directory
        Parent directory containing ``step-XXXXXXXX`` subdirectories.
    last_n
        Number of most-recent checkpoints to keep. ``0`` means keep
        none of the recent ones (only ``best_so_far`` survives).
    best_so_far
        Optional path to a single checkpoint that should be kept
        regardless of recency. Owned by the trainer (val-loss tracker).
        Resolved against ``directory`` for the keep-set comparison so a
        relative or absolute reference both work.
    """
    directory = Path(directory)

    # Sort by parsed step number, descending. The ``step-`` prefix +
    # zero-padded suffix means alphabetical sort would also work, but
    # parsing the integer is the more obvious "what we mean" code.
    candidates = sorted(
        (p for p in directory.iterdir() if p.is_dir() and p.name.startswith(_STEP_PREFIX)),
        key=lambda p: int(p.name[len(_STEP_PREFIX) :]),
        reverse=True,
    )

    keep: set[Path] = set(candidates[:last_n])
    if best_so_far is not None:
        # ``resolve()`` on both sides so symlinks / relative inputs
        # compare correctly. We only add to ``keep`` if the path is
        # actually one of our candidates — silently keeping a path
        # outside the directory would be a no-op anyway.
        best_resolved = Path(best_so_far).resolve()
        for cand in candidates:
            if cand.resolve() == best_resolved:
                keep.add(cand)
                break

    for cand in candidates:
        if cand not in keep:
            shutil.rmtree(cand)


def _inspect(path: Path) -> str:
    """Render a human-readable summary of a checkpoint directory.

    Pulled out of ``__main__`` so the formatting can be unit-tested
    directly if we ever want to. Returns the printable string rather
    than printing it so the caller controls the destination.
    """
    state = load_checkpoint(path)
    lines: list[str] = [
        f"Checkpoint: {path}",
        f"Step: {state.step}",
        "",
        "Config:",
        f"  base_model_id: {state.config.model.base_model_id}",
        f"  dtype:         {state.config.model.dtype}",
        f"  sources:       {state.config.data.sources}",
        f"  max_steps:     {state.config.training.max_steps}",
        f"  lr:            {state.config.training.lr}",
        "",
        "Model tensors:",
    ]
    for key, tensor in state.model_state_dict.items():
        # ``tuple(tensor.shape)`` prints as ``(16, 8)`` rather than
        # ``torch.Size([16, 8])`` — the same form people read off in
        # docs and PyTorch error messages.
        lines.append(f"  {key}: {tuple(tensor.shape)} {tensor.dtype}")
    return "\n".join(lines)


if __name__ == "__main__":
    # Argparse is heavier than a manual sys.argv read, but the standard
    # in this repo is argparse for any user-facing CLI (see
    # ``scripts/sft_smoke.py``); matching keeps muscle memory consistent.
    import argparse

    parser = argparse.ArgumentParser(
        description="Inspect a finpost training checkpoint directory."
    )
    parser.add_argument(
        "--inspect",
        type=Path,
        required=True,
        help="Path to a step-XXXXXXXX/ directory created by save_checkpoint.",
    )
    args = parser.parse_args()
    print(_inspect(args.inspect))
