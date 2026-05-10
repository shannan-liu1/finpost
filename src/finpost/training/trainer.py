"""End-to-end Supervised Fine-Tuning trainer.

This module is the heart of Phase 1: a single ``Trainer(config).train()``
call wires the dataset, optimizer, scheduler, checkpointer, and Weights
& Biases logger into one loop. Everything visible here is a thin glue
layer over the modules in this package; the goal is for every line to
be intelligible without library magic.

Why a class and not a function:
  the loop is small but it has half a dozen pieces of state that need
  to be available to ``train()`` and ``validate()`` in tandem (model,
  loaders, optimizer, scheduler, device, step counter). Threading those
  through function arguments makes the body harder to read, not easier.
  A class with a single ``train()`` entrypoint and a small ``validate()``
  helper keeps the surface narrow without hiding behaviour.

Why we re-implement the loop instead of using ``transformers.Trainer``:
  this is a learning project. Every comment in here is for the user
  reading the code line by line.

Known limitation: DataLoader iterator state is NOT part of the
checkpoint. ``save_checkpoint`` captures model parameters, optimizer
state, scheduler state, RNG snapshots, the global step counter, and
the validated Config — but not the position of the train loader's
iterator. On resume, ``iter(train_loader)`` starts a fresh pass at
batch 0 even when training stopped mid-epoch. In practice this means
a resumed run replays a small number of already-seen batches and may
show a tiny upward blip in loss for the first few post-resume steps.
For the typical SFT use case (large datasets, many epochs) this is
negligible. Restoring iterator state would require subclassing the
DataLoader sampler to checkpoint its index — out of scope for issue 05.
"""

from __future__ import annotations

import os
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import wandb
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer

from finpost.training.checkpoint import (
    apply_retention_policy,
    load_checkpoint,
    save_checkpoint,
)
from finpost.training.config import Config
from finpost.training.dataset import make_loaders
from finpost.training.masking import IGNORE_INDEX
from finpost.training.optim import build_lr_scheduler, build_optimizer
from finpost.training.sft import compute_masked_ce_loss


def _capture_rng_states() -> dict[str, Any]:
    """Snapshot every RNG that influences training stochasticity.

    Mirrors the dict shape that ``checkpoint.save_checkpoint`` documents
    and that ``checkpoint.load_checkpoint`` returns. Inlined here (and
    in tests) instead of exported from ``checkpoint.py`` so the
    checkpoint module's public surface stays narrow.

    The four RNGs cover everything the trainer touches:
      - ``torch``       — model init, dropout, anything calling ``torch.rand*``
      - ``torch_cuda``  — GPU-side RNG (kernel-level dropout, etc.)
      - ``numpy``       — used by ``np.random.*`` if any pipeline reaches for it
      - ``python``      — used by ``random.shuffle`` inside ``_split_examples``
    """
    return {
        "torch": torch.get_rng_state(),
        # ``get_rng_state_all`` returns one tensor per CUDA device. Empty
        # list when CUDA is unavailable so the dict shape is stable.
        "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
        "numpy": np.random.get_state(),
        "python": random.getstate(),
    }


def _apply_rng_states(rng_states: dict[str, Any]) -> None:
    """Restore every RNG captured by ``_capture_rng_states``.

    The opposite of ``_capture_rng_states``. The trainer applies these
    after a checkpoint resume so that the next batch of stochastic
    operations (dropout, any sampler-side shuffle) produces the same
    sequence as it would have in an uninterrupted run.
    """
    torch.set_rng_state(rng_states["torch"])
    if torch.cuda.is_available() and rng_states["torch_cuda"]:
        torch.cuda.set_rng_state_all(rng_states["torch_cuda"])
    np.random.set_state(rng_states["numpy"])
    random.setstate(rng_states["python"])


def _seed_everything(seed: int) -> None:
    """Seed every RNG the trainer touches with one integer.

    Single-source-of-truth pattern: one int from the config controls
    every stochastic component. Tests rely on this for determinism
    (criterion 2): two ``Trainer(config).train()`` calls with the same
    config must produce bit-identical loss curves, which they only can
    if every RNG starts from the same state.
    """
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        # ``manual_seed_all`` seeds every CUDA device; ``manual_seed``
        # only seeds the current device. Cheap to call even when there
        # is just one GPU; harmless when there are zero.
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


class Trainer:
    """End-to-end Supervised Fine-Tuning trainer.

    Responsibilities, in order:
      1. Seed every RNG from ``config.data.seed``.
      2. Load tokenizer + model in the configured dtype.
      3. Build train / val DataLoaders via ``make_loaders``.
      4. Build optimizer + LR scheduler via the factories.
      5. (Optional) restore from ``config.checkpointing.resume_from``.
      6. Initialise Weights & Biases (honours ``WANDB_MODE`` env var).
      7. Run the optimiser loop until ``config.training.max_steps``.

    Construction is intentionally side-effect-free: nothing happens
    until ``train()`` is called. Two ``Trainer(config)`` instances
    therefore differ only in identity, not in any captured state —
    which is what allows ``train()`` to be called twice on a fresh
    Trainer and produce bit-identical loss curves.
    """

    def __init__(self, config: Config) -> None:
        # Hold the config; defer all expensive work to ``train()``. This
        # keeps construction cheap and side-effect-free, and means
        # criterion 2 (determinism over two ``Trainer(config).train()``
        # calls) doesn't need any special bookkeeping in __init__.
        self.config = config
        # Pick the device once. Single-GPU only by spec — no DDP, no
        # device placement gymnastics. CPU fallback so tests and the
        # tiny-gpt2 soft-launch run anywhere.
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # State populated by ``train()``. Declared here so type-checkers
        # and readers know what the Trainer carries; their None default
        # marks "not yet initialised". ``validate()`` raises if called
        # before training has populated these.
        self.model: torch.nn.Module | None = None
        self.tokenizer: Any = None
        self.train_loader: DataLoader | None = None
        self.val_loader: DataLoader | None = None
        self.optimizer: torch.optim.Optimizer | None = None
        self.scheduler: torch.optim.lr_scheduler.LRScheduler | None = None
        self.global_step: int = 0

    # -----------------------------------------------------------------
    # Public entrypoints
    # -----------------------------------------------------------------

    def train(self) -> None:
        """Run the full SFT loop until ``max_steps``."""
        self._setup()
        self._run_training_loop()
        self._teardown()

    def validate(self) -> float:
        """Average masked CE loss over the entire validation loader.

        Public so the trainer can call it inside the loop AND so a user
        can poke it from a notebook to debug a checkpoint. Reads from
        ``self`` rather than taking arguments because ``self.model``,
        ``self.val_loader``, and ``self.device`` are exactly what the
        loop has already prepared; passing them in just to validate
        would invite "did you remember to put the model in eval()?"
        bugs.
        """
        if self.model is None or self.val_loader is None:
            raise RuntimeError(
                "Trainer.validate() called before train() populated model/val_loader"
            )

        # eval() flips dropout / any train-only modules to inference
        # behaviour. We restore train() at the end so the caller (the
        # training loop) doesn't have to remember to do it.
        was_training = self.model.training
        self.model.eval()

        total_loss = 0.0
        total_batches = 0
        # ``no_grad`` saves memory and a touch of time in the val pass.
        # We don't need gradients — only the loss value for logging.
        with torch.no_grad():
            for batch in self.val_loader:
                loss = self._forward_loss(batch)
                # ``.item()`` releases the autograd graph (which we
                # don't have anyway under no_grad) and gives a Python
                # float. Accumulate in fp32 — the loss itself comes back
                # in whatever dtype the model uses.
                total_loss += float(loss.detach().float().item())
                total_batches += 1

        if was_training:
            self.model.train()

        # Empty val loader → return 0.0 rather than dividing by zero.
        # This is a safety net for the smoke-test path where val is
        # tiny; real training always has at least one batch.
        return total_loss / total_batches if total_batches > 0 else 0.0

    # -----------------------------------------------------------------
    # Setup
    # -----------------------------------------------------------------

    def _setup(self) -> None:
        """Seeds, model, tokenizer, loaders, optimizer, scheduler, wandb."""
        # 1. Seeds first. Every later object that uses a RNG must do so
        # AFTER this call so its initial state is deterministic.
        _seed_everything(self.config.data.seed)

        # 2. Tokenizer. GPT-style tokenizers (including tiny-gpt2 and
        # Qwen) often have no pad_token; reuse eos_token. The collator
        # uses eos_token_id as a between-document separator, so it must
        # be set; the attention_mask the collator builds tells the
        # model where padding is regardless of which id we use here.
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.model.base_model_id)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # 3. Model. ``getattr(torch, ...)`` is the standard way to turn
        # a config string ("bfloat16") into a torch.dtype. We move to
        # device immediately so subsequent .grad allocations live there.
        dtype = getattr(torch, self.config.model.dtype)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.model.base_model_id,
            dtype=dtype,
            use_safetensors=self.config.model.use_safetensors,
        ).to(self.device)
        self.model.train()

        # 4. Data. ``make_loaders`` is the single source of truth for
        # how examples become packed batches; the trainer never sees
        # raw text.
        self.train_loader, self.val_loader = make_loaders(self.config, self.tokenizer)

        # 5. Optimizer + scheduler. The factories implement the
        # standard "two param groups, cosine with linear warmup"
        # recipe. ``total_steps`` for the scheduler is the configured
        # ``max_steps`` — that's what the cosine decays over.
        self.optimizer = build_optimizer(
            self.model,
            lr=self.config.training.lr,
            weight_decay=self.config.training.weight_decay,
        )
        self.scheduler = build_lr_scheduler(
            self.optimizer,
            total_steps=self.config.training.max_steps,
            warmup_steps=self.config.training.warmup_steps,
        )

        # 6. Resume (optional). Order matters: seed first so model
        # init is deterministic; THEN load checkpoint to overwrite
        # parameters and RNG. Re-seeding after the load would clobber
        # the resumed RNG snapshot and break criterion 3.
        if self.config.checkpointing.resume_from is not None:
            self._load_resume(self.config.checkpointing.resume_from)

        # 7. Weights & Biases. ``mode="disabled"`` honours WANDB_MODE
        # automatically (wandb checks the env var); passing it
        # explicitly belt-and-suspenders the offline test path.
        wandb.init(
            project=self.config.logging.wandb_project,
            name=self.config.logging.run_name,
            config=self.config.model_dump(mode="json"),
            mode=os.environ.get("WANDB_MODE"),
        )

    def _load_resume(self, resume_from: Path) -> None:
        """Restore model, optimizer, scheduler, RNG state, and step."""
        # ``checkpoint.load_checkpoint`` is a pure read; the trainer is
        # responsible for applying each piece. ``strict=False`` on the
        # model state dict is required because safetensors drops one
        # key of any tied pair (e.g. GPT-2's wte / lm_head); the fresh
        # model already has the tying in place so the missing key is
        # harmless.
        state = load_checkpoint(resume_from)
        assert self.model is not None and self.optimizer is not None and self.scheduler is not None
        self.model.load_state_dict(state.model_state_dict, strict=False)
        self.optimizer.load_state_dict(state.optimizer_state_dict)
        self.scheduler.load_state_dict(state.scheduler_state_dict)
        _apply_rng_states(state.rng_states)
        self.global_step = state.step

    # -----------------------------------------------------------------
    # Training loop
    # -----------------------------------------------------------------

    def _run_training_loop(self) -> None:
        """Iterate over (cycling) train batches until ``max_steps``."""
        assert (
            self.train_loader is not None
            and self.optimizer is not None
            and self.scheduler is not None
        )

        max_steps = self.config.training.max_steps
        grad_accum = self.config.training.grad_accum_steps
        grad_clip = self.config.training.grad_clip
        val_every = self.config.training.val_every_n_steps
        ckpt_every = self.config.training.checkpoint_every_n_steps

        # Loss accumulated across micro-batches, reset on every
        # optimizer step. We log the AVERAGE micro-batch loss for the
        # step (so the curve is comparable across grad_accum settings),
        # which means dividing by grad_accum before adding.
        accumulated_loss = 0.0
        # Counter of micro-batches consumed since the last optimizer
        # step. When this hits ``grad_accum_steps`` we apply the update.
        micro_step = 0
        # Wall-clock + token bookkeeping for tokens_per_sec. Reset every
        # 50 optimizer steps. ``time.perf_counter`` is the right clock
        # for elapsed wall time; ``time.time`` jumps when the system
        # clock is adjusted.
        window_tokens = 0
        window_start = time.perf_counter()

        # Cycling iterator: when the loader exhausts, start a new pass.
        # Gives us "train for max_steps regardless of dataset size",
        # which matches the spec's steps-primary cadence.
        loader_iter = iter(self.train_loader)
        while self.global_step < max_steps:
            try:
                batch = next(loader_iter)
            except StopIteration:
                loader_iter = iter(self.train_loader)
                batch = next(loader_iter)

            # Forward + masked CE loss. ``compute_masked_ce_loss`` returns
            # whatever dtype the model uses; we cast to fp32 during
            # accumulation below (see ``loss.detach().float()`` a few lines
            # down) so precision isn't lost across micro-batches in bf16.
            loss = self._forward_loss(batch)

            # Useful tokens = response positions = labels != IGNORE_INDEX.
            # We count them on CPU because the labels are already there
            # and this avoids a host/device sync inside the hot loop.
            useful_tokens = int((batch["labels"] != IGNORE_INDEX).sum().item())
            window_tokens += useful_tokens

            # Scaled backward: dividing by grad_accum makes the
            # accumulated gradient have the same magnitude as a single
            # step over the full effective batch. Without this, a
            # grad_accum=N run would step with N× the gradient and
            # behave like a much higher learning rate.
            (loss / grad_accum).backward()
            accumulated_loss += float(loss.detach().float().item()) / grad_accum
            micro_step += 1

            if micro_step % grad_accum != 0:
                # Mid-accumulation: no optimizer update, no logging
                # yet — the partial loss isn't a meaningful step.
                continue

            # End of a full effective batch. Clip first (returns the
            # PRE-clip norm, which is the value worth logging — it
            # tells you whether clipping actually fired). Then step
            # the optimizer and scheduler in lockstep.
            grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), grad_clip)
            self.optimizer.step()
            self.scheduler.step()
            self.optimizer.zero_grad()

            self.global_step += 1
            # Logging happens AFTER the increment so wandb's recorded
            # step matches our internal step counter exactly.
            self._log_train_metrics(
                step=self.global_step,
                loss=accumulated_loss,
                grad_norm=float(grad_norm),
            )

            # Tokens-per-sec window. Log only every 50 steps to avoid
            # noisy throughput numbers from per-step jitter.
            if self.global_step % 50 == 0:
                elapsed = time.perf_counter() - window_start
                if elapsed > 0:
                    wandb.log(
                        {"train/tokens_per_sec": window_tokens / elapsed},
                        step=self.global_step,
                    )
                window_tokens = 0
                window_start = time.perf_counter()

            # Reset micro-batch accumulators for the next effective batch.
            accumulated_loss = 0.0
            micro_step = 0

            # Validation cadence. Done in eval()+no_grad inside
            # ``validate()``; we just log the resulting scalar.
            if self.global_step % val_every == 0:
                val_loss = self.validate()
                wandb.log({"val/loss": val_loss}, step=self.global_step)

            # Checkpoint cadence. Save first, then prune. ``best_so_far``
            # is None for now (no val-loss tracking yet — out of scope
            # for this issue); retention keeps the most recent ``last_n``.
            if self.global_step % ckpt_every == 0:
                self._save_checkpoint()

            # Hit the step cap inside the inner loop so we don't keep
            # consuming batches we'll never train on.
            if self.global_step >= max_steps:
                break

    def _forward_loss(self, batch: dict[str, Any]) -> torch.Tensor:
        """Run forward pass + masked CE loss for one (micro-)batch.

        Shared by training and validation so the masking and dtype
        handling stay in one place. Returns the loss tensor (still
        attached to the autograd graph for the training caller; the
        validation caller wraps in ``no_grad`` so the graph never gets
        built).
        """
        assert self.model is not None
        # Move each tensor to the training device. Dicts have no .to()
        # method that recurses; iterating explicitly avoids surprising
        # the reader and keeps the ``document_boundaries`` list (which
        # is a Python object, not a tensor) on the host.
        input_ids = batch["input_ids"].to(self.device)
        labels = batch["labels"].to(self.device)
        # Collator emits int64 attention_mask; PyTorch's SDPA path
        # requires bool or float masks. Cast to bool explicitly: True
        # means "attend here", False means "block".
        attention_mask = batch["attention_mask"].to(self.device).bool()
        position_ids = batch["position_ids"].to(self.device)

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
        )
        return compute_masked_ce_loss(outputs.logits, labels)

    # -----------------------------------------------------------------
    # Logging + checkpointing helpers
    # -----------------------------------------------------------------

    def _log_train_metrics(self, *, step: int, loss: float, grad_norm: float) -> None:
        """One ``wandb.log`` call with all per-step training scalars."""
        assert self.optimizer is not None
        # The optimizer's first param group's lr is the canonical
        # current LR. Both groups always share the same scheduler
        # multiplier, so reading from group 0 is sufficient.
        current_lr = self.optimizer.param_groups[0]["lr"]
        wandb.log(
            {
                "train/loss": loss,
                "train/lr": current_lr,
                "train/grad_norm": grad_norm,
            },
            step=step,
        )

    def _save_checkpoint(self) -> Path:
        """Save a checkpoint at the current step and apply retention."""
        assert (
            self.model is not None
            and self.optimizer is not None
            and self.scheduler is not None
        )
        save_dir = Path(self.config.checkpointing.save_dir)
        # ``save_checkpoint`` requires the parent directory to exist;
        # creating it lazily here means a fresh run doesn't have to
        # remember to ``mkdir -p`` outside.
        save_dir.mkdir(parents=True, exist_ok=True)
        rng_states = _capture_rng_states()
        path = save_checkpoint(
            directory=save_dir,
            step=self.global_step,
            model=self.model,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            rng_states=rng_states,
            config=self.config,
        )
        # Retention pruning: keep only the configured ``last_n`` newest.
        # ``best_so_far=None`` because val-loss tracking is out of scope
        # for this issue.
        apply_retention_policy(
            directory=save_dir,
            last_n=self.config.checkpointing.retention_last_n,
            best_so_far=None,
        )
        return path

    # -----------------------------------------------------------------
    # Teardown
    # -----------------------------------------------------------------

    def _teardown(self) -> None:
        """Final checkpoint + close the wandb run cleanly."""
        # Save a final checkpoint UNLESS the previous step already
        # landed on the cadence boundary — in that case ``_save_checkpoint``
        # has already written this exact step and re-saving would error
        # on the duplicate directory name.
        ckpt_every = self.config.training.checkpoint_every_n_steps
        if self.global_step > 0 and self.global_step % ckpt_every != 0:
            self._save_checkpoint()
        # ``finish`` flushes the run record. Idempotent: safe to call
        # even when ``init`` ran in disabled mode.
        wandb.finish()
