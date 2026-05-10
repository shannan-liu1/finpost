"""Command-line entry point for ``python -m finpost.training.train``.

Wires the Phase 1 training stack from issues 01-05 behind a single
command:

    python -m finpost.training.train --config experiments/baseline.yaml

Responsibilities (intentionally narrow — the heavy lifting is in
``Trainer``):

  1. Parse a small set of CLI flags.
  2. Load the YAML config, apply CLI overrides into the dict BEFORE
     ``Config.model_validate`` so the resulting Config is itself
     validated and frozen — never patched after the fact.
  3. Print the effective config and a "steps per epoch" hint so the
     user can see at a glance what run they actually got.
  4. Construct ``Trainer(config)`` and call ``.train()``.

Why YAML overrides happen at the dict level rather than via Pydantic's
``model_copy(update=...)``:
  the sub-models are frozen, and ``model_copy`` on the top-level Config
  cannot reach into a frozen ``training`` sub-model without a second
  copy step. Loading the YAML, mutating the plain dict, and then
  re-validating is shorter, keeps validation in one place, and means
  every run hits exactly one ``Config.model_validate`` call.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import IO

import torch
import yaml

from finpost.training.config import Config
from finpost.training.trainer import Trainer


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    """Define the CLI surface in one place.

    The flag list is deliberately small — anything richer (multiple
    overrides, sweeps, env-var injection) belongs in a downstream
    runner, not in the trainer entry point. Each supported flag pins
    one user need:

      ``--config``       — the YAML to run; required, since this is the
                           only way to specify model/data/training.
      ``--device``       — override device selection. CI runs on CPU,
                           Colab on cuda; the default (``Trainer``'s
                           cuda-if-available pick) covers most cases.
      ``--max-steps``    — short-circuit ``training.max_steps`` from the
                           YAML for soft launches, smoke tests, and
                           the local TinyGPT canary.
      ``--resume-from``  — point at a ``step-XXXXXXXX/`` directory to
                           continue a run; equivalent to setting
                           ``checkpointing.resume_from`` in the YAML.
    """
    parser = argparse.ArgumentParser(
        prog="python -m finpost.training.train",
        description="Run Phase 1 Supervised Fine-Tuning from a YAML config.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to a training YAML (e.g. experiments/baseline.yaml).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Override device selection (e.g. 'cpu', 'cuda', 'cuda:0').",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="Override training.max_steps from the YAML.",
    )
    parser.add_argument(
        "--resume-from",
        type=Path,
        default=None,
        help="Override checkpointing.resume_from with a step-XXXXXXXX/ directory.",
    )
    return parser.parse_args(argv)


def resolve_config(
    *,
    config_path: Path,
    max_steps_override: int | None,
    resume_from_override: Path | None,
) -> Config:
    """Load the YAML, splice CLI overrides into the dict, validate.

    Pulled out of ``main`` so unit tests can exercise the merge
    semantics without touching argparse or stdout. CLI flags WIN over
    YAML values: the spec is explicit on this and a user passing
    ``--max-steps 20`` for a soft launch needs to be sure the trainer
    sees 20 even when the file says 3000.

    The merge happens at the dict level so the final
    ``Config.model_validate`` is the single source of validation
    truth. Doing it via ``model_copy(update=...)`` would require
    rebuilding the frozen sub-models manually, which is more code to
    read and a second place to make a typo.
    """
    with config_path.open("r", encoding="utf-8") as fp:
        raw = yaml.safe_load(fp)
    if not isinstance(raw, dict):
        raise ValueError(
            f"YAML root must be a mapping; got {type(raw).__name__} from {config_path}"
        )

    if max_steps_override is not None:
        # ``training`` is required by the schema, so this key must
        # already exist in the YAML. ``setdefault`` keeps the diff
        # minimal — we only touch the field we're overriding.
        raw.setdefault("training", {})["max_steps"] = max_steps_override

    if resume_from_override is not None:
        # ``checkpointing`` is optional; create the section if the
        # YAML omitted it entirely. Pydantic accepts a ``str`` here
        # and coerces to ``Path`` during validation.
        raw.setdefault("checkpointing", {})["resume_from"] = str(resume_from_override)

    return Config.model_validate(raw)


def print_effective_config(config: Config, *, stream: IO[str] | None = None) -> None:
    """Print a one-screen summary of the resolved Config.

    We print BEFORE training starts so that:
      - a misconfigured run is obvious from the first second of output,
      - the run's identity (model, lr, batch sizes, save dir) lives in
        the log next to the loss curve when someone re-reads it later.

    The "steps per epoch" line is the user-visible reminder of the
    steps↔epochs relationship. We don't compute the actual epoch
    count here — that requires loading the dataset, which is
    ``Trainer._setup``'s job. Instead we print the formula and the
    inputs the user needs to derive the number once the loader has
    reported its dataset size.
    """
    out = stream if stream is not None else sys.stdout

    effective_batch = (
        config.training.per_device_batch_size * config.training.grad_accum_steps
    )

    lines = [
        "Effective config:",
        f"  model:                  {config.model.base_model_id}",
        f"  dtype:                  {config.model.dtype}",
        f"  data sources:           {config.data.sources}",
        f"  max_steps:              {config.training.max_steps}",
        f"  warmup_steps:           {config.training.warmup_steps}",
        f"  lr:                     {config.training.lr}",
        f"  per_device_batch_size:  {config.training.per_device_batch_size}",
        f"  grad_accum_steps:       {config.training.grad_accum_steps}",
        f"  effective batch size:   {effective_batch}",
        f"  max_seq_len:            {config.packing.max_seq_len}",
        f"  save_dir:               {config.checkpointing.save_dir}",
        f"  resume_from:            {config.checkpointing.resume_from}",
        f"  run_name:               {config.logging.run_name}",
        # The literal phrase "steps per epoch" is load-bearing — the
        # CLI test pins it. The formula is the actionable part for
        # the user; we leave the absolute number to the Trainer's
        # post-load logging.
        f"  steps per epoch:        len(train_dataset) / {effective_batch}"
        " (computed once Trainer loads the dataset)",
    ]
    out.write("\n".join(lines) + "\n")
    out.flush()


def main(argv: list[str] | None = None) -> None:
    """Top-level entry point.

    ``argv=None`` lets argparse read ``sys.argv`` (the production path);
    passing an explicit list makes in-process tests possible. Returns
    ``None`` rather than an exit code: argparse exits the process
    itself on bad input, and a successful train() finishes via
    ``wandb.finish()`` and a normal Python return.
    """
    args = _parse_args(argv)
    config = resolve_config(
        config_path=args.config,
        max_steps_override=args.max_steps,
        resume_from_override=args.resume_from,
    )
    print_effective_config(config)

    trainer = Trainer(config)
    if args.device is not None:
        # Trainer picks device in __init__ ("cuda if available else cpu");
        # _setup() then does ``model.to(self.device)``. Overriding the
        # attribute between construction and ``train()`` lets the CLI
        # honour ``--device`` without modifying Trainer (out of scope
        # for this issue).
        trainer.device = torch.device(args.device)
    trainer.train()


if __name__ == "__main__":
    main()
