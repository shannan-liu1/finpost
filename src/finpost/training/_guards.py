"""Minimal numerical guards for the training loop.

Lives in its own module so the regression test can import the guard
without pulling in the full trainer (which transitively imports
``datasets``, ``transformers``, ``wandb``, etc.). Keeps the test
runnable on lean local environments that only have ``torch``.
"""

from __future__ import annotations

import torch


def check_finite_loss(loss: torch.Tensor, step: int) -> None:
    """Raise loudly if the training loss is NaN or inf.

    Called once per training micro-batch. Without this guard the trainer
    silently logs NaN to wandb and continues updating parameters with
    NaN gradients, destroying the model over many steps. Failing fast
    means a broken recipe surfaces as a Python traceback (and a non-zero
    subprocess exit code, which is what the runpod-canary cell relies on
    as its pass criterion).

    Validation-time NaN is intentionally NOT checked here — validation
    is a no-grad measurement and a NaN in val is informative for
    debugging rather than corrupting; the loop continues so subsequent
    val passes can still happen.
    """
    if not torch.isfinite(loss):
        raise RuntimeError(
            f"Non-finite loss at step {step}: {loss.item()}"
        )
