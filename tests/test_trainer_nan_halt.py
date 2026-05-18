"""Regression test for the non-finite-loss guard in the trainer.

Without this guard the trainer silently logged NaN to wandb and kept
training with NaN gradients, eventually destroying the model. The fix is
a 2-line helper called once per micro-batch.

The test imports ONLY the helper, not the full ``Trainer`` class. The
``Trainer`` import would pull in ``finpost.training.config`` (pydantic),
which on older local pydantic installs fails with ``cannot import name
'model_validator'``. The helper is a pure function on a tensor + int, so
isolating the test from that dependency means the regression check runs
on any environment that has torch.
"""

from __future__ import annotations

import pytest
import torch

from finpost.training._guards import check_finite_loss


def test_check_finite_loss_passes_on_finite_value() -> None:
    """A normal finite loss must not raise — guard is a no-op on success."""
    check_finite_loss(torch.tensor(2.5), step=7)


def test_check_finite_loss_raises_on_nan() -> None:
    """NaN loss must raise with the step number in the message.

    The step number in the error matters: when the canary or production
    run dies, the first thing you want to know is "at what step?".
    """
    with pytest.raises(RuntimeError, match="Non-finite loss at step 42"):
        check_finite_loss(torch.tensor(float("nan")), step=42)


def test_check_finite_loss_raises_on_inf() -> None:
    """Inf is also non-finite and also needs to halt training."""
    with pytest.raises(RuntimeError, match="Non-finite loss"):
        check_finite_loss(torch.tensor(float("inf")), step=0)


def test_check_finite_loss_raises_on_negative_inf() -> None:
    """-inf can come from log(0) in the loss path; also must halt."""
    with pytest.raises(RuntimeError, match="Non-finite loss"):
        check_finite_loss(torch.tensor(float("-inf")), step=3)
