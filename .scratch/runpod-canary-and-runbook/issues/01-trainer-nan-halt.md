# 01 - Add non-finite-loss guard to trainer

- **Status:** Ready
- **Ready for agent:** yes
- **Depends on:** none

## Goal

Make the trainer fail loud on the first non-finite loss instead of silently logging NaN to wandb and continuing to update parameters with NaN gradients. Two lines in the training loop.

## Scope

**In scope:**
- Insert non-finite check immediately after `loss = self._forward_loss(batch)` in `Trainer._run_training_loop` (`src/finpost/training/trainer.py`).
- Error message includes `self.global_step` and `loss.item()`.
- New regression test `tests/test_trainer_nan_halt.py` that red-greens on the guard.

**Out of scope:**
- Validation-pass NaN detection. The guard is training-only; validation NaN is informative for debugging and shouldn't crash the run.
- Configurable tolerance for transient NaNs. Not needed for our recipe; can be added behind a flag later if a future recipe requires it.
- Detecting NaN gradients post-backward. Loss-NaN is the upstream cause; if the loss is finite but the grads aren't, something deeper is wrong and the existing grad_clip log will show it.

## Implementation

In `src/finpost/training/trainer.py`, add a small module-level helper:

```python
def _check_finite_loss(loss: torch.Tensor, step: int) -> None:
    """Raise loudly if the training loss is NaN or inf.

    Called once per training micro-batch. Without this guard the trainer
    silently logs NaN to wandb and continues updating parameters with
    NaN gradients, destroying the model over many steps.
    """
    if not torch.isfinite(loss):
        raise RuntimeError(
            f"Non-finite loss at step {step}: {loss.item()}"
        )
```

Then in `_run_training_loop`, locate this line:

```python
loss = self._forward_loss(batch)
```

Insert immediately after:

```python
_check_finite_loss(loss, self.global_step)
```

`torch` is already imported at the top of the file. Why a separate helper rather than inlining the two lines: it lets the unit test import and exercise the check WITHOUT importing the full `Trainer` class (which transitively imports `finpost.training.config`, which requires a modern `pydantic`). Keeps the test runnable on any environment that has `torch`.

## Test

`tests/test_trainer_nan_halt.py` (new). Imports only the helper:

```python
import pytest
import torch
from finpost.training.trainer import _check_finite_loss


def test_check_finite_loss_passes_on_finite_value() -> None:
    _check_finite_loss(torch.tensor(2.5), step=7)


def test_check_finite_loss_raises_on_nan() -> None:
    with pytest.raises(RuntimeError, match="Non-finite loss at step 42"):
        _check_finite_loss(torch.tensor(float("nan")), step=42)


def test_check_finite_loss_raises_on_inf() -> None:
    with pytest.raises(RuntimeError, match="Non-finite loss"):
        _check_finite_loss(torch.tensor(float("inf")), step=0)
```

Three cases: finite passes silently, NaN raises with step in the message, inf also raises. No `Trainer` instantiation, no `Config` import, no pydantic dependency — runs on any box with `torch`.

## Acceptance criteria

1. `pytest tests/test_trainer_nan_halt.py -v` passes.
2. Reverting the guard (`git stash` the two-line insertion, rerun the test) makes the test fail with a clear "expected RuntimeError" assertion.
3. `pytest tests/test_loss_dtype.py tests/test_dataset.py tests/test_trainer*.py -v` all still pass (no regressions).
4. The guard fires before the optimizer step, before wandb logging, before checkpoint cadence — i.e. the next line after the guard is the existing `useful_tokens = ...` calculation, untouched.
