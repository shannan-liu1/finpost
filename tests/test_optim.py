"""Tests for the optimizer and learning-rate scheduler factories.

Each test pins one invariant of ``build_optimizer`` / ``build_lr_scheduler``:

Optimizer:
1. Two parameter groups exist (decay + no-decay).
2. Linear bias and LayerNorm weight/bias land in the no-decay group.
3. Linear weight (2-D) lands in the decay group.
4. Partition is exhaustive and disjoint: union covers every model parameter,
   intersection is empty.
5. Peak LR is propagated to both groups.

Scheduler:
1. Immediately after construction (step 0), every group's LR is 0
   (linear warmup starts at zero).
2. After ``warmup_steps`` calls to ``.step()``, every group's LR equals peak.
3. After ``total_steps`` calls to ``.step()``, every group's LR is ~0
   (cosine has decayed to its minimum).
4. LR is monotonically non-decreasing during warmup.
5. LR is monotonically non-increasing during the cosine decay phase.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from finpost.training.optim import build_lr_scheduler, build_optimizer


def _tiny_model() -> nn.Module:
    """A two-layer model exercising every parameter shape we care about.

    - ``Linear`` contributes a 2-D weight (decay) and a 1-D bias (no-decay).
    - ``LayerNorm`` contributes a 1-D weight and a 1-D bias (both no-decay).

    Using a hand-built module rather than loading a real Hugging Face
    model keeps the test fast and self-contained.
    """
    return nn.Sequential(
        nn.Linear(4, 8),
        nn.LayerNorm(8),
        nn.Linear(8, 2),
    )


# -----------------------------------------------------------------------------
# build_optimizer
# -----------------------------------------------------------------------------


def test_optimizer_has_two_param_groups() -> None:
    model = _tiny_model()
    opt = build_optimizer(model, lr=1e-4, weight_decay=0.1)
    assert len(opt.param_groups) == 2


def test_bias_and_norm_params_have_zero_weight_decay() -> None:
    """Bias and LayerNorm/RMSNorm parameters must not be weight-decayed.

    The standard recipe: applying weight decay to 1-D parameters (biases
    and norm scales) hurts training without compensating gain.
    """
    model = _tiny_model()
    opt = build_optimizer(model, lr=1e-4, weight_decay=0.1)

    # Build a name lookup so we can identify which group each tensor went into.
    name_for_id = {id(p): name for name, p in model.named_parameters()}

    no_decay_group = next(g for g in opt.param_groups if g["weight_decay"] == 0.0)
    no_decay_names = {name_for_id[id(p)] for p in no_decay_group["params"]}

    # Two Linears + one LayerNorm. The Linears each contribute a bias
    # (1-D); the LayerNorm contributes weight + bias (both 1-D).
    expected_no_decay = {
        "0.bias",  # Linear(4, 8) bias
        "1.weight",  # LayerNorm weight
        "1.bias",  # LayerNorm bias
        "2.bias",  # Linear(8, 2) bias
    }
    assert no_decay_names == expected_no_decay


def test_linear_weights_go_into_decay_group() -> None:
    model = _tiny_model()
    opt = build_optimizer(model, lr=1e-4, weight_decay=0.1)

    name_for_id = {id(p): name for name, p in model.named_parameters()}
    decay_group = next(g for g in opt.param_groups if g["weight_decay"] == 0.1)
    decay_names = {name_for_id[id(p)] for p in decay_group["params"]}

    # Only the 2-D Linear weights should be decayed.
    assert decay_names == {"0.weight", "2.weight"}


def test_param_groups_partition_all_model_parameters() -> None:
    """Every parameter appears in exactly one group; nothing is dropped."""
    model = _tiny_model()
    opt = build_optimizer(model, lr=1e-4, weight_decay=0.1)

    all_param_ids = {id(p) for p in model.parameters()}
    grouped_ids: list[int] = []
    for group in opt.param_groups:
        grouped_ids.extend(id(p) for p in group["params"])

    # No duplicates across groups.
    assert len(grouped_ids) == len(set(grouped_ids))
    # Exhaustive: every model parameter is assigned somewhere.
    assert set(grouped_ids) == all_param_ids


def test_optimizer_peak_lr_is_set_on_both_groups() -> None:
    model = _tiny_model()
    opt = build_optimizer(model, lr=3e-4, weight_decay=0.1)
    for group in opt.param_groups:
        assert group["lr"] == 3e-4


# -----------------------------------------------------------------------------
# build_lr_scheduler
# -----------------------------------------------------------------------------


def _fresh_optimizer(peak_lr: float = 1e-3) -> torch.optim.AdamW:
    """A minimal optimizer for scheduler tests.

    The scheduler's behavior depends only on the param-group ``lr``
    fields, not on the underlying parameters, so a single-group AdamW
    over a single tensor is enough.
    """
    dummy = nn.Linear(2, 2)
    return torch.optim.AdamW(dummy.parameters(), lr=peak_lr)


def _take_optimizer_step(opt: torch.optim.Optimizer) -> None:
    """Run a no-op optimizer step.

    PyTorch warns if ``scheduler.step()`` is called before
    ``optimizer.step()``. In real training the trainer's loop satisfies
    this naturally; in these tests we never run a forward/backward, so
    we install zero gradients and step the optimizer once between
    scheduler construction and our first ``sched.step()`` call.
    """
    for group in opt.param_groups:
        for p in group["params"]:
            p.grad = torch.zeros_like(p)
    opt.step()


def test_scheduler_lr_is_zero_at_step_zero() -> None:
    """Linear warmup must start at exactly zero, not at the peak.

    After ``LambdaLR.__init__`` PyTorch evaluates the lambda at
    ``last_epoch=0`` and writes the result to ``param_groups[*]['lr']``.
    So if the warmup factor at step 0 is 0, the LR we read here is 0.
    """
    peak = 1e-3
    opt = _fresh_optimizer(peak_lr=peak)
    _ = build_lr_scheduler(opt, total_steps=1000, warmup_steps=100)

    for group in opt.param_groups:
        assert group["lr"] == 0.0


def test_scheduler_lr_reaches_peak_at_end_of_warmup() -> None:
    peak = 1e-3
    opt = _fresh_optimizer(peak_lr=peak)
    sched = build_lr_scheduler(opt, total_steps=1000, warmup_steps=100)
    _take_optimizer_step(opt)  # silence the "step before optimizer" warning

    # Advance to the end of warmup. After 100 .step() calls,
    # last_epoch=100 and the warmup factor evaluates to 1.0.
    for _ in range(100):
        sched.step()

    for group in opt.param_groups:
        assert math.isclose(group["lr"], peak, rel_tol=0.0, abs_tol=1e-12)


def test_scheduler_lr_is_zero_at_total_steps() -> None:
    peak = 1e-3
    opt = _fresh_optimizer(peak_lr=peak)
    sched = build_lr_scheduler(opt, total_steps=1000, warmup_steps=100)
    _take_optimizer_step(opt)

    for _ in range(1000):
        sched.step()

    # cosine(pi) = -1, so 0.5 * (1 + cos(pi)) = 0 exactly. Allow a tiny
    # absolute slack to absorb floating-point rounding through the
    # division and the cosine call.
    for group in opt.param_groups:
        assert abs(group["lr"]) < 1e-6


def test_scheduler_is_monotone_non_decreasing_during_warmup() -> None:
    peak = 1e-3
    opt = _fresh_optimizer(peak_lr=peak)
    sched = build_lr_scheduler(opt, total_steps=1000, warmup_steps=100)
    _take_optimizer_step(opt)

    # Snapshot LR at each warmup step.
    lrs = [opt.param_groups[0]["lr"]]
    for _ in range(100):
        sched.step()
        lrs.append(opt.param_groups[0]["lr"])

    for prev, curr in zip(lrs[:-1], lrs[1:], strict=True):
        assert curr >= prev


def test_scheduler_is_monotone_non_increasing_after_warmup() -> None:
    peak = 1e-3
    opt = _fresh_optimizer(peak_lr=peak)
    sched = build_lr_scheduler(opt, total_steps=1000, warmup_steps=100)
    _take_optimizer_step(opt)

    # Burn through warmup first.
    for _ in range(100):
        sched.step()

    lrs = [opt.param_groups[0]["lr"]]
    for _ in range(900):
        sched.step()
        lrs.append(opt.param_groups[0]["lr"])

    for prev, curr in zip(lrs[:-1], lrs[1:], strict=True):
        assert curr <= prev
