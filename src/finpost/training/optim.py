"""Optimizer and learning-rate scheduler factories.

Two small factories that turn a model and a few scalars into the
exact ``torch.optim`` objects the trainer will use:

  ``build_optimizer``      — AdamW with weight decay split into two
                             parameter groups: 1-D parameters (biases
                             and norm scales) get ``weight_decay=0``;
                             everything else gets the configured
                             weight decay.
  ``build_lr_scheduler``   — cosine schedule with linear warmup,
                             implemented inline as a ``LambdaLR``.

The split-decay recipe is standard practice for transformer training.
Weight decay on bias and norm parameters tends to slightly degrade
training without measurable upside; modern Hugging Face / PyTorch
training scripts all do this split. We do it from scratch here (rather
than importing a helper) because the rule is short and worth seeing.

The schedule is the ubiquitous "warmup then cosine to zero" used by
nearly every transformer SFT recipe. Implementing it ourselves keeps
the control flow fully visible — six lines of math, no ``transformers``
dependency.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn


def build_optimizer(
    model: nn.Module,
    lr: float,
    weight_decay: float,
) -> torch.optim.AdamW:
    """Construct AdamW with the standard two-group weight-decay split.

    Group 1 (decay): every parameter whose name does not end in
    ``.bias`` and whose tensor has rank > 1 — i.e. the matmul weights.
    Group 2 (no decay): biases and any 1-D parameters (LayerNorm /
    RMSNorm scale and shift). The ``param.ndim == 1`` test is what
    catches both LayerNorm and RMSNorm without needing to introspect
    module types: norm parameters are always 1-D.

    Parameters
    ----------
    model
        The model whose parameters will be optimized. Only parameters
        with ``requires_grad=True`` are included; frozen parameters
        are skipped (passing them to AdamW wastes memory on unused
        moment buffers).
    lr
        Peak learning rate. Set on both groups; the scheduler will
        scale this down via ``LambdaLR``'s multiplicative factor.
    weight_decay
        Decay coefficient for the matmul-weights group. The other
        group always uses ``0.0``.

    Returns
    -------
    A ``torch.optim.AdamW`` with exactly two ``param_groups``.
    """
    decay_params: list[nn.Parameter] = []
    no_decay_params: list[nn.Parameter] = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            # Frozen layers (e.g. an embedding we've explicitly frozen)
            # have no gradient, so AdamW would just allocate unused
            # state for them. Skip.
            continue

        # The standard rule: bias parameters by name, and any 1-D
        # parameter (which catches LayerNorm/RMSNorm scales and any
        # other 1-D learnable vector) go into the no-decay bucket.
        if name.endswith(".bias") or param.ndim == 1:
            no_decay_params.append(param)
        else:
            decay_params.append(param)

    # Two explicit groups. The order isn't load-bearing; tests look up
    # by ``weight_decay`` value rather than index.
    param_groups = [
        {"params": decay_params, "weight_decay": weight_decay},
        {"params": no_decay_params, "weight_decay": 0.0},
    ]
    return torch.optim.AdamW(param_groups, lr=lr)


def build_lr_scheduler(
    optimizer: torch.optim.Optimizer,
    total_steps: int,
    warmup_steps: int,
) -> torch.optim.lr_scheduler.LambdaLR:
    """Cosine LR schedule with linear warmup, as a ``LambdaLR``.

    The returned scheduler multiplies the optimizer's base ``lr`` by a
    factor that depends on the current step:

      step < warmup_steps : factor = step / warmup_steps    (linear ramp 0 -> 1)
      step >= warmup_steps: factor = 0.5 * (1 + cos(pi * progress))
                            where progress = (step - warmup_steps) / decay_steps

    At ``step == warmup_steps`` the cosine branch evaluates to 1.0
    (cos(0) = 1), which is the same value the warmup branch would
    return at its endpoint, so the schedule is continuous across the
    boundary. At ``step == total_steps`` the cosine branch evaluates to
    0.0 exactly (cos(pi) = -1), so the LR decays all the way to zero
    by the final step.

    ``LambdaLR.__init__`` evaluates the lambda once at construction
    time at ``last_epoch=0``, so immediately after this returns the
    optimizer's per-group ``lr`` reflects ``factor(0)`` = 0.

    Parameters
    ----------
    optimizer
        Any ``torch.optim`` optimizer; the scheduler reads its base LRs
        and writes back scaled LRs to each ``param_groups[i]['lr']``.
    total_steps
        Total optimizer steps for the run. Must be > ``warmup_steps``.
    warmup_steps
        Number of optimizer steps over which to linearly ramp from
        zero to peak. Must be > 0 to avoid division by zero in the
        warmup branch.

    Returns
    -------
    A ``torch.optim.lr_scheduler.LambdaLR`` ready for the trainer to
    call ``.step()`` on once per optimizer step.
    """
    # Pulled out of the closure so the math is readable in one place.
    # ``decay_steps`` is the denominator for the cosine progress; if it
    # were 0 the cosine branch would divide by zero, so we require
    # warmup_steps < total_steps. The Config validator already enforces
    # this, but check here too because this factory is also called
    # directly from tests with hand-constructed integers.
    decay_steps = total_steps - warmup_steps

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            # Linear: 0 at step=0, 1 at step=warmup_steps.
            return step / warmup_steps
        # Cosine half-period from 0 to pi: factor goes 1 -> 0 as
        # progress goes 0 -> 1.
        progress = (step - warmup_steps) / decay_steps
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
