"""Supervised Fine-Tuning loss and training step.

The masked cross-entropy loss is computed explicitly here rather than
relying on the model's built-in ``outputs.loss`` (which Hugging Face
models compute internally when you pass ``labels=`` to forward).
Computing it ourselves makes the shift-by-one for next-token prediction
and the label-masking visible line by line — which is the entire point
of this trainer being from-scratch.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from finpost.training.masking import IGNORE_INDEX


def compute_masked_ce_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
) -> torch.Tensor:
    """Masked cross-entropy loss for next-token prediction.

    Causal language modeling shifts: the logit at position ``t`` is the
    distribution predicting the token at position ``t+1``. So we align
    by slicing:

        shift_logits = logits[:, :-1, :]   # positions 0..T-2 predict 1..T-1
        shift_labels = labels[:, 1:]       # the next token at each position

    Then we flatten across batch and time and call ``F.cross_entropy``
    with ``ignore_index=IGNORE_INDEX``. The ignore-index argument
    excludes those positions from BOTH the numerator (loss sum) and the
    denominator (count for the mean), so prompt tokens contribute
    nothing to the loss.

    Parameters
    ----------
    logits
        Shape ``(batch, seq_len, vocab_size)``. Raw model output before softmax.
    labels
        Shape ``(batch, seq_len)``. Target token IDs, with prompt and
        padding positions set to ``IGNORE_INDEX``.

    Returns
    -------
    A scalar tensor: the mean cross-entropy loss over all non-ignored
    response positions.
    """
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()

    # Flatten so cross_entropy treats every (batch, position) as an
    # independent classification problem over the vocabulary.
    flat_logits = shift_logits.view(-1, shift_logits.size(-1))
    flat_labels = shift_labels.view(-1)

    return F.cross_entropy(
        flat_logits,
        flat_labels,
        ignore_index=IGNORE_INDEX,
        reduction="mean",
    )


def train_step(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    labels: torch.Tensor,
    optimizer: torch.optim.Optimizer,
) -> float:
    """Run one SFT training step: forward, loss, backward, optimize.

    Returns the loss as a Python float for logging. The forward pass
    deliberately does NOT pass ``labels=`` to the model, so the model
    returns logits without computing its own internal loss — we want
    our explicit ``compute_masked_ce_loss`` to be the only loss path.
    """
    optimizer.zero_grad()
    outputs = model(input_ids=input_ids)
    loss = compute_masked_ce_loss(outputs.logits, labels)
    loss.backward()
    optimizer.step()
    return loss.item()
