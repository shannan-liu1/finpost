"""Regression tests for numerical stability of compute_masked_ce_loss.

Background: training Qwen2.5-0.5B in bf16 at batch=16, seq=1024 on A40
diverged to NaN within 20 steps because cross_entropy was being called
on bf16 logits over a 152k-token vocabulary. The softmax tail
probabilities underflowed in bf16, log(0) = -inf, loss = NaN.

The fix in compute_masked_ce_loss is to upcast logits to fp32 before
cross_entropy. These tests guard against reintroducing the bug.
"""

from __future__ import annotations

import torch

from finpost.training.masking import IGNORE_INDEX
from finpost.training.sft import compute_masked_ce_loss


def test_loss_finite_with_bf16_logits_and_large_vocab() -> None:
    """Loss must stay finite when logits are bf16 over a Qwen-scale vocab.

    Without the fp32 upcast, this combination triggers bf16 softmax
    underflow on a non-trivial fraction of batches.
    """
    torch.manual_seed(0)
    # Smaller than the real (16, 1024, 152064) to keep the test cheap,
    # but the vocab dim is what drives the underflow — keep that close to
    # production. 50k is large enough to reliably trigger the bug pre-fix.
    batch, seq_len, vocab = 4, 128, 50_000

    # Disperse the logits so the softmax tails are very small (this is
    # what underflows in bf16).
    logits = (torch.randn(batch, seq_len, vocab) * 10.0).to(torch.bfloat16)

    labels = torch.randint(0, vocab, (batch, seq_len))
    # Mask half the positions as prompt (the realistic SFT pattern).
    labels[:, : seq_len // 2] = IGNORE_INDEX

    loss = compute_masked_ce_loss(logits, labels)

    assert torch.isfinite(loss), (
        f"Loss must be finite even with bf16 logits + large vocab, got {loss!r}. "
        "This indicates the fp32 upcast in compute_masked_ce_loss is missing."
    )


def test_loss_is_fp32_even_when_logits_are_bf16() -> None:
    """The loss tensor returned must be fp32 regardless of input dtype.

    The upcast happens inside compute_masked_ce_loss; callers should not
    receive a bf16 loss back (which would propagate the precision issue
    into subsequent .item() / .backward() / accumulation paths).
    """
    torch.manual_seed(0)
    batch, seq_len, vocab = 2, 32, 1000
    logits = torch.randn(batch, seq_len, vocab).to(torch.bfloat16)
    labels = torch.randint(0, vocab, (batch, seq_len))

    loss = compute_masked_ce_loss(logits, labels)

    assert loss.dtype == torch.float32, (
        f"Loss must be fp32 after the bf16-stability upcast, got {loss.dtype}. "
        "compute_masked_ce_loss should call .float() on logits before cross_entropy."
    )


def test_loss_finite_with_fp32_logits_unchanged() -> None:
    """Sanity check: fp32 logits still produce finite loss after the fix."""
    torch.manual_seed(0)
    batch, seq_len, vocab = 2, 32, 1000
    logits = torch.randn(batch, seq_len, vocab, dtype=torch.float32)
    labels = torch.randint(0, vocab, (batch, seq_len))

    loss = compute_masked_ce_loss(logits, labels)

    assert torch.isfinite(loss)
    assert loss.dtype == torch.float32
