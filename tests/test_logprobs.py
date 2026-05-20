"""Equivalence tests for the fused-cross-entropy log-prob helper.

These tests exist to prevent silent regressions of the optimization
documented in ``src/finpost/training/logprobs.py``. Two implementations
of the same quantity must agree numerically:

  1. The naive form: ``F.log_softmax(...).gather(...)`` followed by
     masking ignored positions. This is what the codebase used before
     the cross-entropy refactor.
  2. The fused form: ``-F.cross_entropy(..., reduction='none',
     ignore_index=IGNORE_INDEX)`` from
     ``token_log_probs_via_cross_entropy``. This is the current code.

If the two ever disagree, the refactor has broken the math and the
test below fires. The naive reference is kept inline so the test
file is self-contained - readers can compare what the two
implementations are claimed to compute.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from finpost.training.logprobs import token_log_probs_via_cross_entropy
from finpost.training.masking import IGNORE_INDEX


def _reference_token_log_probs(
    logits: torch.Tensor,
    labels: torch.Tensor,
    *,
    ignore_index: int = IGNORE_INDEX,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Naive reference: log_softmax + gather + mask. Slow but obvious.

    Kept in the test file (not in production code) so the assertion
    below has something to compare against.
    """
    shifted_logits = logits[:, :-1, :]
    shifted_labels = labels[:, 1:]
    response_mask = shifted_labels != ignore_index
    # gather has no ignore_index support, so substitute a safe index
    # for ignored positions and then zero them out by multiplying by
    # the mask afterward.
    safe_labels = shifted_labels.masked_fill(~response_mask, 0)
    per_token = (
        F.log_softmax(shifted_logits, dim=-1)
        .gather(dim=-1, index=safe_labels.unsqueeze(-1))
        .squeeze(-1)
    )
    per_token = per_token * response_mask
    return per_token, response_mask


def test_fused_form_matches_naive_log_softmax_gather() -> None:
    """The fused kernel must produce the same per-token log-probs as
    the explicit log_softmax + gather + mask form, up to float
    precision. If this ever fails the refactor has changed semantics.
    """
    torch.manual_seed(0)
    # Realistic causal-LM shape: batch=3, seq=7, vocab=17. Mixed
    # response/prompt/padding labels so ignore_index handling is
    # exercised on every row.
    logits = torch.randn(3, 7, 17)
    labels = torch.tensor(
        [
            [IGNORE_INDEX, IGNORE_INDEX, 3, 5, 9, 11, IGNORE_INDEX],
            [IGNORE_INDEX, 0, 1, 2, IGNORE_INDEX, IGNORE_INDEX, IGNORE_INDEX],
            [IGNORE_INDEX, IGNORE_INDEX, IGNORE_INDEX, 16, 16, 16, 16],
        ]
    )

    fused_logps, fused_mask = token_log_probs_via_cross_entropy(logits, labels)
    ref_logps, ref_mask = _reference_token_log_probs(logits, labels)

    assert torch.equal(fused_mask, ref_mask)
    assert torch.allclose(fused_logps, ref_logps, atol=1e-6, rtol=1e-5)


def test_fused_form_returns_zero_at_ignored_positions() -> None:
    """The cross_entropy kernel returns 0 for ignored targets. The
    helper must surface that - callers (e.g. ``sequence_log_probs``)
    rely on it to skip the mask-multiply step before summing.
    """
    logits = torch.randn(1, 4, 5)
    labels = torch.tensor([[IGNORE_INDEX, 2, IGNORE_INDEX, 3]])

    token_logps, response_mask = token_log_probs_via_cross_entropy(logits, labels)

    # shifted_labels = [2, IGNORE, 3], response_mask = [True, False, True]
    assert response_mask.tolist() == [[True, False, True]]
    # Position 1 (corresponds to shifted_label == IGNORE_INDEX) must be
    # exactly zero, not "approximately zero." The kernel guarantees this.
    assert token_logps[0, 1].item() == 0.0
    # Non-ignored positions should be < 0 (log of a probability).
    assert token_logps[0, 0].item() < 0.0
    assert token_logps[0, 2].item() < 0.0


def test_fused_form_handles_high_confidence_targets() -> None:
    """Worked check from the module docstring at small scale.

    For ``logits = [2.0, 1.0]`` and target ``1``, log-probability is
    ``1.0 - log(e^2 + e^1) ~= -1.3133``.
    """
    # seq=2 -> after shift, seq=1. Place the (2.0, 1.0) at position 0.
    logits = torch.tensor([[[2.0, 1.0], [0.0, 0.0]]])
    labels = torch.tensor([[IGNORE_INDEX, 1]])

    token_logps, _ = token_log_probs_via_cross_entropy(logits, labels)

    expected = 1.0 - torch.tensor([2.0, 1.0]).exp().sum().log()
    assert torch.allclose(token_logps[0, 0], expected, atol=1e-6)


def test_fused_form_rejects_short_sequences() -> None:
    """Sequence length 1 has no valid shifted positions; the helper
    must raise rather than silently return an empty tensor.
    """
    logits = torch.zeros((1, 1, 4))
    labels = torch.zeros((1, 1), dtype=torch.long)

    try:
        token_log_probs_via_cross_entropy(logits, labels)
    except ValueError as exc:
        assert "sequence length >= 2" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("expected sequence length 1 to raise")
