"""Behavior tests for Direct Preference Optimization math."""

from __future__ import annotations

import torch

from finpost.training.masking import IGNORE_INDEX


def test_compute_dpo_loss_matches_closed_form_and_metrics() -> None:
    """DPO optimizes the policy-vs-reference preference margin."""
    from finpost.training.dpo import compute_dpo_loss

    policy_chosen = torch.tensor([-1.0, -2.0])
    policy_rejected = torch.tensor([-2.0, -3.5])
    ref_chosen = torch.tensor([-1.5, -2.5])
    ref_rejected = torch.tensor([-2.0, -3.0])

    loss, metrics = compute_dpo_loss(
        policy_chosen_logps=policy_chosen,
        policy_rejected_logps=policy_rejected,
        ref_chosen_logps=ref_chosen,
        ref_rejected_logps=ref_rejected,
        beta=0.2,
    )

    policy_margin = policy_chosen - policy_rejected
    reference_margin = ref_chosen - ref_rejected
    expected = -torch.nn.functional.logsigmoid(0.2 * (policy_margin - reference_margin)).mean()

    assert loss == expected
    assert metrics["preference_accuracy"] == torch.tensor(1.0)
    assert metrics["policy_margin"] == torch.tensor(1.25)
    assert metrics["reference_margin"] == torch.tensor(0.5)


def test_dpo_loss_penalizes_preference_reversal_more_than_correct_order() -> None:
    """A policy that prefers rejected answers should have higher loss."""
    from finpost.training.dpo import compute_dpo_loss

    correct_loss, _ = compute_dpo_loss(
        policy_chosen_logps=torch.tensor([-1.0]),
        policy_rejected_logps=torch.tensor([-3.0]),
        ref_chosen_logps=torch.tensor([-2.0]),
        ref_rejected_logps=torch.tensor([-2.0]),
        beta=0.1,
    )
    reversed_loss, _ = compute_dpo_loss(
        policy_chosen_logps=torch.tensor([-3.0]),
        policy_rejected_logps=torch.tensor([-1.0]),
        ref_chosen_logps=torch.tensor([-2.0]),
        ref_rejected_logps=torch.tensor([-2.0]),
        beta=0.1,
    )

    assert reversed_loss > correct_loss


def test_sequence_log_probs_only_counts_response_targets() -> None:
    """Prompt labels and padding labels are excluded after LM shifting."""
    from finpost.training.dpo import sequence_log_probs

    # Shape: batch=1, seq=5, vocab=4. Each shifted label has a deliberately
    # high logit at the target token, but prompt/pad labels should not count.
    logits = torch.zeros((1, 5, 4), dtype=torch.float32)
    logits[0, 1, 2] = 4.0
    logits[0, 2, 3] = 4.0
    labels = torch.tensor([[IGNORE_INDEX, IGNORE_INDEX, 2, 3, IGNORE_INDEX]])

    logps, counts = sequence_log_probs(logits, labels)

    token_2_logp = torch.log_softmax(logits[0, 1], dim=-1)[2]
    token_3_logp = torch.log_softmax(logits[0, 2], dim=-1)[3]
    assert logps == (token_2_logp + token_3_logp).unsqueeze(0)
    assert counts.tolist() == [2]


def test_sequence_log_probs_rejects_empty_response_rows() -> None:
    """Preference rows with no response labels are invalid training examples."""
    from finpost.training.dpo import sequence_log_probs

    logits = torch.zeros((1, 3, 8), dtype=torch.float32)
    labels = torch.full((1, 3), IGNORE_INDEX, dtype=torch.long)

    try:
        sequence_log_probs(logits, labels)
    except ValueError as exc:
        assert "no response labels" in str(exc)
    else:  # pragma: no cover - keeps the assertion message readable
        raise AssertionError("expected empty response labels to raise")
