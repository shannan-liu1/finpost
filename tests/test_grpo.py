"""Behavior tests for GRPO backend primitives."""

from __future__ import annotations

import torch


def test_group_relative_advantages_are_centered_per_prompt() -> None:
    """GRPO compares completions within each prompt group."""
    from finpost.posttraining.grpo import group_relative_advantages

    rewards = torch.tensor([[1.0, 2.0, 3.0], [5.0, 5.0, 5.0]])

    advantages = group_relative_advantages(rewards)

    assert torch.allclose(advantages[0].mean(), torch.tensor(0.0), atol=1e-6)
    assert torch.allclose(advantages[0].std(unbiased=False), torch.tensor(1.0), atol=1e-6)
    assert advantages[1].tolist() == [0.0, 0.0, 0.0]


def test_grpo_token_loss_prefers_positive_advantage_tokens() -> None:
    """Increasing probability on a positive-advantage sample should lower loss."""
    from finpost.posttraining.grpo import grpo_token_loss

    mask = torch.ones((2, 2), dtype=torch.bool)
    old_logps = torch.zeros((2, 2))
    ref_logps = torch.zeros((2, 2))
    advantages = torch.tensor([1.0, -1.0])
    neutral_policy = torch.zeros((2, 2), requires_grad=True)
    improved_policy = torch.tensor([[0.2, 0.2], [-0.2, -0.2]], requires_grad=True)

    neutral_loss, neutral_metrics = grpo_token_loss(
        policy_logps=neutral_policy,
        old_logps=old_logps,
        ref_logps=ref_logps,
        advantages=advantages,
        response_mask=mask,
        beta=0.0,
    )
    improved_loss, improved_metrics = grpo_token_loss(
        policy_logps=improved_policy,
        old_logps=old_logps,
        ref_logps=ref_logps,
        advantages=advantages,
        response_mask=mask,
        beta=0.0,
    )

    assert improved_loss < neutral_loss
    assert neutral_metrics["mean_kl"] == torch.tensor(0.0)
    assert improved_metrics["mean_advantage"] == torch.tensor(0.0)


def test_grpo_token_loss_penalizes_reference_drift() -> None:
    """The Schulman KL approximator should raise loss when policy drifts."""
    from finpost.posttraining.grpo import grpo_token_loss

    policy_logps = torch.full((1, 3), -0.2, requires_grad=True)
    old_logps = torch.full((1, 3), -0.2)
    ref_logps = torch.zeros((1, 3))
    advantages = torch.tensor([0.0])
    mask = torch.ones((1, 3), dtype=torch.bool)

    loss, metrics = grpo_token_loss(
        policy_logps=policy_logps,
        old_logps=old_logps,
        ref_logps=ref_logps,
        advantages=advantages,
        response_mask=mask,
        beta=0.1,
    )

    assert loss > 0
    assert metrics["mean_kl"] > 0
    assert loss.requires_grad
