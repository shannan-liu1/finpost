"""Group Relative Policy Optimization backend primitives.

The functions here implement the small, testable math core: group-relative
advantages, the KL-controlled token objective, and a logits-to-token-logps
adapter for causal language models.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F

from finpost.training.masking import IGNORE_INDEX


def group_relative_advantages(
    rewards: torch.Tensor,
    *,
    eps: float = 1e-8,
    normalize: bool = True,
) -> torch.Tensor:
    """Return per-prompt relative advantages for grouped completions.

    ``rewards`` has shape ``(prompt_count, group_size)``. Each row is centered
    independently. When ``normalize`` is true, non-degenerate rows are divided
    by their row standard deviation; all-equal rows become zero advantages.
    """
    if rewards.ndim != 2:
        raise ValueError(f"rewards must have shape (prompts, group); got {tuple(rewards.shape)}")
    if rewards.size(1) < 2:
        raise ValueError("GRPO requires at least two completions per prompt group")

    centered = rewards - rewards.mean(dim=-1, keepdim=True)
    if not normalize:
        return centered

    std = rewards.std(dim=-1, unbiased=False, keepdim=True)
    return torch.where(std > eps, centered / std.clamp_min(eps), torch.zeros_like(centered))


def token_log_probs_from_logits(
    logits: torch.Tensor,
    labels: torch.Tensor,
    *,
    ignore_index: int = IGNORE_INDEX,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return per-token log-probs and response mask for causal-LM labels."""
    if logits.ndim != 3:
        raise ValueError(
            f"logits must have shape (batch, seq, vocab); got {tuple(logits.shape)}"
        )
    if labels.ndim != 2:
        raise ValueError(f"labels must have shape (batch, seq); got {tuple(labels.shape)}")
    if logits.shape[:2] != labels.shape:
        raise ValueError(
            "logits and labels batch/sequence dimensions must match: "
            f"{tuple(logits.shape[:2])} vs {tuple(labels.shape)}"
        )
    if logits.size(1) < 2:
        raise ValueError("token_log_probs_from_logits requires sequence length >= 2")

    shifted_logits = logits[:, :-1, :]
    shifted_labels = labels[:, 1:]
    response_mask = shifted_labels != ignore_index
    if (response_mask.sum(dim=-1) == 0).any():
        bad_rows = torch.nonzero(response_mask.sum(dim=-1) == 0, as_tuple=False).flatten()
        raise ValueError(f"GRPO row(s) have no response labels after shifting: {bad_rows.tolist()}")

    safe_labels = shifted_labels.masked_fill(~response_mask, 0)
    token_logps = F.log_softmax(shifted_logits, dim=-1).gather(
        dim=-1,
        index=safe_labels.unsqueeze(-1),
    ).squeeze(-1)
    return token_logps, response_mask


def grpo_token_loss(
    *,
    policy_logps: torch.Tensor,
    old_logps: torch.Tensor,
    ref_logps: torch.Tensor,
    advantages: torch.Tensor,
    response_mask: torch.Tensor,
    beta: float,
    clip_range: float | None = None,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Compute the KL-controlled GRPO token loss.

    ``advantages`` is sequence-level with shape ``(batch,)`` and is broadcast
    across response tokens. ``policy_logps``, ``old_logps``, ``ref_logps``, and
    ``response_mask`` have shape ``(batch, response_seq)``.
    """
    if beta < 0.0:
        raise ValueError("beta must be non-negative")
    if clip_range is not None and clip_range <= 0.0:
        raise ValueError("clip_range must be positive when provided")
    if policy_logps.shape != old_logps.shape or policy_logps.shape != ref_logps.shape:
        raise ValueError("policy, old, and reference log-prob tensors must have the same shape")
    if response_mask.shape != policy_logps.shape:
        raise ValueError("response_mask must match log-prob tensor shape")
    if advantages.shape != (policy_logps.size(0),):
        raise ValueError(
            f"advantages must have shape ({policy_logps.size(0)},); got {tuple(advantages.shape)}"
        )
    if not response_mask.any():
        raise ValueError("response_mask contains no response tokens")

    ratio = torch.exp(policy_logps - old_logps)
    token_advantages = advantages.unsqueeze(-1)
    unclipped_objective = ratio * token_advantages
    if clip_range is not None:
        clipped_ratio = ratio.clamp(1.0 - clip_range, 1.0 + clip_range)
        clipped_objective = clipped_ratio * token_advantages
        policy_objective = torch.minimum(unclipped_objective, clipped_objective)
    else:
        policy_objective = unclipped_objective

    log_ratio_ref_policy = ref_logps - policy_logps
    kl = torch.exp(log_ratio_ref_policy) - log_ratio_ref_policy - 1.0
    token_loss = -(policy_objective - beta * kl)
    mask = response_mask.to(token_loss.dtype)
    loss = (token_loss * mask).sum() / mask.sum()

    detached_mask = mask.detach()
    metrics: dict[str, Any] = {
        "loss": loss.detach(),
        "mean_reward_objective": ((policy_objective.detach() * detached_mask).sum() / mask.sum()),
        "mean_kl": ((kl.detach() * detached_mask).sum() / mask.sum()),
        "mean_advantage": advantages.detach().float().mean(),
        "mean_ratio": ((ratio.detach() * detached_mask).sum() / mask.sum()),
    }
    return loss, metrics


def grpo_loss_from_logits(
    *,
    policy_logits: torch.Tensor,
    labels: torch.Tensor,
    old_logps: torch.Tensor,
    ref_logps: torch.Tensor,
    advantages: torch.Tensor,
    beta: float,
    clip_range: float | None = None,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Convenience wrapper for causal-LM GRPO training loops."""
    policy_logps, response_mask = token_log_probs_from_logits(policy_logits, labels)
    return grpo_token_loss(
        policy_logps=policy_logps,
        old_logps=old_logps,
        ref_logps=ref_logps,
        advantages=advantages,
        response_mask=response_mask,
        beta=beta,
        clip_range=clip_range,
    )
