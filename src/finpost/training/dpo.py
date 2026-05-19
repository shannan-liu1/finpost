"""Direct Preference Optimization loss primitives."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F

from finpost.training.masking import IGNORE_INDEX


def sequence_log_probs(
    logits: torch.Tensor,
    labels: torch.Tensor,
    *,
    ignore_index: int = IGNORE_INDEX,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return summed log-probabilities over non-ignored response labels.

    ``labels`` follows the causal-LM convention used by the SFT trainer:
    prompt and padding positions are set to ``IGNORE_INDEX``. The model
    predicts token ``t`` from logits at position ``t - 1``, so both tensors
    are shifted before gathering token log-probabilities.
    """
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
        raise ValueError("sequence_log_probs requires sequence length >= 2")

    shifted_logits = logits[:, :-1, :]
    shifted_labels = labels[:, 1:]
    response_mask = shifted_labels != ignore_index
    counts = response_mask.sum(dim=-1)
    if (counts == 0).any():
        bad_rows = torch.nonzero(counts == 0, as_tuple=False).flatten().tolist()
        raise ValueError(
            f"DPO batch row(s) have no response labels after shifting: {bad_rows}"
        )

    safe_labels = shifted_labels.masked_fill(~response_mask, 0)
    token_logps = F.log_softmax(shifted_logits, dim=-1).gather(
        dim=-1,
        index=safe_labels.unsqueeze(-1),
    ).squeeze(-1)
    return (token_logps * response_mask).sum(dim=-1), counts


def compute_dpo_loss(
    *,
    policy_chosen_logps: torch.Tensor,
    policy_rejected_logps: torch.Tensor,
    ref_chosen_logps: torch.Tensor,
    ref_rejected_logps: torch.Tensor,
    beta: float,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Compute the sigmoid DPO objective and scalar diagnostics."""
    if beta <= 0.0:
        raise ValueError("beta must be positive")

    policy_margin = policy_chosen_logps - policy_rejected_logps
    reference_margin = ref_chosen_logps - ref_rejected_logps
    logits = beta * (policy_margin - reference_margin)
    loss = -F.logsigmoid(logits).mean()

    detached_logits = logits.detach()
    metrics: dict[str, Any] = {
        "loss": loss.detach(),
        "preference_accuracy": (detached_logits > 0).float().mean(),
        "policy_margin": policy_margin.detach().mean(),
        "reference_margin": reference_margin.detach().mean(),
        "reward_margin": detached_logits.mean(),
        "chosen_reward": (
            beta * (policy_chosen_logps - ref_chosen_logps)
        ).detach().mean(),
        "rejected_reward": (
            beta * (policy_rejected_logps - ref_rejected_logps)
        ).detach().mean(),
    }
    return loss, metrics


def dpo_loss_from_logits(
    *,
    policy_chosen_logits: torch.Tensor,
    policy_rejected_logits: torch.Tensor,
    ref_chosen_logits: torch.Tensor,
    ref_rejected_logits: torch.Tensor,
    chosen_labels: torch.Tensor,
    rejected_labels: torch.Tensor,
    beta: float,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Convenience wrapper for the common trainer path."""
    policy_chosen_logps, chosen_counts = sequence_log_probs(policy_chosen_logits, chosen_labels)
    policy_rejected_logps, rejected_counts = sequence_log_probs(
        policy_rejected_logits,
        rejected_labels,
    )
    ref_chosen_logps, _ = sequence_log_probs(ref_chosen_logits, chosen_labels)
    ref_rejected_logps, _ = sequence_log_probs(ref_rejected_logits, rejected_labels)

    return dpo_loss_from_logps(
        policy_chosen_logps=policy_chosen_logps,
        policy_rejected_logps=policy_rejected_logps,
        ref_chosen_logps=ref_chosen_logps,
        ref_rejected_logps=ref_rejected_logps,
        chosen_counts=chosen_counts,
        rejected_counts=rejected_counts,
        beta=beta,
    )


def dpo_loss_from_policy_logits(
    *,
    policy_chosen_logits: torch.Tensor,
    policy_rejected_logits: torch.Tensor,
    chosen_labels: torch.Tensor,
    rejected_labels: torch.Tensor,
    ref_chosen_logps: torch.Tensor,
    ref_rejected_logps: torch.Tensor,
    beta: float,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """DPO loss when reference sequence log-probs were precomputed."""
    policy_chosen_logps, chosen_counts = sequence_log_probs(policy_chosen_logits, chosen_labels)
    policy_rejected_logps, rejected_counts = sequence_log_probs(
        policy_rejected_logits,
        rejected_labels,
    )
    return dpo_loss_from_logps(
        policy_chosen_logps=policy_chosen_logps,
        policy_rejected_logps=policy_rejected_logps,
        ref_chosen_logps=ref_chosen_logps,
        ref_rejected_logps=ref_rejected_logps,
        chosen_counts=chosen_counts,
        rejected_counts=rejected_counts,
        beta=beta,
    )


def dpo_loss_from_logps(
    *,
    policy_chosen_logps: torch.Tensor,
    policy_rejected_logps: torch.Tensor,
    ref_chosen_logps: torch.Tensor,
    ref_rejected_logps: torch.Tensor,
    chosen_counts: torch.Tensor,
    rejected_counts: torch.Tensor,
    beta: float,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """DPO loss from already-summed sequence log-probabilities."""
    loss, metrics = compute_dpo_loss(
        policy_chosen_logps=policy_chosen_logps,
        policy_rejected_logps=policy_rejected_logps,
        ref_chosen_logps=ref_chosen_logps,
        ref_rejected_logps=ref_rejected_logps,
        beta=beta,
    )
    metrics["chosen_response_tokens"] = chosen_counts.detach().float().mean()
    metrics["rejected_response_tokens"] = rejected_counts.detach().float().mean()
    return loss, metrics
