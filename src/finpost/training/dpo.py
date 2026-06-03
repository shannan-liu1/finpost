"""Direct Preference Optimization loss primitives."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F

from finpost.training.logprobs import token_log_probs_via_cross_entropy
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
    predicts token ``t`` from logits at position ``t - 1``, so both
    tensors are shifted internally before computing log-probabilities.

    The per-token computation is delegated to
    ``finpost.training.logprobs.token_log_probs_via_cross_entropy`` -
    see that module's docstring for the derivation of why
    ``F.cross_entropy`` is the right primitive here (it computes the
    same quantity as ``log_softmax + gather`` without materializing
    the full ``log_softmax`` tensor, which would be ~1.24 GB per
    forward at Qwen3-4B scale).
    """
    token_logps, response_mask = token_log_probs_via_cross_entropy(
        logits, labels, ignore_index=ignore_index
    )
    counts = response_mask.sum(dim=-1)
    if (counts == 0).any():
        bad_rows = torch.nonzero(counts == 0, as_tuple=False).flatten().tolist()
        raise ValueError(
            f"DPO batch row(s) have no response labels after shifting: {bad_rows}"
        )

    # ``token_logps`` is exactly 0 at ignored positions (the cross-entropy
    # kernel handles ignore_index for us), so we can sum directly - no
    # separate mask multiplication is needed.
    return token_logps.sum(dim=-1), counts


def compute_dpo_loss(
    *,
    policy_chosen_logps: torch.Tensor,
    policy_rejected_logps: torch.Tensor,
    ref_chosen_logps: torch.Tensor,
    ref_rejected_logps: torch.Tensor,
    beta: float,
    sample_weights: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Compute the sigmoid DPO objective and scalar diagnostics."""
    if beta <= 0.0:
        raise ValueError("beta must be positive")

    policy_margin = policy_chosen_logps - policy_rejected_logps
    reference_margin = ref_chosen_logps - ref_rejected_logps
    logits = beta * (policy_margin - reference_margin)
    per_pair_loss = -F.logsigmoid(logits)

    weights: torch.Tensor | None = None
    if sample_weights is not None:
        if sample_weights.shape != per_pair_loss.shape:
            raise ValueError(
                "sample_weights must match the DPO batch shape; got "
                f"{tuple(sample_weights.shape)} vs {tuple(per_pair_loss.shape)}"
            )
        weights = sample_weights.to(device=per_pair_loss.device, dtype=per_pair_loss.dtype)
        if (weights <= 0).any():
            raise ValueError("sample_weights must be strictly positive")
        loss = (per_pair_loss * weights).sum() / weights.sum()
    else:
        loss = per_pair_loss.mean()

    detached_logits = logits.detach()
    preference_hits = (detached_logits > 0).float()
    if weights is None:
        preference_accuracy = preference_hits.mean()
    else:
        detached_weights = weights.detach()
        preference_accuracy = (preference_hits * detached_weights).sum() / detached_weights.sum()
    metrics: dict[str, Any] = {
        "loss": loss.detach(),
        "preference_accuracy": preference_accuracy,
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
    if weights is not None:
        metrics["mean_pair_weight"] = weights.detach().mean()
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
    sample_weights: torch.Tensor | None = None,
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
        sample_weights=sample_weights,
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
    sample_weights: torch.Tensor | None = None,
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
        sample_weights=sample_weights,
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
    sample_weights: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """DPO loss from already-summed sequence log-probabilities."""
    loss, metrics = compute_dpo_loss(
        policy_chosen_logps=policy_chosen_logps,
        policy_rejected_logps=policy_rejected_logps,
        ref_chosen_logps=ref_chosen_logps,
        ref_rejected_logps=ref_rejected_logps,
        beta=beta,
        sample_weights=sample_weights,
    )
    metrics["chosen_response_tokens"] = chosen_counts.detach().float().mean()
    metrics["rejected_response_tokens"] = rejected_counts.detach().float().mean()
    return loss, metrics
