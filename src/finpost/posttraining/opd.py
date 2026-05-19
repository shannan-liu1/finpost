"""On-Policy Distillation pair construction.

OPD reuses DPO mechanics, but the preference pairs come from current-policy
rollouts scored by a verifier instead of a fixed offline preference source.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Literal

from finpost.data.schema import Source
from finpost.training.preference_data import DPOPreferenceExample

WeightingMode = Literal["uniform", "adaptive"]


@dataclass(frozen=True)
class OPDRollout:
    """One current-policy completion with a verifier reward."""

    prompt_id: str
    prompt: str
    completion: str
    reward: float
    metadata: dict[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class OPDPair:
    """One DPO-compatible pair produced from on-policy rollouts."""

    prompt_id: str
    prompt: str
    chosen: str
    rejected: str
    chosen_reward: float
    rejected_reward: float
    weight: float = 1.0
    metadata: dict[str, object] = field(default_factory=dict)

    def to_dpo_preference_example(self, *, source: Source) -> DPOPreferenceExample:
        """Convert this OPD pair into the existing DPO data contract."""
        metadata = dict(self.metadata)
        metadata["opd_weight"] = self.weight
        return DPOPreferenceExample(
            prompt=self.prompt,
            chosen=self.chosen,
            rejected=self.rejected,
            source=source,
            prompt_id=self.prompt_id,
            chosen_grade={"reward": self.chosen_reward},
            rejected_grade={"reward": self.rejected_reward},
            metadata=metadata,
        )


def success_bucket(success_rate: float) -> str:
    """Bucket a prompt by current-policy rollout success rate."""
    if not 0.0 <= success_rate <= 1.0:
        raise ValueError("success_rate must be between 0 and 1")
    if success_rate >= 0.8:
        return "easy"
    if success_rate <= 0.2:
        return "hard"
    return "ambiguous"


def adaptive_weight(success_rate: float) -> float:
    """Return the runbook OPD training weight for a success rate bucket."""
    bucket = success_bucket(success_rate)
    if bucket == "easy":
        return 0.25
    if bucket == "hard":
        return 0.5
    return 1.0


def _success_rate(group: list[OPDRollout]) -> float:
    if not group:
        raise ValueError("cannot compute success rate for an empty group")
    successes = sum(1 for rollout in group if rollout.reward > 0.0)
    return successes / len(group)


def _validate_group(group: list[OPDRollout]) -> None:
    prompts = {rollout.prompt for rollout in group}
    if len(prompts) != 1:
        raise ValueError(f"rollouts for prompt_id={group[0].prompt_id!r} disagree on prompt text")


def build_opd_pairs(
    rollouts: list[OPDRollout],
    *,
    weighting: WeightingMode = "uniform",
    min_reward_margin: float = 0.0,
) -> list[OPDPair]:
    """Build one best-vs-worst OPD pair per prompt group.

    Tied groups are skipped because they carry no preference signal.
    """
    if weighting not in ("uniform", "adaptive"):
        raise ValueError("weighting must be 'uniform' or 'adaptive'")
    if min_reward_margin < 0.0:
        raise ValueError("min_reward_margin must be non-negative")

    grouped: dict[str, list[OPDRollout]] = defaultdict(list)
    for rollout in rollouts:
        grouped[rollout.prompt_id].append(rollout)

    pairs: list[OPDPair] = []
    for prompt_id, group in grouped.items():
        if len(group) < 2:
            continue
        _validate_group(group)
        ordered = sorted(group, key=lambda rollout: rollout.reward)
        rejected = ordered[0]
        chosen = ordered[-1]
        reward_margin = chosen.reward - rejected.reward
        if reward_margin <= min_reward_margin:
            continue

        rate = _success_rate(group)
        bucket = success_bucket(rate)
        weight = adaptive_weight(rate) if weighting == "adaptive" else 1.0
        metadata: dict[str, object] = {
            "bucket": bucket,
            "success_rate": rate,
            "group_size": len(group),
            "reward_margin": reward_margin,
        }
        pairs.append(
            OPDPair(
                prompt_id=prompt_id,
                prompt=chosen.prompt,
                chosen=chosen.completion,
                rejected=rejected.completion,
                chosen_reward=chosen.reward,
                rejected_reward=rejected.reward,
                weight=weight,
                metadata=metadata,
            )
        )

    return pairs


def to_dpo_examples(pairs: list[OPDPair], *, source: Source) -> list[DPOPreferenceExample]:
    """Convert OPD pairs into DPO examples for the existing trainer path."""
    return [pair.to_dpo_preference_example(source=source) for pair in pairs]
