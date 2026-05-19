"""Behavior tests for OPD pair construction."""

from __future__ import annotations


def test_build_opd_pairs_chooses_best_over_worst_per_prompt() -> None:
    """OPD should turn current-policy verified rollouts into preference pairs."""
    from finpost.posttraining.opd import OPDRollout, build_opd_pairs

    rollouts = [
        OPDRollout(prompt_id="p0", prompt="Q?", completion="bad", reward=0.0),
        OPDRollout(prompt_id="p0", prompt="Q?", completion="good", reward=1.0),
        OPDRollout(prompt_id="p1", prompt="Q2?", completion="tie-a", reward=0.5),
        OPDRollout(prompt_id="p1", prompt="Q2?", completion="tie-b", reward=0.5),
    ]

    pairs = build_opd_pairs(rollouts)

    assert len(pairs) == 1
    assert pairs[0].prompt_id == "p0"
    assert pairs[0].chosen == "good"
    assert pairs[0].rejected == "bad"
    assert pairs[0].chosen_reward == 1.0
    assert pairs[0].rejected_reward == 0.0


def test_build_opd_pairs_uses_adaptive_difficulty_weights() -> None:
    """Ambiguous prompt groups should carry more training weight than easy groups."""
    from finpost.posttraining.opd import OPDRollout, build_opd_pairs

    easy_rollouts = [
        OPDRollout(prompt_id="easy", prompt="E?", completion="ok1", reward=1.0),
        OPDRollout(prompt_id="easy", prompt="E?", completion="ok2", reward=1.0),
        OPDRollout(prompt_id="easy", prompt="E?", completion="bad", reward=0.0),
        OPDRollout(prompt_id="easy", prompt="E?", completion="ok3", reward=1.0),
        OPDRollout(prompt_id="easy", prompt="E?", completion="ok4", reward=1.0),
    ]
    ambiguous_rollouts = [
        OPDRollout(prompt_id="amb", prompt="A?", completion="ok1", reward=1.0),
        OPDRollout(prompt_id="amb", prompt="A?", completion="bad1", reward=0.0),
        OPDRollout(prompt_id="amb", prompt="A?", completion="ok2", reward=1.0),
        OPDRollout(prompt_id="amb", prompt="A?", completion="bad2", reward=0.0),
    ]

    pairs = build_opd_pairs(easy_rollouts + ambiguous_rollouts, weighting="adaptive")
    by_id = {pair.prompt_id: pair for pair in pairs}

    assert by_id["amb"].weight > by_id["easy"].weight
    assert by_id["amb"].metadata["bucket"] == "ambiguous"
    assert by_id["easy"].metadata["bucket"] == "easy"


def test_opd_pair_converts_to_dpo_preference_example() -> None:
    """OPD should reuse the existing DPO training data contract."""
    from finpost.posttraining.opd import OPDPair

    pair = OPDPair(
        prompt_id="p0",
        prompt="Q?",
        chosen="good",
        rejected="bad",
        chosen_reward=1.0,
        rejected_reward=0.0,
        weight=0.5,
        metadata={"bucket": "hard"},
    )

    dpo_example = pair.to_dpo_preference_example(source="finchain")

    assert dpo_example.prompt == "Q?"
    assert dpo_example.chosen == "good"
    assert dpo_example.rejected == "bad"
    assert dpo_example.source == "finchain"
    assert dpo_example.metadata["opd_weight"] == 0.5
    assert dpo_example.chosen_grade == {"reward": 1.0}
    assert dpo_example.rejected_grade == {"reward": 0.0}
