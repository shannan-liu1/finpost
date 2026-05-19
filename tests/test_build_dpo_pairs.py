"""Behavior tests for offline DPO pair construction."""

from __future__ import annotations


def test_build_pairs_from_completions_pairs_correct_against_incorrect() -> None:
    """Each prompt yields chosen/rejected pairs only when both sides exist."""
    from scripts.build_dpo_pairs import CompletionRecord, build_pairs_from_completions

    records = [
        CompletionRecord(
            prompt_id="gsm8k-train-0",
            prompt="1+1?",
            source="gsm8k",
            gold_answer="2",
            sample_index=0,
            completion="#### 2",
            predicted_answer="2",
            correct=True,
        ),
        CompletionRecord(
            prompt_id="gsm8k-train-0",
            prompt="1+1?",
            source="gsm8k",
            gold_answer="2",
            sample_index=1,
            completion="#### 3",
            predicted_answer="3",
            correct=False,
        ),
        CompletionRecord(
            prompt_id="gsm8k-train-0",
            prompt="1+1?",
            source="gsm8k",
            gold_answer="2",
            sample_index=2,
            completion="Reasoning. #### 2",
            predicted_answer="2",
            correct=True,
        ),
        CompletionRecord(
            prompt_id="math-train-0",
            prompt="x?",
            source="math",
            gold_answer="x",
            sample_index=0,
            completion="\\boxed{x}",
            predicted_answer="x",
            correct=True,
        ),
    ]

    pairs = build_pairs_from_completions(records)

    assert len(pairs) == 2
    assert [pair["chosen"] for pair in pairs] == ["#### 2", "Reasoning. #### 2"]
    assert [pair["rejected"] for pair in pairs] == ["#### 3", "#### 3"]
    assert pairs[0]["chosen_grade"]["sample_index"] == 0
    assert pairs[0]["rejected_grade"]["sample_index"] == 1
    assert pairs[0]["metadata"]["gold_answer"] == "2"


def test_build_pairs_from_completions_caps_pairs_deterministically() -> None:
    """A per-prompt cap controls pair explosion without losing reproducibility."""
    from scripts.build_dpo_pairs import CompletionRecord, build_pairs_from_completions

    records = [
        CompletionRecord(
            prompt_id="p0",
            prompt="q",
            source="gsm8k",
            gold_answer="a",
            sample_index=idx,
            completion=f"completion {idx}",
            predicted_answer="a" if idx < 3 else "b",
            correct=idx < 3,
        )
        for idx in range(6)
    ]

    first = build_pairs_from_completions(records, max_pairs_per_prompt=4, seed=13)
    second = build_pairs_from_completions(records, max_pairs_per_prompt=4, seed=13)

    assert len(first) == 4
    assert first == second


def test_summarize_completion_groups_counts_pairability() -> None:
    """The manifest exposes all-correct and all-incorrect groups."""
    from scripts.build_dpo_pairs import CompletionRecord, summarize_completion_groups

    records = [
        CompletionRecord("p0", "q", "gsm8k", "a", 0, "a", "a", True),
        CompletionRecord("p0", "q", "gsm8k", "a", 1, "b", "b", False),
        CompletionRecord("p1", "q", "gsm8k", "a", 0, "a", "a", True),
        CompletionRecord("p1", "q", "gsm8k", "a", 1, "a", "a", True),
        CompletionRecord("p2", "q", "math", "x", 0, "y", "y", False),
    ]

    assert summarize_completion_groups(records) == {
        "prompt_count": 3,
        "pairable_prompt_count": 1,
        "all_correct_prompt_count": 1,
        "all_incorrect_prompt_count": 1,
        "empty_prompt_count": 0,
    }
