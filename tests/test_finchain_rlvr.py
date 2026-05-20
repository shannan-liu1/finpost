"""Behavior tests for FinChain RLVR trainer adapters."""

from __future__ import annotations


def test_build_finchain_prompt_rows_formats_prompt_and_keeps_gold_answer() -> None:
    """Industry trainers need prompt rows while the verifier still needs gold answers."""
    from finpost.data.schema import Example
    from finpost.posttraining.finchain_rlvr import build_finchain_prompt_rows

    rows = build_finchain_prompt_rows(
        [
            Example(
                id="finchain-train-0",
                source="finchain",
                prompt="What is revenue minus cost?",
                response="Final Answer: 7",
                final_answer="7",
                topic="income_statement",
            )
        ]
    )

    assert rows[0]["prompt"].startswith("<|im_start|>user\n")
    assert rows[0]["prompt"].endswith("<|im_start|>assistant\n")
    assert rows[0]["raw_prompt"] == "What is revenue minus cost?"
    assert rows[0]["gold_answer"] == "7"
    assert rows[0]["prompt_id"] == "finchain-train-0"
    assert rows[0]["topic"] == "income_statement"


def test_finchain_binary_rewards_scores_strings_and_chat_completions() -> None:
    """The same verifier reward should work for TRL raw and conversational outputs."""
    from finpost.posttraining.finchain_rlvr import finchain_binary_rewards

    rewards = finchain_binary_rewards(
        [
            "Reasoning...\nFinal Answer: 42",
            [{"role": "assistant", "content": "Reasoning...\nFinal Answer: 41"}],
        ],
        gold_answer=["42", "42"],
    )

    assert rewards == [1.0, 0.0]


def test_finchain_binary_rewards_accepts_online_dpo_positional_shape() -> None:
    """OnlineDPO-style hooks may call reward functions with prompts first."""
    from finpost.posttraining.finchain_rlvr import finchain_binary_rewards

    rewards = finchain_binary_rewards(
        ["prompt 1", "prompt 2"],
        ["Final Answer: 10", "Final Answer: 11"],
        final_answer=["10", "10"],
    )

    assert rewards == [1.0, 0.0]


def test_deterministic_sample_is_reproducible_and_non_mutating() -> None:
    """RunPod prompt exports should be reproducible from the manifest seed."""
    from finpost.data.schema import Example
    from finpost.posttraining.finchain_rlvr import deterministic_sample

    examples = [
        Example(
            id=f"p{idx}",
            source="finchain",
            prompt=f"prompt {idx}",
            response=f"response {idx}",
            final_answer=str(idx),
        )
        for idx in range(5)
    ]

    first = deterministic_sample(examples, n=3, seed=13)
    second = deterministic_sample(examples, n=3, seed=13)

    assert [example.id for example in first] == [example.id for example in second]
    assert [example.id for example in examples] == ["p0", "p1", "p2", "p3", "p4"]
