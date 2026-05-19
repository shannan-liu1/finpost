"""Behavior tests for the FinChain local-data loader."""

from __future__ import annotations

import json


def test_load_finchain_jsonl_normalizes_exported_records(tmp_path) -> None:
    """FinChain exports should become common Example records with metadata."""
    from finpost.data.finchain import load_finchain_jsonl

    path = tmp_path / "finchain.jsonl"
    path.write_text(
        json.dumps(
            {
                "id": "compound-interest-0",
                "problem": "What is the ending value?",
                "solution": "Step 1: compute interest.\nFinal Answer: $1,210.00",
                "answer": "$1,210.00",
                "domain": "Personal Finance",
                "topic": "Compound Interest",
                "subtopic": "Annual compounding",
                "level": "intermediate",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    examples = load_finchain_jsonl(path, split="train")

    assert len(examples) == 1
    assert examples[0].id == "compound-interest-0"
    assert examples[0].source == "finchain"
    assert examples[0].prompt == "What is the ending value?"
    assert examples[0].final_answer == "1210.00"
    assert examples[0].difficulty == 3
    assert examples[0].domain == "Personal Finance"
    assert examples[0].topic == "Compound Interest"
    assert examples[0].subtopic == "Annual compounding"


def test_finchain_loader_extracts_answer_from_solution_when_answer_missing(tmp_path) -> None:
    """A generated template row can omit answer if the trace contains a final marker."""
    from finpost.data.finchain import load_finchain_jsonl

    path = tmp_path / "finchain.jsonl"
    path.write_text(
        json.dumps(
            {
                "question": "What is the ratio?",
                "reasoning": "Step 1: Divide assets by liabilities.\nAnswer: 1.25",
                "topic": "Financial Ratios",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    examples = load_finchain_jsonl(path, split="test")

    assert examples[0].id == "finchain-test-0"
    assert examples[0].response.startswith("Step 1")
    assert examples[0].final_answer == "1.25"
    assert examples[0].topic == "Financial Ratios"
