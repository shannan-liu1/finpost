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


def test_resolve_max_new_tokens_by_source_overrides_global_budget() -> None:
    """Pair generation should avoid spending MATH-sized token budgets on GSM8K."""
    from scripts.build_dpo_pairs import resolve_max_new_tokens_by_source

    assert resolve_max_new_tokens_by_source(
        sources=["gsm8k", "math", "finchain"],
        max_new_tokens=768,
        max_new_tokens_gsm8k=256,
        max_new_tokens_math=None,
        max_new_tokens_finchain=512,
    ) == {"gsm8k": 256, "math": 768, "finchain": 512}


def test_shard_examples_assigns_disjoint_deterministic_prompt_slices() -> None:
    """Parallel rollout workers should sample disjoint prompt shards."""
    from finpost.data.schema import Example
    from scripts.build_dpo_pairs import shard_examples

    examples = [
        Example(
            id=f"p{idx}",
            source="finchain",
            prompt=f"prompt {idx}",
            response=f"response {idx}",
            final_answer=str(idx),
        )
        for idx in range(7)
    ]

    shard_0 = shard_examples(examples, shard_id=0, num_shards=2)
    shard_1 = shard_examples(examples, shard_id=1, num_shards=2)

    assert [example.id for example in shard_0] == ["p0", "p2", "p4", "p6"]
    assert [example.id for example in shard_1] == ["p1", "p3", "p5"]
    assert {example.id for example in shard_0}.isdisjoint(
        {example.id for example in shard_1}
    )


def test_select_train_prompts_can_load_finchain(monkeypatch) -> None:
    """The offline comparator can build preference pairs from FinChain train prompts."""
    import scripts.build_dpo_pairs as pair_builder
    from finpost.data.schema import Example

    monkeypatch.setattr(
        pair_builder,
        "load_finchain",
        lambda split="train": [
            Example(
                id=f"finchain-{split}-{idx}",
                source="finchain",
                prompt=f"prompt {idx}",
                response=f"response {idx}",
                final_answer=str(idx),
            )
            for idx in range(4)
        ],
    )

    selected = pair_builder.select_train_prompts(
        sources=["finchain"],
        heldout_train_n=3,
        seed=123,
    )

    assert len(selected) == 3
    assert {example.source for example in selected} == {"finchain"}


def test_merge_dpo_pair_shards_combines_full_shard_set(tmp_path) -> None:
    """Shard outputs should merge back into single-run artifact shape."""
    import json

    from scripts.merge_dpo_pair_shards import merge_shards

    shard_dirs = []
    for shard_id in range(2):
        shard_dir = tmp_path / f"shard-{shard_id:02d}-of-02"
        shard_dir.mkdir()
        completion = {
            "prompt_id": f"p{shard_id}",
            "prompt": f"prompt {shard_id}",
            "source": "finchain",
            "gold_answer": str(shard_id),
            "sample_index": 0,
            "completion": f"Final Answer: {shard_id}",
            "predicted_answer": str(shard_id),
            "correct": True,
        }
        pair = {
            "prompt": f"prompt {shard_id}",
            "chosen": f"Final Answer: {shard_id}",
            "rejected": "Final Answer: 999",
            "source": "finchain",
            "prompt_id": f"p{shard_id}",
            "chosen_grade": {"correct": True, "sample_index": 0},
            "rejected_grade": {"correct": False, "sample_index": 1},
            "metadata": {"gold_answer": str(shard_id)},
        }
        (shard_dir / "completions.jsonl").write_text(
            json.dumps(completion) + "\n",
            encoding="utf-8",
        )
        (shard_dir / "pairs.jsonl").write_text(
            json.dumps(pair) + "\n",
            encoding="utf-8",
        )
        (shard_dir / "manifest.json").write_text(
            json.dumps(
                {
                    "model_checkpoint": "model",
                    "verifier": "verifier",
                    "sources": ["finchain"],
                    "heldout_train_n": 2,
                    "shard_id": shard_id,
                    "num_shards": 2,
                    "samples_per_prompt": 2,
                    "generation_batch_size": 4,
                    "max_new_tokens": 128,
                    "max_new_tokens_by_source": {"finchain": 128},
                    "temperature": 0.8,
                    "top_p": 0.95,
                    "max_pairs_per_prompt": None,
                    "seed": 42,
                    "dtype": "bfloat16",
                }
            ),
            encoding="utf-8",
        )
        shard_dirs.append(shard_dir)

    manifest = merge_shards(shard_dirs=shard_dirs, out_dir=tmp_path / "merged")

    assert manifest["completion_count"] == 2
    assert manifest["pair_count"] == 2
    assert manifest["source_counts"] == {"finchain": 2}
    assert (tmp_path / "merged" / "completions.jsonl").exists()
    assert (tmp_path / "merged" / "pairs.jsonl").exists()
    assert (tmp_path / "merged" / "manifest.json").exists()
