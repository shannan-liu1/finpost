"""Behavior tests for DPO preference-pair data loading and collation."""

from __future__ import annotations

import json

import torch

from finpost.training.masking import IGNORE_INDEX


class TinyTokenizer:
    """Tokenizer test double: deterministic IDs without model-specific behavior."""

    eos_token_id = 99
    pad_token_id = 0
    name_or_path = "tiny-tokenizer"

    def __call__(self, text: str, *, add_special_tokens: bool = False) -> dict[str, list[int]]:
        del add_special_tokens
        return {"input_ids": [ord(ch) for ch in text]}


def test_load_preference_pairs_reads_jsonl_and_preserves_metadata(tmp_path) -> None:
    """The DPO dataset is plain JSONL so pair generation and training decouple."""
    from finpost.training.preference_data import load_preference_pairs

    path = tmp_path / "pairs.jsonl"
    path.write_text(
        json.dumps(
            {
                "prompt": "1+1?",
                "chosen": "2",
                "rejected": "3",
                "source": "gsm8k",
                "prompt_id": "gsm8k-0",
                "chosen_grade": {"correct": True},
                "rejected_grade": {"correct": False},
                "metadata": {"sample_ids": ["a", "b"]},
            }
        )
        + "\n",
        encoding="utf-8",
    )

    pairs = load_preference_pairs(path)

    assert len(pairs) == 1
    assert pairs[0].prompt == "1+1?"
    assert pairs[0].chosen == "2"
    assert pairs[0].rejected == "3"
    assert pairs[0].source == "gsm8k"
    assert pairs[0].metadata == {"sample_ids": ["a", "b"]}


def test_dpo_collator_masks_prompt_and_padding_for_both_sides() -> None:
    """Only response tokens should contribute to chosen/rejected logprobs."""
    from finpost.training.dataset import serialize_prompt, serialize_response
    from finpost.training.preference_data import DPOCollator, DPOPreferenceExample

    tokenizer = TinyTokenizer()
    prompt = "Q?"
    chosen = "long"
    rejected = "no"
    collator = DPOCollator(tokenizer=tokenizer, max_seq_len=128)

    batch = collator(
        [
            DPOPreferenceExample(
                prompt=prompt,
                chosen=chosen,
                rejected=rejected,
                source="gsm8k",
                prompt_id="p0",
            )
        ]
    )

    prompt_len = len(serialize_prompt(prompt))
    chosen_response_len = len(serialize_response(chosen))
    rejected_response_len = len(serialize_response(rejected))

    assert batch["chosen_labels"][0, :prompt_len].tolist() == [IGNORE_INDEX] * prompt_len
    assert batch["rejected_labels"][0, :prompt_len].tolist() == [IGNORE_INDEX] * prompt_len
    assert batch["chosen_labels"][0, prompt_len : prompt_len + chosen_response_len].tolist() == [
        ord(ch) for ch in serialize_response(chosen)
    ]
    rejected_labels = batch["rejected_labels"][0, prompt_len : prompt_len + rejected_response_len]
    assert rejected_labels.tolist() == [ord(ch) for ch in serialize_response(rejected)]

    rejected_pad_start = prompt_len + rejected_response_len
    assert batch["rejected_labels"][0, rejected_pad_start:].tolist() == [IGNORE_INDEX] * (
        batch["rejected_labels"].shape[1] - rejected_pad_start
    )
    assert batch["chosen_input_ids"].shape == batch["rejected_input_ids"].shape


def test_dpo_collator_rejects_truncated_rows_without_response_labels() -> None:
    """Bad max_seq_len settings should fail before a GPU run starts."""
    from finpost.training.dataset import serialize_prompt
    from finpost.training.preference_data import DPOCollator, DPOPreferenceExample

    prompt = "this prompt is too long"
    collator = DPOCollator(tokenizer=TinyTokenizer(), max_seq_len=len(serialize_prompt(prompt)))

    try:
        collator(
            [
                DPOPreferenceExample(
                    prompt=prompt,
                    chosen="correct",
                    rejected="wrong",
                    source="math",
                )
            ]
        )
    except ValueError as exc:
        assert "no response tokens" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("expected no response tokens to raise")


def test_tokenized_preference_dataset_cache_avoids_retokenizing(tmp_path) -> None:
    """The full DPO run should reuse canary tokenization work from disk."""
    from finpost.training.preference_data import load_or_build_tokenized_preference_dataset

    pairs_path = tmp_path / "pairs.jsonl"
    pairs_path.write_text(
        json.dumps(
            {
                "prompt": "Q?",
                "chosen": "long",
                "rejected": "no",
                "source": "gsm8k",
                "prompt_id": "p0",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    cache_path = tmp_path / "pairs.tokenized.pt"

    first = load_or_build_tokenized_preference_dataset(
        pairs_path=pairs_path,
        tokenizer=TinyTokenizer(),
        max_seq_len=128,
        cache_path=cache_path,
    )

    class ExplodingTokenizer(TinyTokenizer):
        def __call__(self, text: str, *, add_special_tokens: bool = False) -> dict[str, list[int]]:
            raise AssertionError(f"cache miss retokenized {text!r}")

    second = load_or_build_tokenized_preference_dataset(
        pairs_path=pairs_path,
        tokenizer=ExplodingTokenizer(),
        max_seq_len=128,
        cache_path=cache_path,
    )

    assert cache_path.exists()
    assert len(first) == len(second) == 1
    assert first[0].chosen_input_ids.tolist() == second[0].chosen_input_ids.tolist()


def test_dpo_collator_emits_cached_reference_logps() -> None:
    """Cached reference scores should move through collation with the batch."""
    from finpost.training.preference_data import (
        DPOCollator,
        TokenizedDPOPreferenceExample,
    )

    example = TokenizedDPOPreferenceExample(
        chosen_input_ids=torch.tensor([1, 2, 3]),
        chosen_attention_mask=torch.ones(3, dtype=torch.long),
        chosen_labels=torch.tensor([IGNORE_INDEX, 2, 3]),
        rejected_input_ids=torch.tensor([1, 2, 4]),
        rejected_attention_mask=torch.ones(3, dtype=torch.long),
        rejected_labels=torch.tensor([IGNORE_INDEX, 2, 4]),
        source="gsm8k",
        prompt_id="p0",
        ref_chosen_logp=-1.25,
        ref_rejected_logp=-2.5,
    )

    batch = DPOCollator(tokenizer=None, max_seq_len=8, pad_token_id=0)([example])

    assert batch["ref_chosen_logps"].tolist() == [-1.25]
    assert batch["ref_rejected_logps"].tolist() == [-2.5]
