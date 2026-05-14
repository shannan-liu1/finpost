"""Behavior tests for SFT dataset splitting and packed collation."""

from __future__ import annotations

import torch

from finpost.data.schema import Example
from finpost.training.config import (
    Config,
    DataConfig,
    ModelConfig,
    PackingConfig,
    TrainingConfig,
)
from finpost.training.masking import IGNORE_INDEX


class TinyTokenizer:
    """Tokenizer test double: deterministic IDs without model-specific behavior."""

    eos_token_id = 99

    def __call__(self, text: str, *, add_special_tokens: bool = False) -> dict[str, list[int]]:
        del add_special_tokens
        return {"input_ids": [ord(ch) for ch in text]}


def _example(source: str, idx: int) -> Example:
    return Example(
        id=f"{source}-{idx}",
        source=source,  # type: ignore[arg-type]
        prompt=f"{source} question {idx}?",
        response=f"{source} answer {idx}",
        final_answer=str(idx),
    )


def _config(*, seed: int = 7, val_split_pct: float = 20.0) -> Config:
    return Config(
        model=ModelConfig(base_model_id="sshleifer/tiny-gpt2", dtype="float32"),
        data=DataConfig(
            sources=["gsm8k", "math"],
            val_split_pct=val_split_pct,
            seed=seed,
        ),
        training=TrainingConfig(max_steps=10, lr=1e-4, warmup_steps=1, per_device_batch_size=2),
        packing=PackingConfig(max_seq_len=64, isolate_documents=True),
    )


def test_make_loaders_builds_reproducible_disjoint_stratified_splits(monkeypatch) -> None:
    """The held-out validation set is deterministic, disjoint, and per-source."""
    from finpost.training import dataset as dataset_module

    gsm8k = [_example("gsm8k", i) for i in range(100)]
    math = [_example("math", i) for i in range(100)]
    monkeypatch.setattr(dataset_module, "load_gsm8k", lambda split="train": gsm8k)
    monkeypatch.setattr(dataset_module, "load_math", lambda split="train": math)

    first_train, first_val = dataset_module.make_loaders(_config(seed=13), TinyTokenizer())
    second_train, second_val = dataset_module.make_loaders(_config(seed=13), TinyTokenizer())

    assert first_val.dataset.example_ids == second_val.dataset.example_ids
    assert set(first_train.dataset.example_ids).isdisjoint(first_val.dataset.example_ids)

    val_sources = [ex.source for ex in first_val.dataset.examples]
    assert val_sources.count("gsm8k") == 20
    assert val_sources.count("math") == 20


def test_dataset_tokenizes_serialized_prompt_and_masks_prompt_length(monkeypatch) -> None:
    """Items are tokenizer-agnostic IDs plus the serialized prompt length."""
    from finpost.training import dataset as dataset_module

    records = [_example("gsm8k", 0), _example("math", 0)]
    monkeypatch.setattr(dataset_module, "load_gsm8k", lambda split="train": records[:1])
    monkeypatch.setattr(dataset_module, "load_math", lambda split="train": records[1:])

    ds = dataset_module.PhasedSFTDataset(
        data_config=DataConfig(sources=["gsm8k", "math"], val_split_pct=0.0, seed=1),
        tokenizer=TinyTokenizer(),
        split="train",
    )

    input_ids, prompt_length, source = ds[0]
    serialized_prompt = dataset_module.serialize_prompt(records[0].prompt)
    serialized_response = dataset_module.serialize_response(records[0].response)

    assert source == "gsm8k"
    assert input_ids.tolist() == [ord(ch) for ch in serialized_prompt + serialized_response]
    assert prompt_length == len(serialized_prompt)


def test_collator_packs_rows_masks_labels_and_resets_positions() -> None:
    """Packed rows preserve per-document loss masks and RoPE positions."""
    from finpost.training.dataset import PackingCollator, TokenizedSFTExample

    collator = PackingCollator(max_seq_len=8, eos_token_id=99, isolate_documents=False)
    batch = collator(
        [
            TokenizedSFTExample(torch.tensor([10, 11, 12]), prompt_length=2, source="gsm8k"),
            TokenizedSFTExample(torch.tensor([20, 21]), prompt_length=1, source="math"),
            TokenizedSFTExample(torch.tensor([30, 31, 32, 33]), prompt_length=1, source="gsm8k"),
        ]
    )

    assert batch["input_ids"].tolist() == [
        [10, 11, 12, 99, 20, 21],
        [30, 31, 32, 33, 0, 0],
    ]
    assert batch["labels"].tolist() == [
        [IGNORE_INDEX, IGNORE_INDEX, 12, IGNORE_INDEX, IGNORE_INDEX, 21],
        [IGNORE_INDEX, 31, 32, 33, IGNORE_INDEX, IGNORE_INDEX],
    ]
    assert batch["position_ids"].tolist() == [
        [0, 1, 2, 0, 0, 1],
        [0, 1, 2, 3, 0, 0],
    ]
    assert [row[-1]["end"] for row in batch["document_boundaries"]] == [6, 4]
    assert batch["input_ids"].shape[1] <= 8


def test_collator_isolates_attention_between_documents() -> None:
    """A 4D mask blocks attention across packed document boundaries AND across the causal direction."""
    from finpost.training.dataset import PackingCollator, TokenizedSFTExample

    collator = PackingCollator(max_seq_len=6, eos_token_id=99, isolate_documents=True)
    batch = collator(
        [
            TokenizedSFTExample(torch.tensor([1, 2, 3]), prompt_length=1, source="gsm8k"),
            TokenizedSFTExample(torch.tensor([4, 5]), prompt_length=1, source="math"),
        ]
    )

    # Packed row: [1, 2, 3, EOS=99, 4, 5]  width=6
    # doc_ids   : [0, 0, 0,    -1, 1, 1]
    mask = batch["attention_mask"]
    assert mask.shape == (1, 1, 6, 6)

    # Causal direction within a document: query position 2 attends to past
    # positions 0 and 1 of the same document.
    assert mask[0, 0, 2, 0].item() == 1
    assert mask[0, 0, 2, 1].item() == 1
    assert mask[0, 0, 2, 2].item() == 1

    # No future leakage even within the same document: query position 0
    # cannot attend to position 2 (would be peeking at the future).
    assert mask[0, 0, 0, 2].item() == 0

    # Cross-document isolation: query position 4 (doc B) cannot attend to
    # any position of doc A or to the EOS separator.
    assert mask[0, 0, 4, 1].item() == 0
    assert mask[0, 0, 4, 3].item() == 0

    # Same-document causal works in doc B too.
    assert mask[0, 0, 5, 4].item() == 1
    assert mask[0, 0, 5, 5].item() == 1


def test_collator_isolated_attention_mask_has_no_empty_query_rows() -> None:
    """Ignored separator and padding queries still need one valid attention target."""
    from finpost.training.dataset import PackingCollator, TokenizedSFTExample

    collator = PackingCollator(max_seq_len=6, eos_token_id=99, isolate_documents=True)
    batch = collator(
        [
            TokenizedSFTExample(torch.tensor([1, 2, 3]), prompt_length=1, source="gsm8k"),
            TokenizedSFTExample(torch.tensor([4, 5]), prompt_length=1, source="math"),
            TokenizedSFTExample(torch.tensor([6]), prompt_length=0, source="gsm8k"),
        ]
    )

    mask = batch["attention_mask"]
    assert mask.shape == (2, 1, 6, 6)
    assert (mask.sum(dim=-1) > 0).all()

    # Row 0 contains a separator at position 3; row 1 contains padding
    # after position 0. Those ignored queries get a self-only fallback so
    # older CUDA SDPA kernels do not see an all-blocked row.
    assert mask[0, 0, 3].tolist() == [0, 0, 0, 1, 0, 0]
    assert mask[1, 0, 1].tolist() == [0, 1, 0, 0, 0, 0]

    # Real training tokens remain document-isolated.
    assert mask[0, 0, 4, 1].item() == 0
    assert mask[0, 0, 5, 4].item() == 1
