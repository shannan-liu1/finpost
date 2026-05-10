"""Dataset and sequence-packing primitives for Phase 1 SFT.

The trainer sees token IDs, not dataset-specific records. This module
bridges that gap:

1. load normalized ``Example`` records from the configured sources,
2. make a deterministic stratified train/val split,
3. serialize prompt/response text in one chat-style format, and
4. pack multiple tokenized examples into rectangular training batches.

Packing is intentionally tokenizer-agnostic. The dataset owns text
serialization and tokenization; the collator only works with input IDs
and prompt lengths, so it works for tiny-gpt2, Qwen, and any tokenizer
that returns integer token IDs.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any, Literal

import torch
from torch.utils.data import DataLoader, Dataset

from finpost.data.gsm8k import load_gsm8k
from finpost.data.math_dataset import load_math
from finpost.data.schema import Example, Source
from finpost.training.config import Config, DataConfig
from finpost.training.masking import IGNORE_INDEX, mask_prompt_tokens

Split = Literal["train", "val"]


@dataclass(frozen=True)
class TokenizedSFTExample:
    """One serialized SFT document after tokenization."""

    input_ids: torch.Tensor
    prompt_length: int
    source: Source
    example_id: str | None = None


def serialize_prompt(prompt: str) -> str:
    """Serialize a user prompt in the Phase 1 chat template."""
    return f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"


def serialize_response(response: str) -> str:
    """Serialize the assistant response paired with ``serialize_prompt``."""
    return f"{response}<|im_end|>\n"


def _tokenize_to_ids(tokenizer: Any, text: str) -> list[int]:
    encoded = tokenizer(text, add_special_tokens=False)
    input_ids = encoded["input_ids"]
    if isinstance(input_ids, torch.Tensor):
        return input_ids.flatten().tolist()
    return list(input_ids)


def _split_examples(
    examples: list[Example],
    *,
    source: Source,
    val_split_pct: float,
    seed: int,
) -> tuple[list[Example], list[Example]]:
    """Split one source deterministically so combined val remains stratified."""
    if val_split_pct == 0.0:
        return examples, []

    rng = random.Random(f"{seed}:{source}")
    indices = list(range(len(examples)))
    rng.shuffle(indices)

    val_count = round(len(examples) * val_split_pct / 100.0)
    if val_count == 0 and examples:
        val_count = 1

    val_indices = set(indices[:val_count])
    train = [ex for idx, ex in enumerate(examples) if idx not in val_indices]
    val = [ex for idx, ex in enumerate(examples) if idx in val_indices]
    return train, val


class PhasedSFTDataset(Dataset):
    """Torch dataset for Phase 1 supervised fine-tuning examples.

    ``__getitem__`` returns ``(input_ids, prompt_length, source)`` as
    requested by the trainer issue. Metadata needed for verification,
    such as ``examples`` and ``example_ids``, remains available on the
    dataset object without widening each training item.
    """

    def __init__(
        self,
        *,
        data_config: DataConfig,
        tokenizer: Any,
        split: Split,
    ) -> None:
        self.data_config = data_config
        self.tokenizer = tokenizer
        self.split = split
        self.examples = self._load_split_examples()
        self.example_ids = [ex.id for ex in self.examples]

    def _load_split_examples(self) -> list[Example]:
        by_source: dict[Source, list[Example]] = {}
        if "gsm8k" in self.data_config.sources:
            by_source["gsm8k"] = load_gsm8k(split="train")
        if "math" in self.data_config.sources:
            by_source["math"] = load_math(split="train")

        selected: list[Example] = []
        for source in self.data_config.sources:
            train, val = _split_examples(
                by_source[source],
                source=source,
                val_split_pct=self.data_config.val_split_pct,
                seed=self.data_config.seed,
            )
            selected.extend(train if self.split == "train" else val)
        return selected

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int, Source]:
        ex = self.examples[idx]
        prompt_text = serialize_prompt(ex.prompt)
        response_text = serialize_response(ex.response)

        prompt_ids = _tokenize_to_ids(self.tokenizer, prompt_text)
        response_ids = _tokenize_to_ids(self.tokenizer, response_text)
        return (
            torch.tensor(prompt_ids + response_ids, dtype=torch.long),
            len(prompt_ids),
            ex.source,
        )


class PackingCollator:
    """Greedily pack tokenized SFT documents into rows up to ``max_seq_len``."""

    def __init__(
        self,
        *,
        max_seq_len: int,
        eos_token_id: int,
        isolate_documents: bool,
        pad_token_id: int = 0,
    ) -> None:
        if max_seq_len <= 0:
            raise ValueError("max_seq_len must be positive")
        self.max_seq_len = max_seq_len
        self.eos_token_id = eos_token_id
        self.isolate_documents = isolate_documents
        self.pad_token_id = pad_token_id

    def __call__(
        self,
        examples: list[TokenizedSFTExample | tuple[torch.Tensor, int, Source]],
    ) -> dict[str, Any]:
        queue = [self._normalize(ex) for ex in examples]
        rows: list[dict[str, Any]] = []
        idx = 0
        while idx < len(queue):
            row, idx = self._pack_one_row(queue, idx)
            rows.append(row)

        width = max((len(row["input_ids"]) for row in rows), default=0)
        input_rows: list[list[int]] = []
        label_rows: list[list[int]] = []
        position_rows: list[list[int]] = []
        padding_masks: list[list[int]] = []
        boundaries: list[list[dict[str, Any]]] = []

        for row in rows:
            pad = width - len(row["input_ids"])
            input_rows.append(row["input_ids"] + [self.pad_token_id] * pad)
            label_rows.append(row["labels"] + [IGNORE_INDEX] * pad)
            position_rows.append(row["position_ids"] + [0] * pad)
            padding_masks.append([1] * len(row["input_ids"]) + [0] * pad)
            boundaries.append(row["document_boundaries"])

        batch: dict[str, Any] = {
            "input_ids": torch.tensor(input_rows, dtype=torch.long),
            "labels": torch.tensor(label_rows, dtype=torch.long),
            "position_ids": torch.tensor(position_rows, dtype=torch.long),
            "document_boundaries": boundaries,
        }
        if self.isolate_documents:
            batch["attention_mask"] = self._isolated_attention_mask(rows, width)
        else:
            batch["attention_mask"] = torch.tensor(padding_masks, dtype=torch.long)
        return batch

    def _normalize(
        self,
        example: TokenizedSFTExample | tuple[torch.Tensor, int, Source],
    ) -> TokenizedSFTExample:
        if isinstance(example, TokenizedSFTExample):
            return example
        input_ids, prompt_length, source = example
        return TokenizedSFTExample(input_ids=input_ids, prompt_length=prompt_length, source=source)

    def _pack_one_row(
        self,
        examples: list[TokenizedSFTExample],
        start_idx: int,
    ) -> tuple[dict[str, Any], int]:
        input_ids: list[int] = []
        labels: list[int] = []
        position_ids: list[int] = []
        doc_ids: list[int] = []
        boundaries: list[dict[str, Any]] = []
        idx = start_idx

        while idx < len(examples):
            ex = self._truncate(examples[idx])
            needs_eos = bool(input_ids)
            additional = len(ex.input_ids) + (1 if needs_eos else 0)
            if input_ids and len(input_ids) + additional > self.max_seq_len:
                break

            if needs_eos:
                input_ids.append(self.eos_token_id)
                labels.append(IGNORE_INDEX)
                position_ids.append(0)
                doc_ids.append(-1)

            doc_start = len(input_ids)
            ids = ex.input_ids.tolist()
            input_ids.extend(ids)
            prompt_length = min(ex.prompt_length, len(ids))
            doc_labels = mask_prompt_tokens(
                torch.tensor([ids], dtype=torch.long),
                torch.tensor([prompt_length], dtype=torch.long),
            )[0].tolist()
            labels.extend(doc_labels)
            position_ids.extend(range(len(ids)))
            doc_index = len(boundaries)
            doc_ids.extend([doc_index] * len(ids))
            boundaries.append(
                {
                    "start": doc_start,
                    "end": doc_start + len(ids),
                    "prompt_length": prompt_length,
                    "source": ex.source,
                    "example_id": ex.example_id,
                }
            )
            idx += 1

        return (
            {
                "input_ids": input_ids,
                "labels": labels,
                "position_ids": position_ids,
                "doc_ids": doc_ids,
                "document_boundaries": boundaries,
            },
            idx,
        )

    def _truncate(self, example: TokenizedSFTExample) -> TokenizedSFTExample:
        if len(example.input_ids) <= self.max_seq_len:
            return example
        return TokenizedSFTExample(
            input_ids=example.input_ids[: self.max_seq_len],
            prompt_length=min(example.prompt_length, self.max_seq_len),
            source=example.source,
            example_id=example.example_id,
        )

    def _isolated_attention_mask(self, rows: list[dict[str, Any]], width: int) -> torch.Tensor:
        masks: list[torch.Tensor] = []
        for row in rows:
            doc_ids = row["doc_ids"] + [-2] * (width - len(row["doc_ids"]))
            doc_tensor = torch.tensor(doc_ids, dtype=torch.long)
            same_document = doc_tensor.unsqueeze(0) == doc_tensor.unsqueeze(1)
            real_document = doc_tensor.unsqueeze(0) >= 0
            mask = (same_document & real_document).long()
            masks.append(mask)
        return torch.stack(masks, dim=0).unsqueeze(1)


def make_loaders(config: Config, tokenizer: Any) -> tuple[DataLoader, DataLoader]:
    """Build train/validation loaders using the configured split and packing."""
    eos_token_id = getattr(tokenizer, "eos_token_id", None)
    if eos_token_id is None:
        raise ValueError("tokenizer must expose eos_token_id for packed document separators")

    pad_token_id = getattr(tokenizer, "pad_token_id", None)
    if pad_token_id is None:
        pad_token_id = 0

    collator = PackingCollator(
        max_seq_len=config.packing.max_seq_len,
        eos_token_id=eos_token_id,
        isolate_documents=config.packing.isolate_documents,
        pad_token_id=pad_token_id,
    )
    train_dataset = PhasedSFTDataset(data_config=config.data, tokenizer=tokenizer, split="train")
    val_dataset = PhasedSFTDataset(data_config=config.data, tokenizer=tokenizer, split="val")

    train_generator = torch.Generator()
    train_generator.manual_seed(config.data.seed)

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.training.per_device_batch_size,
        shuffle=True,
        collate_fn=collator,
        generator=train_generator,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.training.per_device_batch_size,
        shuffle=False,
        collate_fn=collator,
    )
    return train_loader, val_loader
