"""Preference-pair dataset and collation for DPO training."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset

from finpost.data.schema import Source
from finpost.training.dataset import (
    _tokenize_to_ids,
    serialize_prompt,
    serialize_response,
)
from finpost.training.masking import IGNORE_INDEX


@dataclass(frozen=True)
class DPOPreferenceExample:
    """One prompt with a preferred and rejected completion."""

    prompt: str
    chosen: str
    rejected: str
    source: Source
    prompt_id: str | None = None
    chosen_grade: dict[str, Any] | None = None
    rejected_grade: dict[str, Any] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class TokenizedDPOPreferenceExample:
    """One preference pair after tokenizer-dependent work has been done."""

    chosen_input_ids: torch.Tensor
    chosen_attention_mask: torch.Tensor
    chosen_labels: torch.Tensor
    rejected_input_ids: torch.Tensor
    rejected_attention_mask: torch.Tensor
    rejected_labels: torch.Tensor
    source: Source
    prompt_id: str | None = None


def load_preference_pairs(path: str | Path) -> list[DPOPreferenceExample]:
    """Load JSONL DPO pairs produced by ``scripts/build_dpo_pairs.py``."""
    path = Path(path)
    examples: list[DPOPreferenceExample] = []
    with path.open("r", encoding="utf-8") as fp:
        for line_no, line in enumerate(fp, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            raw = json.loads(stripped)
            try:
                examples.append(
                    DPOPreferenceExample(
                        prompt=raw["prompt"],
                        chosen=raw["chosen"],
                        rejected=raw["rejected"],
                        source=raw["source"],
                        prompt_id=raw.get("prompt_id"),
                        chosen_grade=raw.get("chosen_grade"),
                        rejected_grade=raw.get("rejected_grade"),
                        metadata=raw.get("metadata") or {},
                    )
                )
            except KeyError as exc:
                raise ValueError(f"{path}:{line_no} missing required field {exc}") from exc
    if not examples:
        raise ValueError(f"no preference pairs found in {path}")
    return examples


class DPOPreferenceDataset(Dataset):
    """Torch dataset over already-built DPO preference pairs."""

    def __init__(self, pairs_path: str | Path) -> None:
        self.pairs_path = Path(pairs_path)
        self.examples = load_preference_pairs(self.pairs_path)

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> DPOPreferenceExample:
        return self.examples[idx]


class TokenizedDPOPreferenceDataset(Dataset):
    """Torch dataset over tokenized DPO preference pairs."""

    def __init__(self, examples: list[TokenizedDPOPreferenceExample]) -> None:
        if not examples:
            raise ValueError("tokenized preference dataset is empty")
        self.examples = examples

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> TokenizedDPOPreferenceExample:
        return self.examples[idx]


def _encode_side(
    *,
    tokenizer: Any,
    prompt: str,
    response: str,
    max_seq_len: int,
) -> dict[str, torch.Tensor]:
    prompt_ids = _tokenize_to_ids(tokenizer, serialize_prompt(prompt))
    response_ids = _tokenize_to_ids(tokenizer, serialize_response(response))
    input_ids = (prompt_ids + response_ids)[:max_seq_len]
    prompt_length = min(len(prompt_ids), len(input_ids))
    if len(input_ids) <= prompt_length:
        raise ValueError(
            "preference example has no response tokens "
            f"after truncation to max_seq_len={max_seq_len}"
        )

    input_tensor = torch.tensor(input_ids, dtype=torch.long)
    attention_mask = torch.ones_like(input_tensor)
    labels = input_tensor.clone()
    labels[:prompt_length] = IGNORE_INDEX
    if (labels[1:] != IGNORE_INDEX).sum().item() == 0:
        raise ValueError("DPO example has no shifted response labels")
    return {
        "input_ids": input_tensor,
        "attention_mask": attention_mask,
        "labels": labels,
    }


def tokenize_preference_pairs(
    examples: list[DPOPreferenceExample],
    *,
    tokenizer: Any,
    max_seq_len: int,
) -> list[TokenizedDPOPreferenceExample]:
    """Tokenize preference pairs once so the GPU training loop is not blocked."""
    tokenized: list[TokenizedDPOPreferenceExample] = []
    for example in examples:
        chosen = _encode_side(
            tokenizer=tokenizer,
            prompt=example.prompt,
            response=example.chosen,
            max_seq_len=max_seq_len,
        )
        rejected = _encode_side(
            tokenizer=tokenizer,
            prompt=example.prompt,
            response=example.rejected,
            max_seq_len=max_seq_len,
        )
        tokenized.append(
            TokenizedDPOPreferenceExample(
                chosen_input_ids=chosen["input_ids"],
                chosen_attention_mask=chosen["attention_mask"],
                chosen_labels=chosen["labels"],
                rejected_input_ids=rejected["input_ids"],
                rejected_attention_mask=rejected["attention_mask"],
                rejected_labels=rejected["labels"],
                source=example.source,
                prompt_id=example.prompt_id,
            )
        )
    return tokenized


def _tokenized_cache_metadata(
    *,
    pairs_path: Path,
    tokenizer: Any,
    max_seq_len: int,
) -> dict[str, Any]:
    stat = pairs_path.stat()
    tokenizer_id = getattr(tokenizer, "name_or_path", None) or tokenizer.__class__.__name__
    return {
        "version": 1,
        "pairs_path": str(pairs_path.resolve()),
        "pairs_mtime_ns": stat.st_mtime_ns,
        "pairs_size": stat.st_size,
        "tokenizer": str(tokenizer_id),
        "max_seq_len": max_seq_len,
    }


def _serialize_tokenized(
    examples: list[TokenizedDPOPreferenceExample],
) -> list[dict[str, Any]]:
    return [
        {
            "chosen_input_ids": ex.chosen_input_ids,
            "chosen_attention_mask": ex.chosen_attention_mask,
            "chosen_labels": ex.chosen_labels,
            "rejected_input_ids": ex.rejected_input_ids,
            "rejected_attention_mask": ex.rejected_attention_mask,
            "rejected_labels": ex.rejected_labels,
            "source": ex.source,
            "prompt_id": ex.prompt_id,
        }
        for ex in examples
    ]


def _deserialize_tokenized(
    rows: list[dict[str, Any]],
) -> list[TokenizedDPOPreferenceExample]:
    return [
        TokenizedDPOPreferenceExample(
            chosen_input_ids=row["chosen_input_ids"],
            chosen_attention_mask=row["chosen_attention_mask"],
            chosen_labels=row["chosen_labels"],
            rejected_input_ids=row["rejected_input_ids"],
            rejected_attention_mask=row["rejected_attention_mask"],
            rejected_labels=row["rejected_labels"],
            source=row["source"],
            prompt_id=row.get("prompt_id"),
        )
        for row in rows
    ]


def load_or_build_tokenized_preference_dataset(
    *,
    pairs_path: str | Path,
    tokenizer: Any,
    max_seq_len: int,
    cache_path: str | Path | None = None,
    rebuild_cache: bool = False,
) -> TokenizedDPOPreferenceDataset:
    """Load tokenized pairs from cache or build and cache them once."""
    pairs_path = Path(pairs_path)
    metadata = _tokenized_cache_metadata(
        pairs_path=pairs_path,
        tokenizer=tokenizer,
        max_seq_len=max_seq_len,
    )
    if cache_path is None:
        cache_path = pairs_path.with_suffix(".tokenized.pt")
    cache_path = Path(cache_path)

    if cache_path.exists() and not rebuild_cache:
        payload = torch.load(cache_path, weights_only=False)
        if payload.get("metadata") == metadata:
            return TokenizedDPOPreferenceDataset(
                _deserialize_tokenized(payload["examples"])
            )

    examples = tokenize_preference_pairs(
        load_preference_pairs(pairs_path),
        tokenizer=tokenizer,
        max_seq_len=max_seq_len,
    )
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "metadata": metadata,
            "examples": _serialize_tokenized(examples),
        },
        cache_path,
    )
    return TokenizedDPOPreferenceDataset(examples)


class DPOCollator:
    """Pad chosen/rejected sequences and mask prompt/padding labels."""

    def __init__(
        self,
        *,
        tokenizer: Any | None,
        max_seq_len: int,
        pad_token_id: int | None = None,
    ) -> None:
        if max_seq_len <= 1:
            raise ValueError("max_seq_len must be greater than 1")
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        resolved_pad_id = pad_token_id
        if resolved_pad_id is None:
            resolved_pad_id = getattr(tokenizer, "pad_token_id", None)
        if resolved_pad_id is None:
            resolved_pad_id = getattr(tokenizer, "eos_token_id", 0)
        self.pad_token_id = int(resolved_pad_id)

    def __call__(
        self,
        examples: list[DPOPreferenceExample | TokenizedDPOPreferenceExample],
    ) -> dict[str, Any]:
        tokenized = [self._normalize(ex) for ex in examples]
        chosen = [
            {
                "input_ids": ex.chosen_input_ids,
                "attention_mask": ex.chosen_attention_mask,
                "labels": ex.chosen_labels,
            }
            for ex in tokenized
        ]
        rejected = [
            {
                "input_ids": ex.rejected_input_ids,
                "attention_mask": ex.rejected_attention_mask,
                "labels": ex.rejected_labels,
            }
            for ex in tokenized
        ]
        width = max(
            max(len(row["input_ids"]) for row in chosen),
            max(len(row["input_ids"]) for row in rejected),
        )
        chosen_batch = self._pad(chosen, width=width)
        rejected_batch = self._pad(rejected, width=width)
        return {
            "chosen_input_ids": chosen_batch["input_ids"],
            "chosen_attention_mask": chosen_batch["attention_mask"],
            "chosen_labels": chosen_batch["labels"],
            "rejected_input_ids": rejected_batch["input_ids"],
            "rejected_attention_mask": rejected_batch["attention_mask"],
            "rejected_labels": rejected_batch["labels"],
            "prompt_ids": [ex.prompt_id for ex in tokenized],
            "sources": [ex.source for ex in tokenized],
        }

    def _normalize(
        self,
        example: DPOPreferenceExample | TokenizedDPOPreferenceExample,
    ) -> TokenizedDPOPreferenceExample:
        if isinstance(example, TokenizedDPOPreferenceExample):
            return example
        if self.tokenizer is None:
            raise ValueError("raw DPO examples require a tokenizer")
        try:
            return tokenize_preference_pairs(
                [example],
                tokenizer=self.tokenizer,
                max_seq_len=self.max_seq_len,
            )[0]
        except ValueError as exc:
            raise ValueError(
                f"preference example {example.prompt_id or '<unknown>'} has no response tokens "
                f"after truncation to max_seq_len={self.max_seq_len}"
            ) from exc

    def _pad(
        self,
        encoded: list[dict[str, torch.Tensor]],
        *,
        width: int,
    ) -> dict[str, torch.Tensor]:
        input_rows: list[torch.Tensor] = []
        attention_rows: list[torch.Tensor] = []
        label_rows: list[torch.Tensor] = []
        for row in encoded:
            ids = row["input_ids"]
            pad_len = width - len(ids)
            input_rows.append(
                torch.cat(
                    [
                        ids,
                        torch.full((pad_len,), self.pad_token_id, dtype=torch.long),
                    ]
                )
            )
            attention_rows.append(
                torch.cat(
                    [
                        row["attention_mask"],
                        torch.zeros((pad_len,), dtype=torch.long),
                    ]
                )
            )
            label_rows.append(
                torch.cat(
                    [
                        row["labels"],
                        torch.full((pad_len,), IGNORE_INDEX, dtype=torch.long),
                    ]
                )
            )

        input_ids = torch.stack(input_rows)
        attention_mask = torch.stack(attention_rows)
        labels = torch.stack(label_rows)
        if ((labels[:, 1:] != IGNORE_INDEX).sum(dim=-1) == 0).any():
            raise ValueError("DPO batch contains a row with no shifted response labels")
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }
