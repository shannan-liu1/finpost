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
from finpost.training.masking import IGNORE_INDEX, mask_prompt_tokens


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


class DPOCollator:
    """Pad chosen/rejected sequences and mask prompt/padding labels."""

    def __init__(
        self,
        *,
        tokenizer: Any,
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

    def __call__(self, examples: list[DPOPreferenceExample]) -> dict[str, Any]:
        chosen = [self._encode(ex, side="chosen") for ex in examples]
        rejected = [self._encode(ex, side="rejected") for ex in examples]
        chosen_batch = self._pad(chosen)
        rejected_batch = self._pad(rejected)
        return {
            "chosen_input_ids": chosen_batch["input_ids"],
            "chosen_attention_mask": chosen_batch["attention_mask"],
            "chosen_labels": chosen_batch["labels"],
            "rejected_input_ids": rejected_batch["input_ids"],
            "rejected_attention_mask": rejected_batch["attention_mask"],
            "rejected_labels": rejected_batch["labels"],
            "prompt_ids": [ex.prompt_id for ex in examples],
            "sources": [ex.source for ex in examples],
        }

    def _encode(self, example: DPOPreferenceExample, *, side: str) -> dict[str, Any]:
        if side == "chosen":
            response = example.chosen
        elif side == "rejected":
            response = example.rejected
        else:  # pragma: no cover - defensive internal guard
            raise ValueError(f"unknown DPO side: {side}")

        prompt_ids = _tokenize_to_ids(self.tokenizer, serialize_prompt(example.prompt))
        response_ids = _tokenize_to_ids(self.tokenizer, serialize_response(response))
        input_ids = (prompt_ids + response_ids)[: self.max_seq_len]
        prompt_length = min(len(prompt_ids), len(input_ids))
        if len(input_ids) <= prompt_length:
            raise ValueError(
                f"preference example {example.prompt_id or '<unknown>'} has no response tokens "
                f"after truncation to max_seq_len={self.max_seq_len}"
            )
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "prompt_length": prompt_length,
        }

    def _pad(self, encoded: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        width = max(len(row["input_ids"]) for row in encoded)
        input_rows: list[torch.Tensor] = []
        attention_rows: list[torch.Tensor] = []
        prompt_lengths: list[int] = []
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
                        torch.ones((len(ids),), dtype=torch.long),
                        torch.zeros((pad_len,), dtype=torch.long),
                    ]
                )
            )
            prompt_lengths.append(row["prompt_length"])

        input_ids = torch.stack(input_rows)
        attention_mask = torch.stack(attention_rows)
        labels = mask_prompt_tokens(
            input_ids,
            torch.tensor(prompt_lengths, dtype=torch.long),
            attention_mask=attention_mask,
        )
        if ((labels[:, 1:] != IGNORE_INDEX).sum(dim=-1) == 0).any():
            raise ValueError("DPO batch contains a row with no shifted response labels")
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }
