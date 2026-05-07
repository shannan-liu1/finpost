"""Tests for prompt-token masking.

Each test pins one invariant of ``mask_prompt_tokens``:

1. Single example: prompt positions become IGNORE_INDEX, response
   positions keep their token IDs.
2. Batch: per-example prompt lengths are respected independently.
3. The PRD's count invariant: exactly P positions are IGNORE_INDEX,
   exactly R positions match the response tokens.
4. The function does not mutate its input_ids argument.
5. attention_mask, when provided, additionally masks padding.
"""

from __future__ import annotations

import torch

from finpost.training.masking import IGNORE_INDEX, mask_prompt_tokens


def test_single_example_masks_prompt_keeps_response() -> None:
    input_ids = torch.tensor([[10, 20, 30, 40, 50, 60, 70]])
    prompt_lengths = torch.tensor([3])

    labels = mask_prompt_tokens(input_ids, prompt_lengths)

    expected = torch.tensor([[IGNORE_INDEX, IGNORE_INDEX, IGNORE_INDEX, 40, 50, 60, 70]])
    assert torch.equal(labels, expected)


def test_batch_with_different_prompt_lengths() -> None:
    """Each example's prompt mask is independent of the others'."""
    input_ids = torch.tensor(
        [
            [10, 20, 30, 40, 50],  # prompt length 2
            [11, 21, 31, 41, 51],  # prompt length 4
        ]
    )
    prompt_lengths = torch.tensor([2, 4])

    labels = mask_prompt_tokens(input_ids, prompt_lengths)

    expected = torch.tensor(
        [
            [IGNORE_INDEX, IGNORE_INDEX, 30, 40, 50],
            [IGNORE_INDEX, IGNORE_INDEX, IGNORE_INDEX, IGNORE_INDEX, 51],
        ]
    )
    assert torch.equal(labels, expected)


def test_count_invariant_exactly_p_masked_and_r_kept() -> None:
    """PRD acceptance criterion: P positions are IGNORE_INDEX, R match the response."""
    input_ids = torch.tensor([[10, 20, 30, 40, 50, 60, 70]])  # P=3, R=4
    prompt_lengths = torch.tensor([3])
    P, R = 3, 4

    labels = mask_prompt_tokens(input_ids, prompt_lengths)

    assert (labels == IGNORE_INDEX).sum().item() == P
    assert (labels != IGNORE_INDEX).sum().item() == R
    # Response positions match the original token IDs at those positions.
    assert torch.equal(labels[0, P:], input_ids[0, P:])


def test_input_ids_not_mutated() -> None:
    """The function must return a fresh tensor; never modify its input."""
    input_ids = torch.tensor([[10, 20, 30, 40]])
    snapshot = input_ids.clone()
    prompt_lengths = torch.tensor([2])

    _ = mask_prompt_tokens(input_ids, prompt_lengths)

    assert torch.equal(input_ids, snapshot)


def test_attention_mask_masks_padding() -> None:
    """attention_mask=0 positions become IGNORE_INDEX in addition to prompt positions."""
    input_ids = torch.tensor([[10, 20, 30, 0, 0]])  # last 2 are padding
    prompt_lengths = torch.tensor([2])
    attention_mask = torch.tensor([[1, 1, 1, 0, 0]])

    labels = mask_prompt_tokens(input_ids, prompt_lengths, attention_mask)

    expected = torch.tensor([[IGNORE_INDEX, IGNORE_INDEX, 30, IGNORE_INDEX, IGNORE_INDEX]])
    assert torch.equal(labels, expected)
