"""Tests for finpost.safety.

These tests verify the contract of the wrapper without making any
network calls. We patch the underlying ``datasets.load_dataset`` and
inspect the keyword arguments our wrapper passes through.

Two invariants are checked:

1. By default, ``safe_load_dataset`` passes ``trust_remote_code=False``
   to the underlying loader. This is the policy the wrapper exists to
   enforce.

2. If a caller explicitly opts in by passing ``trust_remote_code=True``,
   the wrapper forwards that — it does NOT silently override or block.
   The wrapper is a default-setter, not a hard gate. Hard gates would
   force people to bypass the wrapper entirely, which would defeat the
   purpose.
"""

from __future__ import annotations

from unittest.mock import patch

from finpost.safety import safe_load_dataset


def test_default_trust_remote_code_is_false() -> None:
    """Without an explicit override, the wrapper must pass False."""
    with patch("finpost.safety._hf_load_dataset") as mock_load:
        safe_load_dataset("some/dataset", "main", split="train")

    assert mock_load.call_count == 1
    _, kwargs = mock_load.call_args
    assert kwargs["trust_remote_code"] is False


def test_explicit_override_is_forwarded() -> None:
    """If the caller deliberately sets True, the wrapper forwards True."""
    with patch("finpost.safety._hf_load_dataset") as mock_load:
        safe_load_dataset("some/dataset", trust_remote_code=True)

    _, kwargs = mock_load.call_args
    assert kwargs["trust_remote_code"] is True


def test_revision_pinning_is_forwarded() -> None:
    """The revision argument (used to pin to parquet branches) must reach the loader unchanged."""
    with patch("finpost.safety._hf_load_dataset") as mock_load:
        safe_load_dataset(
            "openai/gsm8k",
            "main",
            split="train",
            revision="refs/convert/parquet",
        )

    _, kwargs = mock_load.call_args
    assert kwargs["revision"] == "refs/convert/parquet"
