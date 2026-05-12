"""Tests for finpost.safety.

These tests verify the contract of the wrapper without making any
network calls. We patch the underlying HuggingFace functions and
inspect the keyword arguments our wrappers pass through.

Invariants checked for each wrapper:

1. By default, ``trust_remote_code=False`` is passed to the underlying
   loader. This is the policy the wrappers exist to enforce.

2. If a caller explicitly opts in by passing ``trust_remote_code=True``,
   the wrapper forwards that — it does NOT silently override or block.
   The wrapper is a default-setter, not a hard gate.

3. For ``safe_load_model``: ``use_safetensors=True`` by default (refuse
   pickle-format weights). The caller can override by passing
   ``use_safetensors=False`` when explicitly required (e.g. test models).
"""

from __future__ import annotations

from unittest.mock import patch

from finpost.safety import safe_load_dataset, safe_load_model, safe_load_tokenizer


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


# =============================================================================
# safe_load_model tests
# =============================================================================


def test_safe_load_model_default_trust_remote_code_is_false() -> None:
    """safe_load_model passes trust_remote_code=False by default."""
    with patch("finpost.safety._AutoModelForCausalLM") as mock_cls:
        safe_load_model("some/model")

    mock_cls.from_pretrained.assert_called_once()
    _, kwargs = mock_cls.from_pretrained.call_args
    assert kwargs["trust_remote_code"] is False


def test_safe_load_model_default_use_safetensors_is_true() -> None:
    """safe_load_model passes use_safetensors=True by default."""
    with patch("finpost.safety._AutoModelForCausalLM") as mock_cls:
        safe_load_model("some/model")

    _, kwargs = mock_cls.from_pretrained.call_args
    assert kwargs["use_safetensors"] is True


def test_safe_load_model_trust_remote_code_override_is_forwarded() -> None:
    """If the caller passes trust_remote_code=True, the wrapper forwards it."""
    with patch("finpost.safety._AutoModelForCausalLM") as mock_cls:
        safe_load_model("some/model", trust_remote_code=True)

    _, kwargs = mock_cls.from_pretrained.call_args
    assert kwargs["trust_remote_code"] is True


def test_safe_load_model_use_safetensors_false_override_is_forwarded() -> None:
    """If the caller passes use_safetensors=False (e.g. for tiny-gpt2), the wrapper forwards it."""
    with patch("finpost.safety._AutoModelForCausalLM") as mock_cls:
        safe_load_model("sshleifer/tiny-gpt2", use_safetensors=False)

    _, kwargs = mock_cls.from_pretrained.call_args
    assert kwargs["use_safetensors"] is False


def test_safe_load_model_extra_kwargs_forwarded() -> None:
    """Extra keyword arguments (e.g. torch_dtype) are forwarded to from_pretrained."""
    import torch

    with patch("finpost.safety._AutoModelForCausalLM") as mock_cls:
        safe_load_model("some/model", torch_dtype=torch.float16)

    _, kwargs = mock_cls.from_pretrained.call_args
    assert kwargs["torch_dtype"] == torch.float16


# =============================================================================
# safe_load_tokenizer tests
# =============================================================================


def test_safe_load_tokenizer_default_trust_remote_code_is_false() -> None:
    """safe_load_tokenizer passes trust_remote_code=False by default."""
    with patch("finpost.safety._AutoTokenizer") as mock_cls:
        safe_load_tokenizer("some/model")

    mock_cls.from_pretrained.assert_called_once()
    _, kwargs = mock_cls.from_pretrained.call_args
    assert kwargs["trust_remote_code"] is False


def test_safe_load_tokenizer_trust_remote_code_override_is_forwarded() -> None:
    """If the caller passes trust_remote_code=True, the wrapper forwards it."""
    with patch("finpost.safety._AutoTokenizer") as mock_cls:
        safe_load_tokenizer("some/model", trust_remote_code=True)

    _, kwargs = mock_cls.from_pretrained.call_args
    assert kwargs["trust_remote_code"] is True


def test_safe_load_tokenizer_extra_kwargs_forwarded() -> None:
    """Extra keyword arguments are forwarded to from_pretrained."""
    with patch("finpost.safety._AutoTokenizer") as mock_cls:
        safe_load_tokenizer("some/model", use_fast=False)

    _, kwargs = mock_cls.from_pretrained.call_args
    assert kwargs["use_fast"] is False
