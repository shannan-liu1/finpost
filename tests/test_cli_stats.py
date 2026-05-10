"""Tests for the pure-function helpers in finpost.data.cli.

The end-to-end CLI run requires the Qwen tokenizer download and is
verified through manual invocation. These tests cover the statistics
computation and formatting in isolation, with no network or tokenizer
dependency.
"""

from __future__ import annotations

import pytest

from finpost.data.cli import LengthStats, compute_length_stats


def test_basic_stats_over_five_values() -> None:
    """A small fixed list with known statistics."""
    stats = compute_length_stats([10, 20, 30, 40, 50])
    assert stats.count == 5
    assert stats.mean == 30.0
    # n // 2 == 2 → sorted_counts[2] == 30
    assert stats.p50 == 30
    # int(5 * 0.95) == 4 → sorted_counts[4] == 50
    assert stats.p95 == 50
    assert stats.max == 50


def test_stats_over_single_value() -> None:
    """Edge case: one value. All stats degenerate to that value."""
    stats = compute_length_stats([42])
    assert stats.count == 1
    assert stats.mean == 42.0
    assert stats.p50 == 42
    assert stats.p95 == 42
    assert stats.max == 42


def test_stats_unsorted_input() -> None:
    """The function must sort internally; input order should not matter."""
    stats_a = compute_length_stats([5, 1, 4, 2, 3])
    stats_b = compute_length_stats([1, 2, 3, 4, 5])
    assert stats_a == stats_b


def test_empty_input_raises() -> None:
    """Statistics over no data are undefined; we raise rather than return zeros."""
    with pytest.raises(ValueError, match="empty list"):
        compute_length_stats([])


def test_format_includes_label_and_all_fields() -> None:
    """The format method should mention the label and every numeric field."""
    stats = LengthStats(count=100, mean=53.7, p50=48, p95=120, max=342)
    out = stats.format("Prompt tokens")
    assert "Prompt tokens" in out
    assert "n=100" in out
    assert "53.7" in out
    assert "48" in out
    assert "120" in out
    assert "342" in out


def test_format_is_multiline() -> None:
    """Output should be a small block, not a single line."""
    stats = LengthStats(count=1, mean=1.0, p50=1, p95=1, max=1)
    out = stats.format("Test")
    assert out.count("\n") == 4  # one header line + 4 metric lines
