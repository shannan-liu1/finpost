"""Tests for the eval_exact CLI.

Covers the acceptance criteria from issue 02:
  1. Seeded subsampling is deterministic (same seed → same indices).
  2. Batched-vs-single generation parity on tiny-gpt2 with shared seed.
  3. OOM fallback halves batch size and retries; fails loudly at batch_size=1.
  4. Output artifact schemas: accuracy_summary.csv and details_*.csv columns.
  5. Re-run with same seed produces byte-identical details_*.csv.

All tests use the ``sshleifer/tiny-gpt2`` model so they run in seconds
on CPU. The tiny model produces garbage math answers — that is fine;
these tests verify infrastructure, not model quality.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import pytest
import torch

from finpost.data.schema import Example

# We import specific functions from eval_exact, which validates the module's
# public surface without running the full CLI entrypoint.
from finpost.evals.eval_exact import (
    RunTracker,
    _count_non_pad_tokens,
    _generate_batch,
    _generate_chunk_with_oom_fallback,
    _parse_checkpoint_pair,
    _sample_examples,
    _set_cuda_determinism,
    _write_accuracy_summary,
    _write_cost_summary,
    _write_details_csv,
    _write_run_metadata,
)

# =============================================================================
# Shared test fixtures and helpers
# =============================================================================


def _make_example(idx: int, source: str = "gsm8k") -> Example:
    """Build a minimal Example for use in tests."""
    return Example(
        id=f"{source}-test-{idx}",
        source=source,  # type: ignore[arg-type]
        prompt=f"What is 2 + {idx}?",
        response=f"Let me think. 2 + {idx} = {2 + idx}.\n#### {2 + idx}",
        final_answer=str(2 + idx),
    )


def _make_examples(n: int, source: str = "gsm8k") -> list[Example]:
    return [_make_example(i, source) for i in range(n)]


# A small fixed set of examples we reuse across multiple tests.
TWENTY_EXAMPLES = _make_examples(20)


# =============================================================================
# 1. Seeded subsampling determinism
# =============================================================================


def test_sample_examples_same_seed_same_indices() -> None:
    """The same seed always produces the same subset, in the same order."""
    first = _sample_examples(TWENTY_EXAMPLES, n=7, seed=42)
    second = _sample_examples(TWENTY_EXAMPLES, n=7, seed=42)
    assert [ex.id for ex in first] == [ex.id for ex in second]


def test_sample_examples_different_seeds_different_indices() -> None:
    """Different seeds produce different subsets (probabilistically true for n=7 from 20)."""
    a = _sample_examples(TWENTY_EXAMPLES, n=7, seed=1)
    b = _sample_examples(TWENTY_EXAMPLES, n=7, seed=2)
    # It is astronomically unlikely that two independent shuffles of 20 items
    # produce the same first 7 with different seeds. Assert the sets differ.
    assert [ex.id for ex in a] != [ex.id for ex in b]


def test_sample_examples_returns_exactly_n() -> None:
    """The returned list has exactly n elements."""
    result = _sample_examples(TWENTY_EXAMPLES, n=5, seed=99)
    assert len(result) == 5


def test_sample_examples_n_equals_total() -> None:
    """When n equals the pool size, all examples are returned (just shuffled)."""
    result = _sample_examples(TWENTY_EXAMPLES, n=20, seed=0)
    assert len(result) == 20
    assert set(ex.id for ex in result) == set(ex.id for ex in TWENTY_EXAMPLES)


def test_sample_examples_n_exceeds_pool_raises() -> None:
    """Asking for more examples than exist should raise, not silently truncate."""
    with pytest.raises(ValueError, match="n=25 exceeds pool size"):
        _sample_examples(TWENTY_EXAMPLES, n=25, seed=0)


def test_sample_examples_n_zero_raises() -> None:
    """n=0 must raise ValueError rather than silently return an empty list."""
    with pytest.raises(ValueError, match="n=0 must be at least 1"):
        _sample_examples(TWENTY_EXAMPLES, n=0, seed=0)


def test_sample_examples_n_negative_raises() -> None:
    """Negative n must raise ValueError."""
    with pytest.raises(ValueError, match="must be at least 1"):
        _sample_examples(TWENTY_EXAMPLES, n=-1, seed=0)


# =============================================================================
# 2. Batched-vs-single generation parity
# =============================================================================
#
# Property: greedy decoding is deterministic and must be independent of how
# the batch is split. A batch of 2 prompts decoded together must produce
# the same token sequence as the same 2 prompts decoded one at a time.
#
# This test downloads tiny-gpt2 (~1 MB, already cached in CI). It exercises
# left-padding correctness — if padding were on the right, the attention mask
# would be wrong and batch != single.


@pytest.mark.slow
def test_batched_vs_single_generation_parity() -> None:
    """Batch generation matches single generation for tiny-gpt2 on CPU."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_id = "sshleifer/tiny-gpt2"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    # Left-padding: required for batched decoder-only generation so all
    # prompts are right-aligned when the model reads position 0.
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        dtype=torch.float32,
    )
    model.eval()

    prompts = [
        "What is 2 + 2?",
        "Solve: if a train travels 60 miles per hour for 3 hours.",
    ]
    max_new_tokens = 16

    # Generate both prompts together in a single batch.
    # _generate_batch returns (texts, token_count, non_pad_tokens, generation_seconds).
    batch_texts, _batch_tokens, _batch_non_pad, _batch_gen_sec = _generate_batch(
        model=model,
        tokenizer=tokenizer,
        prompts=prompts,
        batch_size=2,
        max_new_tokens=max_new_tokens,
        device="cpu",
    )

    # Generate each prompt individually (batch_size=1).
    single_results = [
        _generate_batch(
            model=model,
            tokenizer=tokenizer,
            prompts=[p],
            batch_size=1,
            max_new_tokens=max_new_tokens,
            device="cpu",
        )[0][0]  # [0] = texts list, [0] = first (only) text in that list
        for p in prompts
    ]

    # The new tokens generated must be identical regardless of batch size.
    assert batch_texts[0] == single_results[0], (
        f"Prompt 0 differs:\n  batched:  {batch_texts[0]!r}\n  single:   {single_results[0]!r}"
    )
    assert batch_texts[1] == single_results[1], (
        f"Prompt 1 differs:\n  batched:  {batch_texts[1]!r}\n  single:   {single_results[1]!r}"
    )


# =============================================================================
# 3. OOM fallback
# =============================================================================
#
# The fallback must:
#   a. Halve batch_size on OOM and retry the SAME batch.
#   b. Succeed after the retry if the smaller batch fits.
#   c. Fail loudly (re-raise) if batch_size=1 still OOMs.
#
# We monkeypatch model.generate to inject OOM errors on demand.


def _make_tiny_model_and_tokenizer():
    """Load tiny-gpt2 on CPU in float32 — small and fast."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_id = "sshleifer/tiny-gpt2"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.float32)
    model.eval()
    return model, tokenizer


@pytest.mark.slow
def test_oom_fallback_halves_and_retries() -> None:
    """First call to generate raises OOM; second call (batch/2) succeeds."""
    model, tokenizer = _make_tiny_model_and_tokenizer()

    call_count = 0
    original_generate = model.generate

    def flaky_generate(*args: Any, **kwargs: Any):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            # Simulate OOM on the first attempt.
            raise torch.cuda.OutOfMemoryError("CUDA out of memory (simulated)")
        # Second call (halved batch) succeeds.
        return original_generate(*args, **kwargs)

    model.generate = flaky_generate  # type: ignore[method-assign]

    prompts = ["What is 1 + 1?", "What is 2 + 2?"]
    # With batch_size=2, the first call OOMs. The fallback retries at
    # batch_size=1, which succeeds.
    # _generate_batch returns (texts, token_count, non_pad_tokens, generation_seconds).
    texts, _token_count, _non_pad, _gen_sec = _generate_batch(
        model=model,
        tokenizer=tokenizer,
        prompts=prompts,
        batch_size=2,
        max_new_tokens=8,
        device="cpu",
    )

    # We should get back one generated string per prompt.
    assert len(texts) == 2
    # Exactly three generate calls: first attempt (batch_size=2) OOMed,
    # then the chunk was split into two halves each run at batch_size=1
    # (one call per half = two more calls). Total: 3.
    assert call_count == 3


@pytest.mark.slow
def test_oom_at_batch_size_one_raises() -> None:
    """OOM at batch_size=1 must raise, not loop forever."""
    model, tokenizer = _make_tiny_model_and_tokenizer()

    def always_oom(*args: Any, **kwargs: Any):
        raise torch.cuda.OutOfMemoryError("CUDA out of memory (simulated)")

    model.generate = always_oom  # type: ignore[method-assign]

    with pytest.raises(torch.cuda.OutOfMemoryError):
        _generate_batch(
            model=model,
            tokenizer=tokenizer,
            prompts=["What is 1 + 1?"],
            batch_size=1,
            max_new_tokens=8,
            device="cpu",
        )


# =============================================================================
# 4. Output artifact schemas
# =============================================================================
#
# These tests check the CSV and JSON writers in isolation using synthetic
# data — no model loading required. The schema is the contract between this
# issue and the downstream notebooks / compute-aware workstream.


_EXPECTED_SUMMARY_COLUMNS = {
    "checkpoint",
    "source",
    "n",
    "accuracy",
    "parse_success_rate",
    "generated_tokens",
    "generated_tokens_decoded",
    "elapsed_sec",
}

_EXPECTED_DETAILS_COLUMNS = {
    "example_id",
    "prompt",
    "generated",
    "gold_answer",
    "predicted_answer",
    "parse_success",
    "is_correct",
}

_EXPECTED_COST_FIELDS = {
    "run_name",
    "start_time",
    "end_time",
    "elapsed_sec",
    "generation_seconds",
    "gpu_type",
    "dtype",
    "generated_tokens",
    "generated_tokens_decoded",
    "tokens_per_second",
    "estimated_cost_usd",
}

_EXPECTED_METADATA_FIELDS = {
    "device",
    "dtype",
    "torch_version",
    "transformers_version",
    "cuda_version",
    "seed",
    "eval_n",
    "git_sha",
    "generation_settings",
    "checkpoints",
}


def test_accuracy_summary_csv_has_required_columns(tmp_path: Path) -> None:
    """accuracy_summary.csv contains exactly the required columns."""
    rows = [
        {
            "checkpoint": "base",
            "source": "gsm8k",
            "n": 4,
            "accuracy": 0.25,
            "parse_success_rate": 0.5,
            "generated_tokens": 100,
            "generated_tokens_decoded": 80,
            "elapsed_sec": 3.2,
        }
    ]
    _write_accuracy_summary(rows, out_dir=tmp_path)

    csv_path = tmp_path / "accuracy_summary.csv"
    assert csv_path.exists()

    with csv_path.open(newline="") as f:
        reader = csv.DictReader(f)
        assert set(reader.fieldnames or []) == _EXPECTED_SUMMARY_COLUMNS

        actual_rows = list(reader)
    assert len(actual_rows) == 1
    assert actual_rows[0]["checkpoint"] == "base"
    assert actual_rows[0]["source"] == "gsm8k"


def test_accuracy_summary_json_has_required_fields(tmp_path: Path) -> None:
    """accuracy_summary.json contains a list with the required keys per entry."""
    rows = [
        {
            "checkpoint": "combined",
            "source": "math",
            "n": 4,
            "accuracy": 0.0,
            "parse_success_rate": 0.0,
            "generated_tokens": 80,
            "generated_tokens_decoded": 64,
            "elapsed_sec": 2.1,
        }
    ]
    _write_accuracy_summary(rows, out_dir=tmp_path)

    json_path = tmp_path / "accuracy_summary.json"
    assert json_path.exists()

    data = json.loads(json_path.read_text())
    assert isinstance(data, list)
    assert len(data) == 1
    assert set(data[0].keys()) == _EXPECTED_SUMMARY_COLUMNS


def test_details_csv_has_required_columns(tmp_path: Path) -> None:
    """details_<ckpt>_<src>.csv has exactly the required columns."""
    detail_rows = [
        {
            "example_id": "gsm8k-test-0",
            "prompt": "What is 2 + 2?",
            "generated": "Let me think... #### 4",
            "gold_answer": "4",
            "predicted_answer": "4",
            "parse_success": True,
            "is_correct": True,
        }
    ]
    _write_details_csv(
        rows=detail_rows,
        checkpoint_name="base",
        source_name="gsm8k",
        out_dir=tmp_path,
    )

    csv_path = tmp_path / "details_base_gsm8k.csv"
    assert csv_path.exists()

    with csv_path.open(newline="") as f:
        reader = csv.DictReader(f)
        assert set(reader.fieldnames or []) == _EXPECTED_DETAILS_COLUMNS
        rows = list(reader)

    assert len(rows) == 1
    assert rows[0]["example_id"] == "gsm8k-test-0"


def test_details_csv_handles_newlines_in_generated(tmp_path: Path) -> None:
    """Newlines inside the 'generated' column are properly quoted, not breaking the row count."""
    detail_rows = [
        {
            "example_id": "gsm8k-test-0",
            "prompt": "A question",
            # Newlines and commas in the generated text: the CSV writer must
            # quote this field so the row-count remains 1 when read back.
            "generated": "Step 1: add.\nStep 2: answer is 42, final.\n#### 42",
            "gold_answer": "42",
            "predicted_answer": "42",
            "parse_success": True,
            "is_correct": True,
        }
    ]
    _write_details_csv(
        rows=detail_rows,
        checkpoint_name="ckpt",
        source_name="gsm8k",
        out_dir=tmp_path,
    )

    csv_path = tmp_path / "details_ckpt_gsm8k.csv"
    with csv_path.open(newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    # Must still parse as exactly one data row despite embedded newlines.
    assert len(rows) == 1
    assert "Step 1" in rows[0]["generated"]


def test_cost_summary_json_has_required_fields(tmp_path: Path) -> None:
    """cost_summary.json contains all required fields."""
    _write_cost_summary(
        out_dir=tmp_path,
        run_name="smoke_test",
        start_time="2026-05-11T10:00:00",
        end_time="2026-05-11T10:01:00",
        elapsed_sec=60.0,
        generation_seconds=55.0,
        gpu_type="CPU",
        dtype="float32",
        generated_tokens=500,
        generated_tokens_decoded=420,
        tokens_per_second=500 / 55.0,
        estimated_cost_usd=None,
    )

    json_path = tmp_path / "cost_summary.json"
    assert json_path.exists()

    data = json.loads(json_path.read_text())
    assert set(data.keys()) == _EXPECTED_COST_FIELDS


def test_cost_summary_estimated_cost_null_when_not_supplied(tmp_path: Path) -> None:
    """estimated_cost_usd is JSON null when gpu_cost_per_hour is not supplied."""
    _write_cost_summary(
        out_dir=tmp_path,
        run_name="no_cost",
        start_time="2026-05-11T10:00:00",
        end_time="2026-05-11T10:00:10",
        elapsed_sec=10.0,
        generation_seconds=9.0,
        gpu_type="CPU",
        dtype="float32",
        generated_tokens=50,
        generated_tokens_decoded=40,
        tokens_per_second=50 / 9.0,
        estimated_cost_usd=None,
    )

    data = json.loads((tmp_path / "cost_summary.json").read_text())
    assert data["estimated_cost_usd"] is None


def test_cost_summary_estimated_cost_populated_when_supplied(tmp_path: Path) -> None:
    """estimated_cost_usd is a float when gpu_cost_per_hour is supplied."""
    # 1 hour at $1.39/hr, 3600 seconds elapsed → $1.39
    _write_cost_summary(
        out_dir=tmp_path,
        run_name="with_cost",
        start_time="2026-05-11T10:00:00",
        end_time="2026-05-11T11:00:00",
        elapsed_sec=3600.0,
        generation_seconds=3550.0,
        gpu_type="Tesla T4",
        dtype="bfloat16",
        generated_tokens=10000,
        generated_tokens_decoded=8500,
        tokens_per_second=10000 / 3550.0,
        estimated_cost_usd=1.39,
    )

    data = json.loads((tmp_path / "cost_summary.json").read_text())
    assert data["estimated_cost_usd"] == pytest.approx(1.39)


def test_run_metadata_has_required_fields(tmp_path: Path) -> None:
    """run_metadata.json contains all required fields."""
    generation_settings = {
        "gsm8k": {"max_new_tokens": 256, "batch_size": 8},
    }
    checkpoints = {"base": "Qwen/Qwen2.5-0.5B"}
    _write_run_metadata(
        out_dir=tmp_path,
        device="cpu",
        dtype="float32",
        seed=42,
        eval_n=4,
        generation_settings=generation_settings,
        checkpoints=checkpoints,
    )

    json_path = tmp_path / "run_metadata.json"
    assert json_path.exists()

    data = json.loads(json_path.read_text())
    # Check all required top-level keys are present.
    for field in _EXPECTED_METADATA_FIELDS:
        assert field in data, f"Missing field: {field!r}"

    # device must be a friendly name: "CPU" (not "cpu") when on CPU.
    assert data["device"] == "CPU", (
        f"Expected friendly device name 'CPU', got {data['device']!r}"
    )
    assert data["dtype"] == "float32"
    assert data["seed"] == 42
    assert data["eval_n"] == 4
    assert "gsm8k" in data["generation_settings"]
    # checkpoints field must round-trip the supplied dict.
    assert data["checkpoints"] == {"base": "Qwen/Qwen2.5-0.5B"}, (
        f"checkpoints field mismatch: {data['checkpoints']!r}"
    )


# =============================================================================
# 5. Re-run byte-identity
# =============================================================================
#
# The test runs the CLI main function twice with the same arguments,
# captures the details CSV bytes, and asserts they are identical.
# This confirms no timestamp, nondeterministic dict ordering, or random
# element has crept into the details output file.
#
# We use the tiny-gpt2 model on CPU with n=2 to keep the test fast.
# The details CSV should have no run-specific timestamps (those go to
# cost_summary.json / run_metadata.json, not details).


@pytest.mark.slow
def test_byte_identical_details_csv_on_rerun(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Two runs with the same seed produce byte-identical details_*.csv.

    We monkeypatch source.load_examples to return a fixed synthetic list
    so the test does not require a network connection. The byte-identity
    property being tested is about the generation loop and CSV writer, not
    about dataset loading.
    """
    from finpost.evals import sources as sources_module
    from finpost.evals.eval_exact import run_eval

    # Build a fixed pool of 10 synthetic examples.
    pool = _make_examples(10, source="gsm8k")

    # Replace the gsm8k EvalSource with one that uses our synthetic pool.
    # We create a new EvalSource with a patched load_examples thunk.
    original_source = sources_module.REGISTRY["gsm8k"]
    patched_source = sources_module.EvalSource(
        name=original_source.name,
        load_examples=lambda: pool,  # return synthetic pool, no network call
        extract_answer=original_source.extract_answer,
        score=original_source.score,
        default_max_new_tokens=original_source.default_max_new_tokens,
    )
    monkeypatch.setitem(sources_module.REGISTRY, "gsm8k", patched_source)

    # Minimal settings for a fast, deterministic run.
    kwargs = dict(
        checkpoints={"tiny": "sshleifer/tiny-gpt2"},
        sources=["gsm8k"],
        n=4,
        seed=42,
        out_dir=tmp_path / "run1",
        batch_sizes={"gsm8k": 2},
        gpu_cost_per_hour=None,
        device="cpu",
    )

    run_eval(**kwargs)
    run_eval(**{**kwargs, "out_dir": tmp_path / "run2"})

    details1 = (tmp_path / "run1" / "details_tiny_gsm8k.csv").read_bytes()
    details2 = (tmp_path / "run2" / "details_tiny_gsm8k.csv").read_bytes()

    assert details1 == details2, (
        "details_*.csv is not byte-identical across runs with the same seed. "
        "Check for timestamps, non-deterministic ordering, or random values "
        "in the per-example output."
    )


# =============================================================================
# 6. RunTracker context manager
# =============================================================================


def test_run_tracker_records_elapsed_and_tokens(tmp_path: Path) -> None:
    """RunTracker records elapsed time and accumulated token count."""
    tracker = RunTracker(out_dir=tmp_path, run_name="test_run", gpu_cost_per_hour=None)

    with tracker:
        tracker.add_generated_tokens(100)
        tracker.add_generated_tokens(50)

    # After __exit__, elapsed_sec is a non-negative float (could be ~0 on
    # fast machines; the key check is that the field was set and tokens summed).
    assert tracker.elapsed_sec >= 0.0
    assert tracker.total_generated_tokens == 150




# =============================================================================
# 6b. Non-pad token counting (MEDIUM 7)
# =============================================================================
#
# _count_non_pad_tokens is the helper that turns a (batch, new_tokens) tensor
# into the "decoded" (content-length) count. The whole point of the MEDIUM 7
# fix is that this number diverges from new_token_ids.numel() when one
# sequence in a batch hits EOS before others, so HuggingFace's generate()
# fills the trailing positions with pad_token_id. tiny-gpt2 with the budgets
# we use in slow tests does not hit EOS, so an end-to-end test never
# exercises the divergence and a silently-broken implementation that just
# returns numel() would still pass. These tests use a synthetic tensor so
# the divergent path IS exercised.


def test_count_non_pad_tokens_excludes_pad_fill() -> None:
    """A batch with one early-stop and one full-length sequence: count excludes pad fill."""
    pad_id = 0
    # Row 0: 3 real tokens, then EOS-style pad fill in the last 2 slots.
    # Row 1: 5 real tokens, no pad fill.
    new_token_ids = torch.tensor(
        [
            [1, 2, 3, pad_id, pad_id],
            [4, 5, 6, 7, 8],
        ]
    )
    # Rectangular count is 10 (2 rows × 5 cols); non-pad is 3 + 5 = 8.
    assert new_token_ids.numel() == 10
    assert _count_non_pad_tokens(new_token_ids, pad_id) == 8


def test_count_non_pad_tokens_no_pad_in_batch() -> None:
    """When no pad fill exists, non-pad count equals rectangular count."""
    pad_id = 0
    new_token_ids = torch.tensor([[1, 2, 3], [4, 5, 6]])
    assert _count_non_pad_tokens(new_token_ids, pad_id) == 6


def test_count_non_pad_tokens_all_pad() -> None:
    """An entirely-pad tensor (degenerate case) returns 0."""
    pad_id = 0
    new_token_ids = torch.tensor([[pad_id, pad_id], [pad_id, pad_id]])
    assert _count_non_pad_tokens(new_token_ids, pad_id) == 0


def test_count_non_pad_tokens_pad_id_none_falls_back_to_numel() -> None:
    """When the tokenizer has no pad_token_id, fall back to the rectangular count."""
    new_token_ids = torch.tensor([[1, 2, 3], [4, 5, 6]])
    assert _count_non_pad_tokens(new_token_ids, None) == 6


def test_count_non_pad_tokens_returns_python_int() -> None:
    """Return type must be int (not torch.Tensor) so it sums cleanly across batches."""
    new_token_ids = torch.tensor([[1, 0], [1, 0]])
    result = _count_non_pad_tokens(new_token_ids, 0)
    assert type(result) is int


# =============================================================================
# 7. CUDA determinism flags (Bug 5)
# =============================================================================
#
# _set_cuda_determinism must set the three torch determinism flags and the
# CUBLAS_WORKSPACE_CONFIG env var when device starts with "cuda".
# On CPU it must be a no-op so we don't waste cycles on non-CUDA runs.


def test_cuda_determinism_flags_set_on_cuda_device(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """_set_cuda_determinism sets all required flags when device='cuda'."""
    # Capture calls to torch.use_deterministic_algorithms.
    calls: list[tuple] = []

    def mock_use_deterministic(val: bool, warn_only: bool = False) -> None:
        calls.append((val, warn_only))

    monkeypatch.setattr(torch, "use_deterministic_algorithms", mock_use_deterministic)

    # Capture the pre-call cudnn state so we can restore it after the test
    # (avoids polluting the process state for subsequent tests).
    original_det = torch.backends.cudnn.deterministic
    original_bench = torch.backends.cudnn.benchmark

    _set_cuda_determinism("cuda")

    # torch.use_deterministic_algorithms must have been called once with
    # (True, warn_only=True).
    assert len(calls) == 1, (
        f"Expected 1 call to torch.use_deterministic_algorithms, got {len(calls)}"
    )
    assert calls[0] == (True, True), (
        f"Expected (True, warn_only=True), got {calls[0]}"
    )

    # torch.backends.cudnn.deterministic must be True.
    assert torch.backends.cudnn.deterministic is True, (
        "torch.backends.cudnn.deterministic was not set to True"
    )
    # torch.backends.cudnn.benchmark must be False.
    assert torch.backends.cudnn.benchmark is False, (
        "torch.backends.cudnn.benchmark was not set to False"
    )

    # CUBLAS_WORKSPACE_CONFIG must be set.
    import os
    assert "CUBLAS_WORKSPACE_CONFIG" in os.environ, (
        "CUBLAS_WORKSPACE_CONFIG env var was not set"
    )

    # Restore cudnn settings to avoid polluting other tests.
    torch.backends.cudnn.deterministic = original_det
    torch.backends.cudnn.benchmark = original_bench


def test_cuda_determinism_flags_not_set_on_cpu(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """_set_cuda_determinism is a no-op when device='cpu'."""
    calls: list[tuple] = []

    def mock_use_deterministic(val: bool, warn_only: bool = False) -> None:
        calls.append((val, warn_only))

    monkeypatch.setattr(torch, "use_deterministic_algorithms", mock_use_deterministic)

    _set_cuda_determinism("cpu")

    assert calls == [], (
        "torch.use_deterministic_algorithms should NOT be called for device='cpu'"
    )


# =============================================================================
# 8. OOM fallback empty-list guard (Bug 6)
# =============================================================================
#
# _generate_chunk_with_oom_fallback must return ([], 0, batch_size) immediately
# when prompts is empty. Without this guard, the recursive split on a
# 1-prompt chunk that OOMs produces an empty right half and passes it back
# through _tokenize_and_generate, eventually crashing.


@pytest.mark.slow
def test_generate_chunk_empty_prompts_returns_empty() -> None:
    """_generate_chunk_with_oom_fallback([]) returns ([], 0, batch_size) immediately."""
    model, tokenizer = _make_tiny_model_and_tokenizer()

    result = _generate_chunk_with_oom_fallback(
        model=model,
        tokenizer=tokenizer,
        prompts=[],
        batch_size=4,
        max_new_tokens=8,
        device="cpu",
    )

    # Must return ([], 0, 0, 4, 0.0) — texts empty, zero rectangular tokens,
    # zero non-pad tokens, batch_size unchanged, zero generation seconds.
    assert result == ([], 0, 0, 4, 0.0), (
        f"Expected ([], 0, 0, 4, 0.0) for empty prompts, got {result!r}"
    )


@pytest.mark.slow
def test_generate_chunk_oom_on_single_prompt_at_batch_size_gt_1_recovers() -> None:
    """OOM on a 1-prompt chunk at batch_size=2 recovers via the OOM fallback.

    This is the exact bug path: batch_size=2, 1 prompt, OOM fires.
    The split produces prompts[:1] = [prompt] and prompts[1:] = [].
    Without the empty-list guard the empty right half crashes; with it
    the function returns the single generated text cleanly.
    """
    model, tokenizer = _make_tiny_model_and_tokenizer()

    call_count = 0
    original_generate = model.generate

    def oom_once(*args: Any, **kwargs: Any):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            raise torch.cuda.OutOfMemoryError("CUDA out of memory (simulated)")
        return original_generate(*args, **kwargs)

    model.generate = oom_once  # type: ignore[method-assign]

    texts, token_count, non_pad_count, effective_bs, gen_sec = _generate_chunk_with_oom_fallback(
        model=model,
        tokenizer=tokenizer,
        prompts=["What is 1 + 1?"],
        batch_size=2,
        max_new_tokens=8,
        device="cpu",
    )

    # Must return exactly one text and a positive token count.
    assert len(texts) == 1, f"Expected 1 text, got {len(texts)}"
    assert token_count > 0, "token_count should be > 0 after successful generation"
    # non_pad_count must not exceed the rectangular token_count, and should
    # be positive whenever generation actually emitted content.
    assert 0 < non_pad_count <= token_count, (
        f"non_pad_count={non_pad_count} must satisfy 0 < non_pad_count <= "
        f"token_count={token_count}"
    )
    # effective_bs should be 1 (the halved size that actually succeeded).
    assert effective_bs == 1, f"Expected effective_bs=1, got {effective_bs}"
    # gen_sec should be a non-negative float.
    assert gen_sec >= 0.0, f"Expected gen_sec >= 0.0, got {gen_sec}"


# =============================================================================
# S1. Path-traversal guard on checkpoint names
# =============================================================================
#
# The checkpoint name flows directly into the output filename
# ``details_{name}_{source}.csv``.  Names containing path separators or ..
# must be rejected at parse time, before any model loading.


def test_parse_checkpoint_pair_valid_names_accepted() -> None:
    """Well-formed names parse without error."""
    assert _parse_checkpoint_pair("base=Qwen/Qwen2.5-0.5B") == ("base", "Qwen/Qwen2.5-0.5B")
    assert _parse_checkpoint_pair("combined_step_1000=/tmp/ckpt") == (
        "combined_step_1000",
        "/tmp/ckpt",
    )
    assert _parse_checkpoint_pair("qwen-0.5b=sshleifer/tiny-gpt2") == (
        "qwen-0.5b",
        "sshleifer/tiny-gpt2",
    )
    # Verify dot is allowed in the name (M4 allowlist: letters/digits/dot/underscore/hyphen).
    assert _parse_checkpoint_pair("gpt.tiny=sshleifer/tiny-gpt2") == (
        "gpt.tiny",
        "sshleifer/tiny-gpt2",
    )


def test_parse_checkpoint_pair_dotdot_rejected() -> None:
    """A name containing '..' raises ArgumentTypeError (path traversal guard).

    '../' is rejected because '/' is not in the allowlist — the regex
    rejects the entire name at the '/' character.
    """
    with pytest.raises(argparse.ArgumentTypeError, match="must match"):
        _parse_checkpoint_pair("../etc=sshleifer/tiny-gpt2")


def test_parse_checkpoint_pair_forward_slash_in_name_rejected() -> None:
    """A name containing '/' raises ArgumentTypeError."""
    with pytest.raises(argparse.ArgumentTypeError, match="must match"):
        _parse_checkpoint_pair("a/b=sshleifer/tiny-gpt2")


def test_parse_checkpoint_pair_backslash_in_name_rejected() -> None:
    r"""A name containing '\\' raises ArgumentTypeError."""
    with pytest.raises(argparse.ArgumentTypeError, match="must match"):
        _parse_checkpoint_pair("a\\b=sshleifer/tiny-gpt2")


def test_parse_checkpoint_pair_null_byte_in_name_rejected() -> None:
    """A name containing a null byte raises ArgumentTypeError."""
    with pytest.raises(argparse.ArgumentTypeError, match="must match"):
        _parse_checkpoint_pair("a\x00b=sshleifer/tiny-gpt2")


def test_parse_checkpoint_pair_space_in_name_rejected() -> None:
    """A name containing a space raises ArgumentTypeError."""
    with pytest.raises(argparse.ArgumentTypeError, match="must match"):
        _parse_checkpoint_pair("my run=sshleifer/tiny-gpt2")


def test_parse_checkpoint_pair_accented_char_in_name_rejected() -> None:
    """A name containing an accented character raises ArgumentTypeError."""
    with pytest.raises(argparse.ArgumentTypeError, match="must match"):
        _parse_checkpoint_pair("café=sshleifer/tiny-gpt2")


def test_parse_checkpoint_pair_tab_in_name_rejected() -> None:
    """A name containing a tab character raises ArgumentTypeError."""
    with pytest.raises(argparse.ArgumentTypeError, match="must match"):
        _parse_checkpoint_pair("a\tb=sshleifer/tiny-gpt2")


# =============================================================================
# RG1. Non-empty --out-dir warning
# =============================================================================
#
# When run_eval is called on a directory that already contains files, a
# warning must be printed to stderr so the operator notices the potential
# overwrite before it happens.


@pytest.mark.slow
def test_run_eval_warns_on_nonempty_outdir(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture,
) -> None:
    """run_eval prints a stderr warning when --out-dir is non-empty."""
    from finpost.evals import sources as sources_module
    from finpost.evals.eval_exact import run_eval

    pool = _make_examples(10, source="gsm8k")
    original_source = sources_module.REGISTRY["gsm8k"]
    patched_source = sources_module.EvalSource(
        name=original_source.name,
        load_examples=lambda: pool,
        extract_answer=original_source.extract_answer,
        score=original_source.score,
        default_max_new_tokens=original_source.default_max_new_tokens,
    )
    monkeypatch.setitem(sources_module.REGISTRY, "gsm8k", patched_source)

    out_dir = tmp_path / "run_rg1"
    kwargs = dict(
        checkpoints={"tiny": "sshleifer/tiny-gpt2"},
        sources=["gsm8k"],
        n=4,
        seed=42,
        out_dir=out_dir,
        batch_sizes={"gsm8k": 2},
        gpu_cost_per_hour=None,
        device="cpu",
    )

    # First run: directory is freshly created — no warning.
    run_eval(**kwargs)
    captured = capsys.readouterr()
    assert "WARNING" not in captured.err, (
        "No warning expected on first run into an empty directory"
    )

    # Second run: directory now contains the artifacts from the first run.
    run_eval(**kwargs)
    captured = capsys.readouterr()
    assert "WARNING" in captured.err, (
        "Expected a non-empty-dir warning on the second run"
    )
    assert "non-empty" in captured.err.lower()


# =============================================================================
# RG2. Pre-run cost estimate
# =============================================================================
#
# When --gpu-cost-per-hour is supplied, a cost estimate must appear on
# stdout BEFORE any model loading.  We verify the estimate appears when the
# flag is set and is absent when it is not.


@pytest.mark.slow
def test_run_eval_prints_cost_estimate_when_flag_set(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture,
) -> None:
    """run_eval prints a workload + cost estimate when gpu_cost_per_hour is provided."""
    from finpost.evals import sources as sources_module
    from finpost.evals.eval_exact import run_eval

    pool = _make_examples(10, source="gsm8k")
    original_source = sources_module.REGISTRY["gsm8k"]
    patched_source = sources_module.EvalSource(
        name=original_source.name,
        load_examples=lambda: pool,
        extract_answer=original_source.extract_answer,
        score=original_source.score,
        default_max_new_tokens=original_source.default_max_new_tokens,
    )
    monkeypatch.setitem(sources_module.REGISTRY, "gsm8k", patched_source)

    run_eval(
        checkpoints={"tiny": "sshleifer/tiny-gpt2"},
        sources=["gsm8k"],
        n=4,
        seed=42,
        out_dir=tmp_path / "rg2_with_cost",
        batch_sizes={"gsm8k": 2},
        gpu_cost_per_hour=0.79,
        device="cpu",
    )
    captured = capsys.readouterr()
    # Workload line always printed.
    assert "Workload estimate" in captured.out
    assert "generations" in captured.out
    # Cost line only printed when gpu_cost_per_hour is set.
    assert "~100 tok/s" in captured.out
    assert "$0." in captured.out or "$" in captured.out


@pytest.mark.slow
def test_run_eval_no_cost_estimate_when_flag_absent(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture,
) -> None:
    """run_eval does NOT print a cost estimate when gpu_cost_per_hour is None."""
    from finpost.evals import sources as sources_module
    from finpost.evals.eval_exact import run_eval

    pool = _make_examples(10, source="gsm8k")
    original_source = sources_module.REGISTRY["gsm8k"]
    patched_source = sources_module.EvalSource(
        name=original_source.name,
        load_examples=lambda: pool,
        extract_answer=original_source.extract_answer,
        score=original_source.score,
        default_max_new_tokens=original_source.default_max_new_tokens,
    )
    monkeypatch.setitem(sources_module.REGISTRY, "gsm8k", patched_source)

    run_eval(
        checkpoints={"tiny": "sshleifer/tiny-gpt2"},
        sources=["gsm8k"],
        n=4,
        seed=42,
        out_dir=tmp_path / "rg2_no_cost",
        batch_sizes={"gsm8k": 2},
        gpu_cost_per_hour=None,
        device="cpu",
    )
    captured = capsys.readouterr()
    # Workload line still printed.
    assert "Workload estimate" in captured.out
    # Cost line must NOT appear.
    assert "~100 tok/s" not in captured.out


# =============================================================================
# RG3. OOM-halving summary
# =============================================================================
#
# After a run that triggered OOM fallbacks, run_eval must print a summary
# to stderr.  We monkeypatch model.generate to inject a single OOM.


@pytest.mark.slow
def test_run_eval_prints_oom_summary_after_oom(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture,
) -> None:
    """run_eval prints OOM-halving summary to stderr when OOM occurred."""
    import finpost.evals.eval_exact as eval_exact_module
    from finpost.evals import sources as sources_module
    from finpost.evals.eval_exact import run_eval

    pool = _make_examples(10, source="gsm8k")
    original_source = sources_module.REGISTRY["gsm8k"]
    patched_source = sources_module.EvalSource(
        name=original_source.name,
        load_examples=lambda: pool,
        extract_answer=original_source.extract_answer,
        score=original_source.score,
        default_max_new_tokens=original_source.default_max_new_tokens,
    )
    monkeypatch.setitem(sources_module.REGISTRY, "gsm8k", patched_source)

    # Intercept model loading to inject a flaky generate().
    original_load = eval_exact_module._load_model_and_tokenizer

    def patched_load(model_path, dtype, device):
        model, tokenizer = original_load(model_path, dtype=dtype, device=device)
        call_count = 0
        original_generate = model.generate

        def oom_once(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise torch.cuda.OutOfMemoryError("CUDA out of memory (simulated)")
            return original_generate(*args, **kwargs)

        model.generate = oom_once  # type: ignore[method-assign]
        return model, tokenizer

    monkeypatch.setattr(eval_exact_module, "_load_model_and_tokenizer", patched_load)

    run_eval(
        checkpoints={"tiny": "sshleifer/tiny-gpt2"},
        sources=["gsm8k"],
        n=4,
        seed=42,
        out_dir=tmp_path / "rg3_oom",
        batch_sizes={"gsm8k": 2},
        gpu_cost_per_hour=None,
        device="cpu",
    )
    captured = capsys.readouterr()
    assert "OOM fallback halved batch size" in captured.err
    assert "batch-size" in captured.err


# =============================================================================
# RB1. _strip_string normalization failure logging
# =============================================================================
#
# When score_math's _strip_string call raises, the failure must be counted.
# At the end of run_eval the count must appear in stderr.


def test_strip_string_failure_increments_counter() -> None:
    """score_math increments _strip_string_failure_count when _strip_string raises."""
    import finpost.evals.sources as sources_module
    from finpost.evals.sources import score_math

    # Reset the counter to a known baseline.
    sources_module._strip_string_failure_count = 0

    # Inject a pathological string that will trigger _remove_right_units's
    # bare assert (two '\text{ ' markers).  The assert fires inside _strip_string,
    # score_math's except-block catches it and increments the counter.
    bad_latex = r"3\text{ apples}\text{ oranges}"
    score_math(bad_latex, "different")

    assert sources_module._strip_string_failure_count == 1, (
        f"Expected _strip_string_failure_count=1, "
        f"got {sources_module._strip_string_failure_count}"
    )
