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
    _generate_batch,
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
    # _generate_batch returns (texts, token_count); unpack accordingly.
    batch_texts, _batch_tokens = _generate_batch(
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
    # _generate_batch returns (texts, token_count).
    texts, _token_count = _generate_batch(
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
    "gpu_type",
    "dtype",
    "generated_tokens",
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
        gpu_type="CPU",
        dtype="float32",
        generated_tokens=500,
        tokens_per_second=8.3,
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
        gpu_type="CPU",
        dtype="float32",
        generated_tokens=50,
        tokens_per_second=5.0,
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
        gpu_type="Tesla T4",
        dtype="bfloat16",
        generated_tokens=10000,
        tokens_per_second=2.77,
        estimated_cost_usd=1.39,
    )

    data = json.loads((tmp_path / "cost_summary.json").read_text())
    assert data["estimated_cost_usd"] == pytest.approx(1.39)


def test_run_metadata_has_required_fields(tmp_path: Path) -> None:
    """run_metadata.json contains all required fields."""
    generation_settings = {
        "gsm8k": {"max_new_tokens": 256, "batch_size": 8},
    }
    _write_run_metadata(
        out_dir=tmp_path,
        device="cpu",
        dtype="float32",
        seed=42,
        eval_n=4,
        generation_settings=generation_settings,
    )

    json_path = tmp_path / "run_metadata.json"
    assert json_path.exists()

    data = json.loads(json_path.read_text())
    # Check all required top-level keys are present.
    for field in _EXPECTED_METADATA_FIELDS:
        assert field in data, f"Missing field: {field!r}"

    assert data["device"] == "cpu"
    assert data["dtype"] == "float32"
    assert data["seed"] == 42
    assert data["eval_n"] == 4
    assert "gsm8k" in data["generation_settings"]


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


def test_run_tracker_writes_cost_summary(tmp_path: Path) -> None:
    """RunTracker.write() produces a well-formed cost_summary.json."""
    tracker = RunTracker(out_dir=tmp_path, run_name="run_tracker_test", gpu_cost_per_hour=None)
    with tracker:
        tracker.add_generated_tokens(200)
    tracker.write()

    json_path = tmp_path / "cost_summary.json"
    assert json_path.exists()
    data = json.loads(json_path.read_text())
    assert data["generated_tokens"] == 200
    assert data["estimated_cost_usd"] is None


def test_run_tracker_estimates_cost_when_rate_supplied(tmp_path: Path) -> None:
    """RunTracker computes estimated_cost_usd from elapsed_sec and rate."""
    tracker = RunTracker(
        out_dir=tmp_path,
        run_name="costed_run",
        gpu_cost_per_hour=3.6,  # $3.60/hr = $0.001/sec
    )
    with tracker:
        tracker.add_generated_tokens(10)
    tracker.write()

    data = json.loads((tmp_path / "cost_summary.json").read_text())
    # estimated_cost_usd is a float (not None) when a rate is supplied.
    # The exact value depends on elapsed time; we only check the type here.
    assert data["estimated_cost_usd"] is not None
    assert isinstance(data["estimated_cost_usd"], float)
    # Cost is non-negative: elapsed_sec >= 0, rate > 0.
    assert data["estimated_cost_usd"] >= 0.0


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
    # Record the cudnn attribute changes via a simple flag.
    deterministic_values: list[bool] = []
    benchmark_values: list[bool] = []

    # torch.backends.cudnn is a module-level object; capture attribute sets
    # by wrapping the assignments in a custom context.
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
