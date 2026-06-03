"""Exact-answer evaluation CLI for FinPost post-training experiments.

Usage (from the repo root)::

    python -m finpost.evals.eval_exact \\
        --checkpoints base=Qwen/Qwen2.5-0.5B combined=results/checkpoints/step-00003000 \\
        --sources finchain \\
        --n 500 \\
        --seed 42 \\
        --out-dir results/evals/base_vs_sft/ \\
        --batch-size-finchain 32 \\
        --gpu-cost-per-hour 1.39

The CLI loads one checkpoint at a time, evaluates it on every requested
source, frees GPU memory, then loads the next checkpoint. This means the
peak GPU memory footprint is one model at a time, which is required on
free-tier T4 GPUs with only 15 GB VRAM.

Five output files are written to ``--out-dir``:

``accuracy_summary.json``
    List of {checkpoint, source, n, accuracy, parse_success_rate,
    generated_tokens, elapsed_sec}.

``accuracy_summary.csv``
    Same data, CSV format.

``details_<checkpoint>_<source>.csv``
    One row per evaluated example: example_id, prompt, generated,
    gold_answer, predicted_answer, parse_success, is_correct.

``run_metadata.json``
    Device name, library versions, dtype, seed, generation settings per
    source, git short SHA.

``cost_summary.json``
    Start/end timestamps, elapsed seconds, GPU type, dtype, token count,
    tokens per second, optional dollar-cost estimate.

Design notes for anyone reading this to learn from it
------------------------------------------------------
- ``_generate_batch`` is explicit: tokenize with left-padding, call
  ``model.generate``, slice off the prompt tokens before decoding. There
  is no "Trainer.predict()" magic wrapper hiding the generation loop.
- Left-padding is used (not right-padding) because decoder-only models
  read from the right edge of the context window. The last token of the
  prompt must be the token the model conditions its first generated token
  on. With right-padding the padding tokens would sit between the prompt
  and the model's output, which corrupts the attention.
- Batch OOM fallback: we catch ``torch.cuda.OutOfMemoryError`` specifically
  (not the broader ``RuntimeError``) so we never accidentally absorb
  unrelated bugs.
- Best-effort byte-identical re-runs: the details CSV contains no timestamps
  or run-specific state. The sample order comes from a seeded shuffle. Greedy
  decoding is fully deterministic on CPU. On CUDA, byte-identity is best-effort:
  ``_set_cuda_determinism`` sets cuDNN determinism flags before any model loading
  when device starts with ``"cuda"``, but ``warn_only=True`` means operations
  that have no deterministic implementation emit a warning and continue rather
  than aborting. Full byte-identity across CUDA re-runs is only guaranteed when
  all ops have deterministic implementations; CPU re-runs are fully deterministic.
"""

from __future__ import annotations

import argparse
import csv
import datetime
import json
import os
import random
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import torch

from finpost.data.finchain_dataset import load_finchain
from finpost.evals.chaineval import (
    CHAINEVAL_AVAILABLE,
    CHAINEVAL_DETAIL_COLUMNS,
    ChainEvalNotAvailable,
    compute_step_f1,
    score_trace,
)
from finpost.evals.sources import REGISTRY, EvalSource
from finpost.safety import safe_load_model, safe_load_tokenizer

# =============================================================================
# Module-level OOM-halving counter (RG3).
#
# _generate_chunk_with_oom_fallback increments this each time it halves the
# batch size. run_eval resets it at entry and prints a summary at exit if
# any OOMs occurred, so the operator knows to lower --batch-size-* for future
# runs. A module-level mutable is intentional: eval_exact is a single-process
# CLI with no concurrency, and this avoids threading a new counter value
# through the entire generation call stack.
# =============================================================================

_oom_halving_count: int = 0

# =============================================================================
# CUDA determinism helper
# =============================================================================


def _set_cuda_determinism(device: str) -> None:
    """Set CUDA determinism flags when running on GPU.

    cuDNN's algorithm selector picks the fastest kernel per operation, which
    can vary across runs and produce different argmax results on logit ties.
    These flags request deterministic kernels so ``details_*.csv`` is
    best-effort byte-identical across re-runs with the same seed on the same
    device. Full byte-identity is only guaranteed on CPU; on CUDA,
    ``warn_only=True`` means operations that have no deterministic
    implementation emit a warning and continue rather than aborting.

    Must be called BEFORE any CUDA tensor allocation so that
    CUBLAS_WORKSPACE_CONFIG takes effect before the CUDA context is
    initialized.

    Parameters
    ----------
    device
        The device string from the CLI (e.g. ``"cuda"`` or ``"cpu"``).
        Does nothing if device does not start with ``"cuda"``.
    """
    if not device.startswith("cuda"):
        # CPU runs are already fully deterministic; no flags needed.
        return

    # CUBLAS_WORKSPACE_CONFIG must be in the environment before CUDA context
    # init. setdefault means we do not override a value the user explicitly set.
    # The only NVIDIA-approved values for deterministic behaviour are
    # ":4096:8" and ":16:8". Warn if the env var is already set to something
    # else so the operator knows determinism may be degraded.
    _VALID_CUBLAS_CONFIGS = {":4096:8", ":16:8"}
    existing_config = os.environ.get("CUBLAS_WORKSPACE_CONFIG")
    if existing_config is not None and existing_config not in _VALID_CUBLAS_CONFIGS:
        print(
            f"[eval_exact] WARNING: CUBLAS_WORKSPACE_CONFIG={existing_config!r} "
            f"is not a deterministic value (expected one of {_VALID_CUBLAS_CONFIGS}). "
            f"CUDA determinism may be degraded.",
            file=sys.stderr,
        )
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

    # warn_only=True: some operations have no deterministic implementation.
    # We log a warning instead of crashing — the intent is determinism where
    # possible, not an abort-on-first-indeterminate-op policy.
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# =============================================================================
# RunTracker — inline cost and throughput helper (~40 lines)
# =============================================================================


class RunTracker:
    """Record start/end timestamps, token count, and optional dollar cost.

    Usage::

        tracker = RunTracker(out_dir=Path("results/"), run_name="run_01",
                             gpu_cost_per_hour=1.39)
        with tracker:
            # ... eval loop ...
            tracker.add_generated_tokens(n)
        # After __exit__, inspect tracker.elapsed_sec, tracker.gpu_type, etc.
        # run_eval calls _write_cost_summary directly (with the correct dtype
        # and generation_seconds) rather than via a write() method here.

    Attributes are public so tests can inspect them after ``__exit__``.
    """

    def __init__(
        self,
        out_dir: Path,
        run_name: str,
        gpu_cost_per_hour: float | None,
    ) -> None:
        self.out_dir = out_dir
        self.run_name = run_name
        self.gpu_cost_per_hour = gpu_cost_per_hour

        # Set by __enter__ / __exit__. Public so tests can read them.
        self.start_time: str = ""
        self.end_time: str = ""
        self.elapsed_sec: float = 0.0
        self.gpu_type: str = "CPU"
        self.total_generated_tokens: int = 0
        self.total_decoded_tokens: int = 0
        self._start_monotonic: float = 0.0

    def __enter__(self) -> RunTracker:
        # ISO-8601 timestamp (local time). Only in run metadata; never
        # inside the per-example details CSV, so byte-identity is preserved.
        self.start_time = datetime.datetime.now().isoformat(timespec="seconds")
        self._start_monotonic = time.monotonic()

        # Detect GPU type. On CPU-only runs, gpu_type stays "CPU".
        if torch.cuda.is_available():
            self.gpu_type = torch.cuda.get_device_name(0)

        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.end_time = datetime.datetime.now().isoformat(timespec="seconds")
        self.elapsed_sec = time.monotonic() - self._start_monotonic

    def add_generated_tokens(self, n: int) -> None:
        """Accumulate rectangular position count across all batches and sources."""
        self.total_generated_tokens += n

    def add_decoded_tokens(self, n: int) -> None:
        """Accumulate non-pad token count (content length) across all batches."""
        self.total_decoded_tokens += n


# =============================================================================
# Seeded subsampling
# =============================================================================


def _sample_examples(
    examples: list[Any],
    n: int,
    seed: int,
) -> list[Any]:
    """Return a seeded random subset of ``n`` examples.

    The same ``seed`` always returns the same items in the same order.
    This is the property that makes re-runs with ``--seed`` byte-identical.

    Parameters
    ----------
    examples
        The full pool of examples (typically the test split).
    n
        Number of examples to return.
    seed
        Integer random seed.

    Raises
    ------
    ValueError
        If ``n`` exceeds the pool size. We fail loudly rather than
        silently returning a truncated list — silent truncation would
        change the evaluation surface without the caller knowing.
    """
    if n < 1:
        raise ValueError(
            f"n={n} must be at least 1. Pass --n 1 or greater."
        )
    if n > len(examples):
        raise ValueError(
            f"n={n} exceeds pool size={len(examples)}. "
            "Use a smaller --n or load the full dataset."
        )

    # Copy then shuffle in-place using a local Random instance so we
    # don't affect the global random state (which might be set for
    # reproducibility elsewhere).
    pool = list(examples)
    rng = random.Random(seed)
    rng.shuffle(pool)
    return pool[:n]


def _shard_sampled_examples_for_parallel_eval(
    examples: list[Any],
    *,
    sample_shard_id: int,
    num_sample_shards: int,
) -> list[Any]:
    """Return one round-robin slice of an already-sampled eval set.

    The caller must sample first, then shard. That preserves the exact same
    logical eval surface as an unsharded run with the same ``--n`` and
    ``--seed`` while allowing independent subprocesses to evaluate disjoint
    example subsets.
    """
    if num_sample_shards < 1:
        raise ValueError("num_sample_shards must be at least 1")
    if sample_shard_id < 0 or sample_shard_id >= num_sample_shards:
        raise ValueError(
            "sample_shard_id must be in [0, num_sample_shards); "
            f"got sample_shard_id={sample_shard_id}, "
            f"num_sample_shards={num_sample_shards}"
        )
    return examples[sample_shard_id::num_sample_shards]


# =============================================================================
# Token-counting helper (MEDIUM 7)
# =============================================================================


def _count_non_pad_tokens(new_token_ids: torch.Tensor, pad_id: int | None) -> int:
    """Count positions in a (batch, new_tokens) tensor whose id is not pad.

    Used to report ``generated_tokens_decoded`` (content length)
    separately from the rectangular position count
    ``new_token_ids.numel()`` (compute). When ``pad_id`` is None there
    is no pad concept, so all positions count as non-pad.

    Returns ``int`` (not a 0-d tensor) so callers can sum across batches
    without per-batch device→host round-trips dominating the time on GPU.

    Factored out so the divergence between rectangular and content
    counts can be unit-tested with a synthetic tensor, instead of
    relying on a model run that happens to hit EOS at the right step.
    """
    if pad_id is None:
        return int(new_token_ids.numel())
    return int((new_token_ids != pad_id).sum().item())


# =============================================================================
# Batched generation with OOM fallback
# =============================================================================


def _generate_batch(
    model: Any,
    tokenizer: Any,
    prompts: list[str],
    batch_size: int,
    max_new_tokens: int,
    device: str,
) -> tuple[list[str], int, int, float]:
    """Generate continuations for ``prompts`` in batches.

    Processes ``prompts`` in chunks of at most ``batch_size``. On
    ``torch.cuda.OutOfMemoryError``, halves the batch size and retries
    the same chunk. If batch_size reaches 1 and still OOMs, raises.

    After an OOM forces a halve, subsequent chunks start from the
    reduced size rather than resetting to the original. This avoids
    repeatedly triggering OOM on a device that already proved the
    original size was too large.

    Parameters
    ----------
    model
        A loaded ``AutoModelForCausalLM`` in eval mode.
    tokenizer
        The matching tokenizer. Must have ``padding_side = "left"`` and
        a ``pad_token`` set.
    prompts
        List of plain text prompts.
    batch_size
        Starting batch size. Will be halved on OOM and the new lower
        value is carried forward to subsequent chunks.
    max_new_tokens
        Token budget for the continuation (set per source in the registry).
    device
        ``"cuda"`` or ``"cpu"``.

    Returns
    -------
    A tuple of (generated_texts, total_new_tokens, non_pad_tokens,
    generation_seconds). ``total_new_tokens`` is the rectangular
    position count (compute); ``non_pad_tokens`` is the count of
    positions whose id is not ``pad_token_id`` (content length). See
    ``_tokenize_and_generate`` for the full distinction.
    """
    all_generated: list[str] = []
    total_tokens: int = 0
    total_non_pad: int = 0
    total_gen_sec: float = 0.0

    # Walk through all prompts in chunks of ``current_batch_size``.
    # The batch size may shrink during iteration if an OOM is encountered;
    # we carry the reduced size forward so we never re-trigger OOM on the
    # same device with the same size.
    i = 0
    current_batch_size = batch_size
    while i < len(prompts):
        chunk = prompts[i : i + current_batch_size]
        (
            generated,
            chunk_tokens,
            chunk_non_pad,
            current_batch_size,
            chunk_gen_sec,
        ) = _generate_chunk_with_oom_fallback(
            model=model,
            tokenizer=tokenizer,
            prompts=chunk,
            batch_size=current_batch_size,
            max_new_tokens=max_new_tokens,
            device=device,
        )
        all_generated.extend(generated)
        total_tokens += chunk_tokens
        total_non_pad += chunk_non_pad
        total_gen_sec += chunk_gen_sec
        i += len(chunk)

    return all_generated, total_tokens, total_non_pad, total_gen_sec


def _generate_chunk_with_oom_fallback(
    model: Any,
    tokenizer: Any,
    prompts: list[str],
    batch_size: int,
    max_new_tokens: int,
    device: str,
) -> tuple[list[str], int, int, int, float]:
    """Generate a single chunk, halving batch size on OOM.

    Recursively halves ``batch_size`` and splits the chunk in two until
    it succeeds or ``batch_size`` would drop below 1.

    Parameters
    ----------
    model, tokenizer, max_new_tokens, device
        As in ``_generate_batch``.
    prompts
        The chunk to generate for (length <= batch_size).
    batch_size
        The batch size to try for this chunk.

    Returns
    -------
    A tuple of (generated_texts, total_new_tokens, non_pad_tokens,
    effective_batch_size, generation_seconds).
    ``total_new_tokens`` is the rectangular position count;
    ``non_pad_tokens`` is the content-length count (positions whose id
    is not pad_token_id) — see ``_tokenize_and_generate`` for the full
    distinction. ``effective_batch_size`` is the smallest batch size
    that actually succeeded; the outer loop in ``_generate_batch``
    carries this forward so subsequent chunks start at the reduced size.
    ``generation_seconds`` is the total time spent inside
    ``model.generate()`` for this chunk (summed across recursive halves).

    Raises
    ------
    torch.cuda.OutOfMemoryError
        If batch_size is already 1 and generation still OOMs.
    """
    # Guard: an empty prompt list can arrive as the right half of a
    # 1-prompt chunk split (prompts[:1] = [prompt], prompts[1:] = []).
    # Without this check _tokenize_and_generate would crash, and the zip
    # in _evaluate_one_source would produce a length-mismatch error.
    if not prompts:
        return [], 0, 0, batch_size, 0.0

    try:
        texts, token_count, non_pad_count, gen_sec = _tokenize_and_generate(
            model=model,
            tokenizer=tokenizer,
            prompts=prompts,
            max_new_tokens=max_new_tokens,
            device=device,
        )
        return texts, token_count, non_pad_count, batch_size, gen_sec
    except torch.cuda.OutOfMemoryError:
        if batch_size <= 1:
            # Can't go smaller. Fail loudly — the caller asked for this
            # to be a hard failure so the user knows the hardware is
            # genuinely insufficient.
            raise

        # Halve the batch size and split the chunk in two.
        new_batch_size = batch_size // 2
        print(
            f"[eval_exact] OOM at batch_size={batch_size}; "
            f"retrying at batch_size={new_batch_size}",
            file=sys.stderr,
        )
        # Increment the run-level counter so we can warn the operator at the
        # end of the run to lower their starting batch size.
        global _oom_halving_count
        _oom_halving_count += 1

        # Free cached GPU memory before retrying.
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Split the chunk into two halves and recurse on each.
        mid = len(prompts) // 2 if len(prompts) > 1 else 1
        left = _generate_chunk_with_oom_fallback(
            model=model,
            tokenizer=tokenizer,
            prompts=prompts[:mid],
            batch_size=new_batch_size,
            max_new_tokens=max_new_tokens,
            device=device,
        )
        right = _generate_chunk_with_oom_fallback(
            model=model,
            tokenizer=tokenizer,
            prompts=prompts[mid:],
            batch_size=new_batch_size,
            max_new_tokens=max_new_tokens,
            device=device,
        )
        left_texts, left_tokens, left_non_pad, left_bs, left_gen_sec = left
        right_texts, right_tokens, right_non_pad, right_bs, right_gen_sec = right
        # Return the minimum effective batch size so the outer loop can
        # carry it forward to prevent re-triggering OOM on later chunks.
        effective_bs = min(left_bs, right_bs)
        total_gen_sec = left_gen_sec + right_gen_sec
        return (
            left_texts + right_texts,
            left_tokens + right_tokens,
            left_non_pad + right_non_pad,
            effective_bs,
            total_gen_sec,
        )


def _tokenize_and_generate(
    model: Any,
    tokenizer: Any,
    prompts: list[str],
    max_new_tokens: int,
    device: str,
) -> tuple[list[str], int, int, float]:
    """Tokenize a batch of prompts, generate, decode only the new tokens.

    This is the inner loop that does the actual HuggingFace model call.
    Kept separate from the OOM-fallback logic so each function has one job.

    Left-padding is required for batched decoder-only generation. With
    left-padding all prompts end at the same position, so the model's
    first generated token for every sequence in the batch is conditioned
    on the correct last prompt token. Right-padding would push the
    last prompt token to different positions for different sequences,
    breaking the causal dependency.

    Parameters
    ----------
    model, tokenizer, max_new_tokens, device
        As in ``_generate_batch``.
    prompts
        The exact list of prompts to generate for. Length must be > 0.

    Returns
    -------
    A tuple of (generated_texts, total_new_tokens, non_pad_tokens,
    generation_seconds). ``generated_texts`` is a list of decoded
    new-token strings, one per prompt. ``total_new_tokens`` is the
    rectangular position count (batch_size * new_token_cols) including
    trailing pad fill for sequences that hit EOS early — this reflects
    the compute spent because model.generate runs all decode steps
    until every sequence in the batch finishes. ``non_pad_tokens`` is
    the count of positions whose token id is not ``pad_token_id``; it
    approximates the meaningful output length and is reported alongside
    ``total_new_tokens`` so cost-per-token consumers can distinguish
    rectangular compute from emitted content. ``generation_seconds`` is
    the wall-clock time spent inside ``model.generate()`` only —
    excludes tokenization and decode overhead.
    """
    # Tokenize with left-padding to the longest prompt in this batch.
    encoding = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,       # pads shorter sequences
        truncation=False,   # do not silently truncate; let OOM surface instead
        add_special_tokens=True,
    )
    input_ids = encoding["input_ids"].to(device)
    attention_mask = encoding["attention_mask"].to(device)

    prompt_len = input_ids.shape[1]  # number of tokens in the (padded) prompt

    # Time only the generate() call so we can report true generation
    # throughput (excluding model load, tokenization, and decode overhead).
    _gen_start = time.monotonic()
    with torch.no_grad():
        output_ids = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=False,    # greedy decoding: deterministic, reproducible
            use_cache=True,     # KV-cache: speeds up autoregressive steps
        )
    generation_seconds = time.monotonic() - _gen_start

    # output_ids shape: (batch, prompt_len + new_tokens).
    # Slice off the prompt tokens to get only the newly generated tokens.
    new_token_ids = output_ids[:, prompt_len:]

    # Count total new token positions directly from the tensor shape.
    # new_token_ids.numel() == batch_size * new_token_cols. Each position
    # corresponds to a real forward pass (model.generate runs until all
    # sequences in the batch finish, not until any single one does).
    total_new_tokens = new_token_ids.numel()

    # Non-pad count: approximates the meaningful output length per the
    # MEDIUM 7 fix. Sequences that hit EOS early have their trailing
    # positions filled with pad_token_id by HuggingFace's generate(),
    # which the rectangular count above includes but a content-length
    # measure should not. Delegated to ``_count_non_pad_tokens`` so the
    # counting logic can be unit-tested directly with a synthetic
    # tensor that contains pad fill, without depending on a real model
    # happening to hit EOS within the budget.
    non_pad_tokens = _count_non_pad_tokens(new_token_ids, tokenizer.pad_token_id)

    # Decode each row separately. skip_special_tokens=True strips pad/eos.
    generated_texts = [
        tokenizer.decode(row, skip_special_tokens=True)
        for row in new_token_ids
    ]
    return generated_texts, total_new_tokens, non_pad_tokens, generation_seconds


# =============================================================================
# Output artifact writers
# =============================================================================


def _write_accuracy_summary(
    rows: list[dict[str, Any]],
    out_dir: Path,
    extra_columns: tuple[str, ...] = (),
) -> None:
    """Write accuracy_summary.json and accuracy_summary.csv.

    Each row in ``rows`` must have at least the keys:
      checkpoint, source, n, accuracy, parse_success_rate,
      generated_tokens, generated_tokens_decoded, elapsed_sec.

    When ``extra_columns`` is non-empty, those column names are appended
    to the field list (and expected to exist on every row). This is how
    the ChainEval aggregate columns get folded into the summary without
    changing the existing call sites.

    Parameters
    ----------
    rows
        List of summary dicts, one per (checkpoint, source) pair.
    out_dir
        Directory where the two files are written.
    extra_columns
        Optional tuple of extra column names to include after the
        baseline fields. Used by the ChainEval path.
    """
    # Field order for CSV. Defined once here so JSON and CSV are consistent.
    fieldnames = [
        "checkpoint",
        "source",
        "n",
        "accuracy",
        "parse_success_rate",
        "generated_tokens",
        "generated_tokens_decoded",
        "elapsed_sec",
        *extra_columns,
    ]

    # JSON — human-readable, machine-parseable.
    json_path = out_dir / "accuracy_summary.json"
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2)

    # CSV — use newline="" so the csv module owns line endings (Windows safe).
    csv_path = out_dir / "accuracy_summary.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_details_csv(
    rows: list[dict[str, Any]],
    checkpoint_name: str,
    source_name: str,
    out_dir: Path,
    extra_columns: tuple[str, ...] = (),
) -> None:
    """Write details_<checkpoint>_<source>.csv.

    Each row in ``rows`` must have at least the keys:
      example_id, prompt, generated, gold_answer, predicted_answer,
      parse_success, is_correct.

    The ``generated`` column may contain newlines and commas; the csv
    module's default quoting (QUOTE_MINIMAL) will wrap such cells in
    double quotes automatically, keeping the row count correct on re-read.

    When ``extra_columns`` is non-empty those column names are appended
    to the header and expected to be present on every row dict. This is
    the hook the ChainEval plumbing uses to add the step/ROUGE/BERT/DTW
    columns without breaking the existing schema.

    Parameters
    ----------
    rows
        Per-example result dicts.
    checkpoint_name
        User-supplied name (e.g. ``"base"``), used in the filename.
    source_name
        Source identifier, used in the filename.
    out_dir
        Output directory.
    extra_columns
        Optional tuple of extra column names to append to the schema.
    """
    fieldnames = [
        "example_id",
        "prompt",
        "generated",
        "gold_answer",
        "predicted_answer",
        "parse_success",
        "is_correct",
        *extra_columns,
    ]

    csv_path = out_dir / f"details_{checkpoint_name}_{source_name}.csv"
    # newline="" is required: Python's universal-newline translation would
    # add an extra \r before the \n the csv module writes, creating \r\r\n
    # on Windows. Passing newline="" delegates line-ending to the csv module.
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_cost_summary(
    out_dir: Path,
    run_name: str,
    start_time: str,
    end_time: str,
    elapsed_sec: float,
    generation_seconds: float,
    gpu_type: str,
    dtype: str,
    generated_tokens: int,
    generated_tokens_decoded: int,
    tokens_per_second: float,
    estimated_cost_usd: float | None,
) -> None:
    """Write cost_summary.json.

    Parameters
    ----------
    out_dir
        Output directory.
    run_name
        Human-readable identifier for the run (from ``--out-dir``).
    start_time, end_time
        ISO-8601 strings set by RunTracker.
    elapsed_sec
        Wall-clock seconds for the full evaluation (includes model load,
        data loading, and writing output files). Used for billing estimates.
    generation_seconds
        Seconds spent inside ``model.generate()`` across all calls and all
        sources. Excludes model load time so this reflects true generation
        throughput.
    gpu_type
        Device name string (e.g. ``"Tesla T4"`` or ``"CPU"``).
    dtype
        Torch dtype string (e.g. ``"torch.bfloat16"``).
    generated_tokens
        Total new token positions across all sources — rectangular count
        including trailing pad fill for sequences that hit EOS early. This
        is the compute cost (real forward passes per position).
    generated_tokens_decoded
        Total non-pad token count across all sources — approximates the
        meaningful output length the models actually emitted. Reported
        alongside ``generated_tokens`` because the two diverge when
        sequences in a batch finish at different steps.
    tokens_per_second
        Throughput computed as ``generated_tokens / generation_seconds``
        (not divided by ``elapsed_sec`` to avoid penalising model load time).
        Uses the rectangular count because that is the actual compute.
    estimated_cost_usd
        Dollar cost if ``--gpu-cost-per-hour`` was supplied, else ``None``.
    """
    data = {
        "run_name": run_name,
        "start_time": start_time,
        "end_time": end_time,
        "elapsed_sec": elapsed_sec,
        "generation_seconds": generation_seconds,
        "gpu_type": gpu_type,
        "dtype": dtype,
        "generated_tokens": generated_tokens,
        "generated_tokens_decoded": generated_tokens_decoded,
        "tokens_per_second": tokens_per_second,
        "estimated_cost_usd": estimated_cost_usd,
    }
    with (out_dir / "cost_summary.json").open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def _write_run_metadata(
    out_dir: Path,
    device: str,
    dtype: str,
    seed: int,
    eval_n: int,
    generation_settings: dict[str, Any],
    checkpoints: dict[str, str],
    source_splits: dict[str, str] | None = None,
) -> None:
    """Write run_metadata.json.

    Contains everything needed to reproduce the run: library versions,
    hardware, eval configuration, checkpoint paths, and the git SHA so
    you can trace back to the exact code state.

    Parameters
    ----------
    out_dir
        Output directory.
    device
        Device string (``"cuda"`` or ``"cpu"``).
    dtype
        Torch dtype string used for all model weights.
    seed
        The ``--seed`` value.
    eval_n
        The ``--n`` value.
    generation_settings
        Dict mapping source name to generation kwargs (max_new_tokens,
        batch_size).
    checkpoints
        Dict mapping user-supplied checkpoint names to their paths or
        Hub IDs (e.g. ``{"base": "Qwen/Qwen2.5-0.5B"}``).
    source_splits
        Optional explicit dataset split by source. FinChain uses this to
        distinguish validation checkpoint selection from held-out test reporting.
    """
    import transformers

    # CUDA version: only meaningful when CUDA is available.
    cuda_version = torch.version.cuda if torch.cuda.is_available() else "N/A"

    # Friendly device name: match the convention used in cost_summary.json
    # (RunTracker.gpu_type). On CUDA, report the human-readable GPU name
    # (e.g. "NVIDIA A40"). On CPU, report "CPU" (capitalised, matching
    # RunTracker's default).
    if device.startswith("cuda") and torch.cuda.is_available():
        device_name = torch.cuda.get_device_name(0)
    else:
        device_name = "CPU"

    # Git short SHA: pass cwd= so the git call succeeds regardless of
    # where the user invokes the CLI from (e.g. cd ~ && python -m ...).
    # Path(__file__).resolve().parent gives the directory containing this
    # file, which is always inside the repo; git walks upward to find .git.
    try:
        git_sha = (
            subprocess.check_output(
                ["git", "rev-parse", "--short", "HEAD"],
                stderr=subprocess.DEVNULL,
                cwd=Path(__file__).resolve().parent,
            )
            .decode()
            .strip()
        )
    except Exception:
        git_sha = "unknown"

    data = {
        "timestamp_utc": datetime.datetime.now(datetime.UTC).isoformat(),
        "command": sys.argv,
        "command_str": " ".join(sys.argv),
        "device": device_name,
        "dtype": dtype,
        "torch_version": torch.__version__,
        "transformers_version": transformers.__version__,
        "cuda_version": cuda_version,
        "seed": seed,
        "eval_n": eval_n,
        "git_sha": git_sha,
        "generation_settings": generation_settings,
        "checkpoints": checkpoints,
        "checkpoint_paths": checkpoints,
        "source_splits": source_splits or {},
    }
    with (out_dir / "run_metadata.json").open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


# =============================================================================
# dtype selection helper
# =============================================================================


def _select_dtype(device: str) -> torch.dtype:
    """Choose the appropriate model dtype for the device.

    Rules from the PRD:
    - CUDA with compute capability >= (8, 0): ``bfloat16``
      (Ampere and newer; T4 is Turing = 7.5, A100 is Ampere = 8.0).
    - CUDA below Ampere: ``float16``
      (T4, V100; bfloat16 is supported on Ampere+ only).
    - CPU: ``float32``
      (bfloat16 on CPU is very slow; float32 is the right default).

    Parameters
    ----------
    device
        ``"cuda"`` or ``"cpu"``.

    Returns
    -------
    A ``torch.dtype`` instance.
    """
    if device == "cpu" or not torch.cuda.is_available():
        return torch.float32

    major, _ = torch.cuda.get_device_capability(0)
    if major >= 8:
        return torch.bfloat16
    return torch.float16


# =============================================================================
# Model loading and unloading
# =============================================================================


def _load_model_and_tokenizer(
    model_path: str,
    dtype: torch.dtype,
    device: str,
) -> tuple[Any, Any]:
    """Load a model and tokenizer from a local path or Hugging Face Hub ID.

    Delegates to ``finpost.safety.safe_load_model`` and
    ``finpost.safety.safe_load_tokenizer``, which enforce defensive
    defaults (trust_remote_code=False, use_safetensors=True).

    The only deliberate exception is ``sshleifer/tiny-gpt2``, a test
    model that ships only legacy .bin (pickle) weights with no safetensors
    variant. That model is used exclusively in unit tests; production Phase
    1 models (Qwen, Gemma) ship safetensors and are unaffected.

    Tokenizer is set to left-padding (required for batched decoder-only
    generation; see ``_tokenize_and_generate`` for the full rationale).

    Parameters
    ----------
    model_path
        Local directory or Hugging Face Hub model identifier.
    dtype
        The compute dtype selected by ``_select_dtype``.
    device
        ``"cuda"`` or ``"cpu"``.

    Returns
    -------
    (model, tokenizer) both moved / configured for ``device``.
    """
    tokenizer = safe_load_tokenizer(model_path)

    # Left-padding: decoder-only models must have the last prompt token
    # at the right edge of the input so the first generated token
    # attends to the correct context.
    tokenizer.padding_side = "left"

    # Many GPT-style tokenizers have no pad_token. Reuse eos_token as pad.
    # The attention_mask ensures the model ignores the padding positions.
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # tiny-gpt2 (sshleifer/tiny-gpt2) ships only .bin (pickle) weights and
    # has no safetensors variant. It is used exclusively in unit tests.
    # For all production models (Qwen, Gemma), use_safetensors=True (the
    # safe_load_model default) refuses pickle weights per SECURITY.md policy.
    if model_path == "sshleifer/tiny-gpt2":
        # Override: use_safetensors=False because this test model ships no
        # safetensors variant. Justification: test-only, not used in production.
        model = safe_load_model(
            model_path,
            use_safetensors=False,
            dtype=dtype,
        ).to(device)
    else:
        model = safe_load_model(
            model_path,
            dtype=dtype,
        ).to(device)

    model.eval()
    return model, tokenizer


def _free_cuda_cache() -> None:
    """Release cached (but currently unused) GPU memory back to the allocator.

    Call this after ``del model, tokenizer`` in the caller's scope.
    Note: ``del`` must happen at the call site, not inside a wrapper function.
    Python's reference semantics mean that passing an object to a function and
    deleting the local parameter only removes the function's binding — the
    caller's variable still holds a live reference and keeps the weights in
    memory. The ``del`` that actually frees memory must rebind the variable
    in the same scope that owns it.
    """
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# =============================================================================
# Core evaluation loop — one checkpoint, one source
# =============================================================================


def _evaluate_one_source(
    model: Any,
    tokenizer: Any,
    source: EvalSource,
    examples: list[Any],
    batch_size: int,
    device: str,
) -> tuple[list[dict[str, Any]], int, int, float]:
    """Run generation + scoring for one (model, source) pair.

    Parameters
    ----------
    model, tokenizer
        Loaded model and tokenizer in eval mode.
    source
        The ``EvalSource`` entry from the registry.
    examples
        The subsampled list of examples to evaluate.
    batch_size
        Starting batch size for generation (may be halved on OOM).
    device
        ``"cuda"`` or ``"cpu"``.

    Returns
    -------
    (detail_rows, total_new_tokens, non_pad_tokens, generation_seconds)
        ``detail_rows`` is a list of per-example dicts ready to write to CSV.
        ``total_new_tokens`` is the rectangular position count (compute).
        ``non_pad_tokens`` is the count of positions whose id is not
        ``pad_token_id`` (content length). ``generation_seconds`` is
        the total time spent inside ``model.generate()`` for this source.
    """
    prompts = [ex.prompt for ex in examples]

    # Generate all continuations for this source in one batched call.
    # _generate_batch handles the chunking, OOM fallback, and token counting.
    (
        generated_texts,
        total_new_tokens,
        non_pad_tokens,
        generation_seconds,
    ) = _generate_batch(
        model=model,
        tokenizer=tokenizer,
        prompts=prompts,
        batch_size=batch_size,
        max_new_tokens=source.default_max_new_tokens,
        device=device,
    )

    # Score each example and build the per-row detail dict.
    detail_rows: list[dict[str, Any]] = []
    for example, generated in zip(examples, generated_texts, strict=True):
        predicted = source.extract_answer(generated)
        parse_success = predicted is not None
        is_correct = source.score(predicted, example.final_answer)

        detail_rows.append({
            "example_id": example.id,
            "prompt": example.prompt,
            "generated": generated,
            "gold_answer": example.final_answer,
            "predicted_answer": predicted if predicted is not None else "",
            "parse_success": parse_success,
            "is_correct": is_correct,
        })

    return detail_rows, total_new_tokens, non_pad_tokens, generation_seconds


# =============================================================================
# Top-level run_eval function (callable from tests and the CLI)
# =============================================================================


def _attach_chaineval_metrics(
    detail_rows: list[dict[str, Any]],
    examples: list[Any],
    source_name: str,
) -> None:
    """Run post-hoc ChainEval scoring on each row in place.

    Iterates the per-example result dicts and calls ``score_trace`` once
    per row with the gold solution (from ``example.response``) and the
    model generation. The resulting metric dict is merged into the row,
    plus a ``step_f1`` derived from step_recall/step_precision.

    If a row's example does not carry a usable ``response`` (the
    FinChain loader stores the full trace there; anything else suggests
    the wrong loader is wired up), this raises with a clear message
    pointing the operator at the FinChain loader rather than silently
    scoring against an empty string.

    Parameters
    ----------
    detail_rows
        Per-example result dicts produced by ``_evaluate_one_source``.
        Mutated in place: ChainEval columns are added to each entry.
    examples
        The same ``Example`` instances that produced the detail rows,
        in the same order. Provides ``response`` (the gold trace).
    source_name
        Used only in error messages so the operator knows which source
        is misconfigured.
    """
    if len(detail_rows) != len(examples):
        # Defensive: the loop below depends on positional alignment.
        raise RuntimeError(
            f"ChainEval scoring: detail_rows ({len(detail_rows)}) and "
            f"examples ({len(examples)}) lengths must match."
        )

    for row, example in zip(detail_rows, examples, strict=True):
        gold_trace = getattr(example, "response", None)
        if not isinstance(gold_trace, str) or not gold_trace.strip():
            raise RuntimeError(
                f"ChainEval needs the gold reasoning trace on Example.response "
                f"for source {source_name!r}, but example_id={example.id!r} has "
                f"a missing or empty response field. Verify that "
                f"`finpost.data.finchain_dataset.load_finchain` populates `response` "
                f"with the full solution trace (not just the final answer)."
            )
        metrics = score_trace(gold_trace, row["generated"])
        step_f1 = compute_step_f1(metrics["step_recall"], metrics["step_precision"])
        row.update(metrics)
        row["step_f1"] = step_f1


def run_eval(
    checkpoints: dict[str, str],
    sources: list[str],
    n: int,
    seed: int,
    out_dir: Path,
    batch_sizes: dict[str, int],
    gpu_cost_per_hour: float | None,
    device: str,
    enable_chaineval: bool = False,
    chaineval_batch_size: int = 8,
    sample_shard_id: int | None = None,
    num_sample_shards: int | None = None,
    finchain_split: str = "test",
) -> None:
    """Run exact-answer evaluation for all (checkpoint, source) pairs.

    This is the core logic, separated from argument parsing so it can
    be called directly from tests and from the Colab/Kaggle notebooks
    (issue 03) without going through ``sys.argv``.

    Parameters
    ----------
    checkpoints
        Dict mapping user-supplied names to model paths or Hub IDs.
        Example: ``{"base": "Qwen/Qwen2.5-0.5B", "combined": "/tmp/ckpt"}``.
    sources
        List of source names to look up in ``REGISTRY``.
    n
        Number of examples per source to evaluate (seeded random sample).
    seed
        Controls both the example sample and generation determinism.
    out_dir
        Directory where all five output files are written. Created if absent.
    batch_sizes
        Dict mapping source name to starting batch size for generation.
        Defaults to 8 if a source is not listed.
    gpu_cost_per_hour
        Optional dollar-per-GPU-hour rate for cost estimate. ``None`` means
        no estimate is written.
    device
        ``"cuda"`` or ``"cpu"``.
    enable_chaineval
        If True, runs the post-hoc ChainEval scorer on every (gold, pred)
        pair after generation. Adds step-alignment, ROUGE, BERTScore, and
        DTW columns to ``details_*.csv`` and per-checkpoint means to
        ``accuracy_summary.{json,csv}``. Requires the ``[chaineval]``
        extra to be installed; raises ``ChainEvalNotAvailable`` if not.
    chaineval_batch_size
        Reserved for future batching of BERTScore + sentence-transformer
        embedding calls. Accepted today so the CLI surface is stable.
    sample_shard_id, num_sample_shards
        Optional example-level sharding controls. When supplied, each source
        is first sampled with ``n`` and ``seed`` exactly as in an unsharded
        run, then sliced round-robin so multiple subprocesses can cover the
        same logical eval set across GPUs.
    finchain_split
        ``"validation"`` for checkpoint selection or ``"test"`` for final
        reporting. This option applies only when ``finchain`` is requested.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Fail fast on a misconfigured ChainEval run. Doing this here, before
    # any model or data load, means the operator sees the install hint in
    # seconds instead of after a long generation pass that gets thrown away.
    if enable_chaineval and not CHAINEVAL_AVAILABLE:
        raise ChainEvalNotAvailable()

    # chaineval_batch_size is a forward-looking knob; reference it so linters
    # do not flag it as unused. (Today the upstream BERTScorer is called one
    # row at a time; the batch size knob exists for a future batching pass.)
    _ = chaineval_batch_size

    if (sample_shard_id is None) != (num_sample_shards is None):
        raise ValueError(
            "sample_shard_id and num_sample_shards must be supplied together"
        )
    if finchain_split not in ("validation", "test"):
        raise ValueError("finchain_split must be 'validation' or 'test'")

    # RG1: warn if --out-dir is non-empty so the operator notices before
    # existing artifacts are silently overwritten by this run.
    existing = list(out_dir.iterdir())
    if existing:
        print(
            f"[eval_exact] WARNING: --out-dir {out_dir} is non-empty "
            f"({len(existing)} files). Existing files will be overwritten.",
            file=sys.stderr,
        )

    global _oom_halving_count
    _oom_halving_count = 0

    # Set CUDA determinism flags before any model loading or CUDA tensor work.
    # This is a no-op on CPU. See _set_cuda_determinism for the rationale.
    _set_cuda_determinism(device)

    # Design note on GPU memory management:
    # After evaluating a checkpoint we ``del model, tokenizer`` here in
    # run_eval's scope. That is the ``del`` that actually frees memory.
    # Passing the model to a helper function and doing ``del`` inside the
    # helper only removes the helper's local binding; the variable here still
    # holds a live reference and keeps the weights allocated. The explicit
    # ``del`` below, followed by _free_cuda_cache(), is what ensures peak
    # GPU footprint is one model at a time — essential on free-tier T4 GPUs.

    # Resolve sources from the registry before touching any model weights.
    # Fail early if an unknown source name was requested.
    eval_sources: list[EvalSource] = []
    for source_name in sources:
        if source_name not in REGISTRY:
            raise ValueError(
                f"Unknown source {source_name!r}. "
                f"Available: {sorted(REGISTRY.keys())}"
            )
        eval_sources.append(REGISTRY[source_name])

    # Select dtype once (device-dependent, consistent across all checkpoints).
    dtype = _select_dtype(device)
    dtype_str = str(dtype)  # e.g. "torch.float32" — goes into metadata

    # Determine the starting batch size for each source.
    # Default to 8 if not explicitly specified.
    resolved_batch_sizes: dict[str, int] = {
        src.name: batch_sizes.get(src.name, 8)
        for src in eval_sources
    }

    # Pre-load all examples from each source (one download per source,
    # before any model loading). Subsample deterministically with the seed.
    print("[eval_exact] Loading and sampling examples from all sources...")
    sampled_examples: dict[str, list[Any]] = {}
    for source in eval_sources:
        all_examples = (
            load_finchain(finchain_split)
            if source.name == "finchain"
            else source.load_examples()
        )
        sampled = _sample_examples(all_examples, n=n, seed=seed)
        if sample_shard_id is not None and num_sample_shards is not None:
            sampled = _shard_sampled_examples_for_parallel_eval(
                sampled,
                sample_shard_id=sample_shard_id,
                num_sample_shards=num_sample_shards,
            )
        sampled_examples[source.name] = sampled
        shard_note = (
            ""
            if sample_shard_id is None
            else f"; shard {sample_shard_id}/{num_sample_shards}"
        )
        print(
            f"  {source.name}: {len(sampled)} examples "
            f"(from {len(all_examples)} total{shard_note})"
        )

    # RG2: Print a pre-run workload and cost estimate so the operator can
    # Ctrl-C before any compute is paid for if the numbers look wrong.
    # total_examples counts each (checkpoint, source) pair independently
    # because each checkpoint must process each source separately.
    total_examples = (
        sum(len(samples) for samples in sampled_examples.values())
        * len(checkpoints)
    )
    worst_case_tokens = total_examples * max(
        s.default_max_new_tokens for s in eval_sources
    )
    print(
        f"[eval_exact] Workload estimate: {total_examples} generations, "
        f"up to {worst_case_tokens:,} new tokens worst case."
    )
    if gpu_cost_per_hour is not None:
        # Conservative throughput: ~100 tok/s on T4 bf16, faster on A40/H100.
        # This is an upper-bound time estimate (worst case generation length).
        rough_hours = worst_case_tokens / 100 / 3600
        rough_cost = rough_hours * gpu_cost_per_hour
        print(
            f"[eval_exact]   At ~100 tok/s, that is ~{rough_hours:.2f}h "
            f"= ~${rough_cost:.2f} at ${gpu_cost_per_hour:.2f}/hr."
        )

    # Build generation_settings for run_metadata.json.
    generation_settings: dict[str, Any] = {
        src.name: {
            "max_new_tokens": src.default_max_new_tokens,
            "batch_size": resolved_batch_sizes[src.name],
            "do_sample": False,
        }
        for src in eval_sources
    }

    # Accumulate summary rows across all (checkpoint, source) pairs.
    summary_rows: list[dict[str, Any]] = []

    # Outer loop: one checkpoint at a time. Load → evaluate all sources → free.
    run_name = out_dir.name  # use the final directory component as run name
    tracker = RunTracker(out_dir=out_dir, run_name=run_name, gpu_cost_per_hour=gpu_cost_per_hour)

    # Track total time spent inside model.generate() across all sources and
    # checkpoints so we can report true generation throughput separately from
    # wall-clock (which includes model load, data loading, and I/O).
    total_generation_seconds: float = 0.0

    with tracker:
        for ckpt_name, ckpt_path in checkpoints.items():
            print(f"\n[eval_exact] Loading checkpoint: {ckpt_name!r} from {ckpt_path!r}")
            model, tokenizer = _load_model_and_tokenizer(ckpt_path, dtype=dtype, device=device)

            for source in eval_sources:
                examples = sampled_examples[source.name]
                batch_size = resolved_batch_sizes[source.name]

                print(
                    f"  Evaluating {ckpt_name!r} on {source.name!r}: "
                    f"n={len(examples)}, batch_size={batch_size}, "
                    f"max_new_tokens={source.default_max_new_tokens}"
                )

                t0 = time.monotonic()
                (
                    detail_rows,
                    new_tokens,
                    new_tokens_decoded,
                    source_gen_sec,
                ) = _evaluate_one_source(
                    model=model,
                    tokenizer=tokenizer,
                    source=source,
                    examples=examples,
                    batch_size=batch_size,
                    device=device,
                )
                elapsed = time.monotonic() - t0

                tracker.add_generated_tokens(new_tokens)
                tracker.add_decoded_tokens(new_tokens_decoded)
                total_generation_seconds += source_gen_sec

                # Post-hoc ChainEval scoring (optional). Runs after generation
                # so it does not affect generation determinism or timing.
                # _attach_chaineval_metrics mutates detail_rows in place,
                # adding the step / ROUGE / BERT / DTW columns.
                if enable_chaineval:
                    print(f"    [chaineval] scoring {len(detail_rows)} rows...")
                    _attach_chaineval_metrics(
                        detail_rows=detail_rows,
                        examples=examples,
                        source_name=source.name,
                    )

                # Compute aggregate metrics.
                n_evaluated = len(detail_rows)
                n_correct = sum(1 for row in detail_rows if row["is_correct"])
                n_parsed = sum(1 for row in detail_rows if row["parse_success"])
                accuracy = n_correct / n_evaluated if n_evaluated > 0 else 0.0
                parse_success_rate = n_parsed / n_evaluated if n_evaluated > 0 else 0.0

                print(
                    f"    accuracy={accuracy:.3f}  parse_success_rate={parse_success_rate:.3f}"
                    f"  elapsed={elapsed:.1f}s  tokens={new_tokens}"
                    f"  decoded={new_tokens_decoded}"
                )

                # Write per-example details CSV.
                _write_details_csv(
                    rows=detail_rows,
                    checkpoint_name=ckpt_name,
                    source_name=source.name,
                    out_dir=out_dir,
                    extra_columns=CHAINEVAL_DETAIL_COLUMNS if enable_chaineval else (),
                )

                # Accumulate for the summary files. ``generated_tokens`` is the
                # rectangular position count (real forward passes / compute);
                # ``generated_tokens_decoded`` is the non-pad content-length
                # count. Both are reported so cost-per-token consumers can pick
                # the appropriate denominator (positions for compute cost,
                # decoded for content rate).
                summary_row: dict[str, Any] = {
                    "checkpoint": ckpt_name,
                    "source": source.name,
                    "n": n_evaluated,
                    "accuracy": accuracy,
                    "parse_success_rate": parse_success_rate,
                    "generated_tokens": new_tokens,
                    "generated_tokens_decoded": new_tokens_decoded,
                    "elapsed_sec": round(elapsed, 3),
                }
                # Append per-checkpoint ChainEval means alongside accuracy.
                # Each entry is the unweighted mean across the n_evaluated rows.
                if enable_chaineval and n_evaluated > 0:
                    for col in CHAINEVAL_DETAIL_COLUMNS:
                        summary_row[col] = (
                            sum(float(row[col]) for row in detail_rows) / n_evaluated
                        )
                elif enable_chaineval:
                    # No examples scored (defensive — _sample_examples already
                    # rejects n<1). Emit 0.0 so the schema stays stable.
                    for col in CHAINEVAL_DETAIL_COLUMNS:
                        summary_row[col] = 0.0
                summary_rows.append(summary_row)

            # Done with this checkpoint. Unbind model and tokenizer here
            # (in run_eval's scope) so the reference count drops to zero and
            # the weights are freed. See the design note near the top of this
            # function for why del must happen here rather than in a helper.
            print(f"  Unloading {ckpt_name!r}")
            del model, tokenizer
            _free_cuda_cache()

    # Write the three summary files (happens outside the tracker context manager,
    # after tracker.elapsed_sec has been set by __exit__).
    _write_accuracy_summary(
        rows=summary_rows,
        out_dir=out_dir,
        extra_columns=CHAINEVAL_DETAIL_COLUMNS if enable_chaineval else (),
    )

    # Write cost_summary.json directly rather than via tracker.write(), so
    # we can supply the correct dtype string (only known after model load).
    # tokens_per_second is computed from generation time (not wall-clock)
    # so it reflects true generation throughput, not model load overhead.
    tokens_per_second = (
        tracker.total_generated_tokens / total_generation_seconds
        if total_generation_seconds > 0
        else 0.0
    )
    estimated_cost_usd: float | None = None
    if gpu_cost_per_hour is not None:
        # Dollar cost uses elapsed_sec (wall-clock) for billing accuracy.
        estimated_cost_usd = (tracker.elapsed_sec / 3600.0) * gpu_cost_per_hour

    _write_cost_summary(
        out_dir=out_dir,
        run_name=run_name,
        start_time=tracker.start_time,
        end_time=tracker.end_time,
        elapsed_sec=tracker.elapsed_sec,
        generation_seconds=total_generation_seconds,
        gpu_type=tracker.gpu_type,
        dtype=dtype_str,
        generated_tokens=tracker.total_generated_tokens,
        generated_tokens_decoded=tracker.total_decoded_tokens,
        tokens_per_second=tokens_per_second,
        estimated_cost_usd=estimated_cost_usd,
    )

    _write_run_metadata(
        out_dir=out_dir,
        device=device,
        dtype=dtype_str,
        seed=seed,
        eval_n=n,
        generation_settings=generation_settings,
        checkpoints=checkpoints,
        source_splits={
            source.name: finchain_split if source.name == "finchain" else "test"
            for source in eval_sources
        },
    )

    # RG3: Summarise OOM-fallback halvings so the operator can tune the
    # starting batch size for the next run and avoid retry overhead.
    if _oom_halving_count > 0:
        print(
            f"[eval_exact] Note: OOM fallback halved batch size "
            f"{_oom_halving_count} time(s) during this run. Consider "
            f"starting future runs with a smaller --batch-size-* to avoid "
            f"the retry overhead.",
            file=sys.stderr,
        )

    print(f"\n[eval_exact] Done. Results written to: {out_dir}")


# =============================================================================
# Argument parsing
# =============================================================================


# Allowlist for checkpoint names: letters, digits, dot, underscore, hyphen only.
# This is a positive allowlist (tighter than a denylist) that rejects spaces,
# accented characters, control characters, slashes, and anything else not
# explicitly permitted. The name flows into the output filename
# ``details_{name}_{source}.csv``, so only filesystem-safe printable ASCII
# characters are allowed.
_SAFE_CHECKPOINT_NAME = re.compile(r"^[A-Za-z0-9_.-]+$")


def _parse_checkpoint_pair(value: str) -> tuple[str, str]:
    """Parse a ``name=path`` pair from the command line.

    Parameters
    ----------
    value
        A string like ``"base=Qwen/Qwen2.5-0.5B"`` or
        ``"combined=/tmp/my_checkpoint"``.

    Returns
    -------
    (name, path) tuple.

    Raises
    ------
    argparse.ArgumentTypeError
        If the value does not contain exactly one ``=`` separator,
        if the name is empty or the path is empty, or if the name
        does not match ``_SAFE_CHECKPOINT_NAME`` (letters, digits,
        dot, underscore, hyphen only).

    Security note: the name flows into the output filename
    ``details_{name}_{source}.csv``.  Without this guard a caller
    passing ``--checkpoints "../etc=path"`` would write the CSV
    outside ``--out-dir``.  We reject the name — not the path —
    because the path is an intentional file-system reference while
    the name is only a human-readable label that becomes part of a
    filename.
    """
    if "=" not in value:
        raise argparse.ArgumentTypeError(
            f"checkpoint must be in the form name=path, got {value!r}"
        )
    name, _, path = value.partition("=")
    if not name:
        raise argparse.ArgumentTypeError(
            f"checkpoint name is empty in {value!r}"
        )
    if not path:
        raise argparse.ArgumentTypeError(
            f"checkpoint path is empty in {value!r}"
        )
    # Guard: the name is used verbatim in the output filename.  Reject any
    # name that contains characters outside the safe allowlist.
    if not _SAFE_CHECKPOINT_NAME.match(name):
        raise argparse.ArgumentTypeError(
            f"checkpoint name {name!r} must match {_SAFE_CHECKPOINT_NAME.pattern} "
            f"(letters, digits, dot, underscore, hyphen only)"
        )
    return name, path


def _build_parser() -> argparse.ArgumentParser:
    """Build and return the CLI argument parser.

    Separated from ``main()`` so it can be tested directly and so the
    help text is easy to inspect.
    """
    parser = argparse.ArgumentParser(
        prog="python -m finpost.evals.eval_exact",
        description=(
            "Evaluate one or more checkpoints on one or more benchmark sources "
            "using exact-answer matching. Writes five output files per run: "
            "accuracy_summary.{json,csv}, details_<ckpt>_<src>.csv, "
            "run_metadata.json, cost_summary.json."
        ),
    )

    parser.add_argument(
        "--checkpoints",
        nargs="+",
        required=True,
        metavar="NAME=PATH",
        help=(
            "One or more name=path pairs. PATH is a local directory or a "
            "Hugging Face model id. Example: "
            "--checkpoints base=Qwen/Qwen2.5-0.5B combined=/tmp/step-00003000 "
            "Each path must be either an HF Hub model id or a local directory "
            "in HF format (containing config.json and tokenizer files). "
            "Raw training checkpoints from finpost.training.save_checkpoint are "
            "NOT supported -- convert them to HF format first."
        ),
    )
    parser.add_argument(
        "--sources",
        nargs="+",
        required=True,
        choices=sorted(REGISTRY.keys()),
        help="Sources to evaluate on. Choices: %(choices)s",
    )
    parser.add_argument(
        "--n",
        type=int,
        required=True,
        help="Number of examples per source (seeded sample of the configured split).",
    )
    parser.add_argument(
        "--finchain-split",
        choices=["validation", "test"],
        default="test",
        help=(
            "FinChain split to evaluate: validation for checkpoint selection, "
            "test only for final reporting. Default: %(default)s."
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed controlling example subsample. Default: %(default)s",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        required=True,
        help="Directory to write output files to. Created if absent.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        choices=["cuda", "cpu"],
        help=(
            "Device to run generation on. Choices: %(choices)s. "
            "Defaults to 'cuda' if available, else 'cpu'."
        ),
    )
    parser.add_argument(
        "--gpu-cost-per-hour",
        type=float,
        default=None,
        metavar="USD",
        help=(
            "GPU cost in USD per hour. If supplied, cost_summary.json will "
            "include estimated_cost_usd. If omitted, estimated_cost_usd is null."
        ),
    )
    parser.add_argument(
        "--sample-shard-id",
        type=int,
        default=None,
        metavar="K",
        help=(
            "Evaluate only shard K of the seeded sample. Must be used with "
            "--num-sample-shards. Sharding is round-robin after sampling."
        ),
    )
    parser.add_argument(
        "--num-sample-shards",
        type=int,
        default=None,
        metavar="N",
        help=(
            "Total number of example shards for --sample-shard-id. This is "
            "for subprocess/GPU-level eval parallelism, not model sharding."
        ),
    )

    parser.add_argument(
        "--enable-chaineval",
        action="store_true",
        help=(
            "Run post-hoc ChainEval scoring (step recall/precision, ROUGE, "
            "BERTScore, DTW) on every (gold, prediction) pair after "
            'generation. Requires installing the [chaineval] extra: '
            'pip install -e ".[chaineval]". Raises immediately if the extra '
            "is not installed."
        ),
    )
    parser.add_argument(
        "--chaineval-batch-size",
        type=int,
        default=8,
        metavar="N",
        help=(
            "Reserved for batching BERTScore and sentence-transformer embedding "
            "calls. Default: %(default)s. Has no effect today; accepted so the "
            "CLI surface is stable when batching is added later."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help=(
            "Validate CLI wiring, resolve requested splits, and write "
            "run_metadata.json without loading model weights or generating."
        ),
    )

    # Per-source batch size flags. One flag per registered source so the
    # help text is explicit. Defaults are conservative and halve on OOM.
    _batch_size_defaults: dict[str, int] = {}
    for source_name in sorted(REGISTRY.keys()):
        default = _batch_size_defaults.get(source_name, 4)
        parser.add_argument(
            f"--batch-size-{source_name}",
            type=int,
            default=default,
            metavar="N",
            help=(
                f"Starting batch size for {source_name} generation. "
                f"Halved on OOM. Default: %(default)s"
            ),
        )

    return parser


def main(argv: list[str] | None = None) -> None:
    """CLI entry point.

    Parameters
    ----------
    argv
        Argument list (defaults to ``sys.argv[1:]``). Exposed as a
        parameter so the CLI can be exercised from tests without
        subprocess.
    """
    parser = _build_parser()
    args = parser.parse_args(argv)

    # Parse checkpoint pairs from the list of "name=path" strings.
    checkpoints: dict[str, str] = {}
    for pair in args.checkpoints:
        name, path = _parse_checkpoint_pair(pair)
        if name in checkpoints:
            parser.error(f"Duplicate checkpoint name: {name!r}")
        checkpoints[name] = path

    # Resolve device: default to cuda if available, else cpu.
    if args.device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    # Collect per-source batch sizes from the parsed flags.
    batch_sizes: dict[str, int] = {}
    for source_name in REGISTRY:
        flag_attr = f"batch_size_{source_name}"
        if hasattr(args, flag_attr):
            batch_sizes[source_name] = getattr(args, flag_attr)

    if args.dry_run:
        args.out_dir.mkdir(parents=True, exist_ok=True)
        source_splits = {
            source_name: args.finchain_split if source_name == "finchain" else "test"
            for source_name in args.sources
        }
        source_counts: dict[str, int] = {}
        for source_name in args.sources:
            if source_name == "finchain":
                source_counts[source_name] = len(load_finchain(args.finchain_split))
            else:
                source_counts[source_name] = len(REGISTRY[source_name].load_examples())
            if args.n > source_counts[source_name]:
                parser.error(
                    f"n={args.n} exceeds pool size={source_counts[source_name]} "
                    f"for {source_name} split={source_splits[source_name]}"
                )

        generation_settings = {
            source_name: {
                "max_new_tokens": REGISTRY[source_name].default_max_new_tokens,
                "batch_size": batch_sizes[source_name],
            }
            for source_name in args.sources
        }
        _write_run_metadata(
            out_dir=args.out_dir,
            device=device,
            dtype=str(_select_dtype(device)),
            seed=args.seed,
            eval_n=args.n,
            generation_settings=generation_settings,
            checkpoints=checkpoints,
            source_splits=source_splits,
        )
        dry_run_summary = {
            "dry_run": True,
            "sources": args.sources,
            "source_splits": source_splits,
            "source_counts": source_counts,
            "requested_n": args.n,
            "checkpoints": checkpoints,
        }
        (args.out_dir / "dry_run_summary.json").write_text(
            json.dumps(dry_run_summary, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(f"[eval_exact] Dry run OK. Results written to: {args.out_dir}")
        return

    run_eval(
        checkpoints=checkpoints,
        sources=args.sources,
        n=args.n,
        seed=args.seed,
        out_dir=args.out_dir,
        batch_sizes=batch_sizes,
        gpu_cost_per_hour=args.gpu_cost_per_hour,
        device=device,
        enable_chaineval=args.enable_chaineval,
        chaineval_batch_size=args.chaineval_batch_size,
        sample_shard_id=args.sample_shard_id,
        num_sample_shards=args.num_sample_shards,
        finchain_split=args.finchain_split,
    )


# =============================================================================
# Module entry point
# =============================================================================


if __name__ == "__main__":
    main()
