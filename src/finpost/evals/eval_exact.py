"""Exact-answer evaluation CLI for Phase 1 post-training experiments.

Usage (from the repo root)::

    python -m finpost.evals.eval_exact \\
        --checkpoints base=Qwen/Qwen2.5-0.5B combined=results/checkpoints/step-00003000 \\
        --sources gsm8k math \\
        --n 500 \\
        --seed 42 \\
        --out-dir results/evals/base_vs_sft/ \\
        --batch-size-gsm8k 8 \\
        --batch-size-math 4 \\
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
- Byte-identical re-runs: the details CSV contains no timestamps or
  run-specific state. The sample order comes from a seeded shuffle.
  Greedy decoding is fully deterministic. Together these guarantee that
  ``details_*.csv`` is byte-for-byte identical across runs with the same
  seed on the same device and dtype.
"""

from __future__ import annotations

import argparse
import csv
import datetime
import json
import random
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from finpost.evals.sources import REGISTRY, EvalSource

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
        tracker.write()  # writes cost_summary.json

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
        """Accumulate generated token count across all batches and sources."""
        self.total_generated_tokens += n

    def write(self) -> None:
        """Write cost_summary.json to out_dir."""
        tokens_per_second = (
            self.total_generated_tokens / self.elapsed_sec
            if self.elapsed_sec > 0
            else 0.0
        )

        # Dollar cost: (elapsed_sec / 3600) * cost_per_hour.
        # Only computed if the caller supplied --gpu-cost-per-hour.
        estimated_cost_usd: float | None = None
        if self.gpu_cost_per_hour is not None:
            estimated_cost_usd = (self.elapsed_sec / 3600.0) * self.gpu_cost_per_hour

        _write_cost_summary(
            out_dir=self.out_dir,
            run_name=self.run_name,
            start_time=self.start_time,
            end_time=self.end_time,
            elapsed_sec=self.elapsed_sec,
            gpu_type=self.gpu_type,
            dtype=str(torch.float32),  # updated after model load if needed
            generated_tokens=self.total_generated_tokens,
            tokens_per_second=tokens_per_second,
            estimated_cost_usd=estimated_cost_usd,
        )


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
) -> list[str]:
    """Generate continuations for ``prompts`` in batches.

    Processes ``prompts`` in chunks of at most ``batch_size``. On
    ``torch.cuda.OutOfMemoryError``, halves the batch size and retries
    the same chunk. If batch_size reaches 1 and still OOMs, raises.

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
        Starting batch size. Will be halved on OOM.
    max_new_tokens
        Token budget for the continuation (set per source in the registry).
    device
        ``"cuda"`` or ``"cpu"``.

    Returns
    -------
    A list of generated strings (new tokens only, decoded), one per prompt.
    """
    all_generated: list[str] = []

    # Walk through all prompts in chunks of ``batch_size``.
    i = 0
    current_batch_size = batch_size
    while i < len(prompts):
        chunk = prompts[i : i + current_batch_size]
        generated = _generate_chunk_with_oom_fallback(
            model=model,
            tokenizer=tokenizer,
            prompts=chunk,
            batch_size=current_batch_size,
            max_new_tokens=max_new_tokens,
            device=device,
        )
        all_generated.extend(generated)
        i += len(chunk)

    return all_generated


def _generate_chunk_with_oom_fallback(
    model: Any,
    tokenizer: Any,
    prompts: list[str],
    batch_size: int,
    max_new_tokens: int,
    device: str,
) -> list[str]:
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
    Generated strings (new tokens only), one per prompt.

    Raises
    ------
    torch.cuda.OutOfMemoryError
        If batch_size is already 1 and generation still OOMs.
    """
    try:
        return _tokenize_and_generate(
            model=model,
            tokenizer=tokenizer,
            prompts=prompts,
            max_new_tokens=max_new_tokens,
            device=device,
        )
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
        return left + right


def _tokenize_and_generate(
    model: Any,
    tokenizer: Any,
    prompts: list[str],
    max_new_tokens: int,
    device: str,
) -> list[str]:
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
    A list of decoded new-token strings, one per prompt.
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

    with torch.no_grad():
        output_ids = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=False,    # greedy decoding: deterministic, reproducible
            use_cache=True,     # KV-cache: speeds up autoregressive steps
        )

    # output_ids shape: (batch, prompt_len + new_tokens).
    # Slice off the prompt tokens to get only the newly generated tokens.
    new_token_ids = output_ids[:, prompt_len:]

    # Decode each row separately. skip_special_tokens=True strips pad/eos.
    generated_texts = [
        tokenizer.decode(row, skip_special_tokens=True)
        for row in new_token_ids
    ]
    return generated_texts


# =============================================================================
# Output artifact writers
# =============================================================================


def _write_accuracy_summary(
    rows: list[dict[str, Any]],
    out_dir: Path,
) -> None:
    """Write accuracy_summary.json and accuracy_summary.csv.

    Each row in ``rows`` must have the keys:
      checkpoint, source, n, accuracy, parse_success_rate,
      generated_tokens, elapsed_sec.

    Parameters
    ----------
    rows
        List of summary dicts, one per (checkpoint, source) pair.
    out_dir
        Directory where the two files are written.
    """
    # Field order for CSV. Defined once here so JSON and CSV are consistent.
    fieldnames = [
        "checkpoint",
        "source",
        "n",
        "accuracy",
        "parse_success_rate",
        "generated_tokens",
        "elapsed_sec",
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
) -> None:
    """Write details_<checkpoint>_<source>.csv.

    Each row in ``rows`` must have the keys:
      example_id, prompt, generated, gold_answer, predicted_answer,
      parse_success, is_correct.

    The ``generated`` column may contain newlines and commas; the csv
    module's default quoting (QUOTE_MINIMAL) will wrap such cells in
    double quotes automatically, keeping the row count correct on re-read.

    Parameters
    ----------
    rows
        Per-example result dicts.
    checkpoint_name
        User-supplied name (e.g. ``"base"``), used in the filename.
    source_name
        Source identifier (e.g. ``"gsm8k"``), used in the filename.
    out_dir
        Output directory.
    """
    fieldnames = [
        "example_id",
        "prompt",
        "generated",
        "gold_answer",
        "predicted_answer",
        "parse_success",
        "is_correct",
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
    gpu_type: str,
    dtype: str,
    generated_tokens: int,
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
        Wall-clock seconds for the full evaluation.
    gpu_type
        Device name string (e.g. ``"Tesla T4"`` or ``"CPU"``).
    dtype
        Torch dtype string (e.g. ``"torch.bfloat16"``).
    generated_tokens
        Total new tokens produced across all sources.
    tokens_per_second
        Throughput.
    estimated_cost_usd
        Dollar cost if ``--gpu-cost-per-hour`` was supplied, else ``None``.
    """
    data = {
        "run_name": run_name,
        "start_time": start_time,
        "end_time": end_time,
        "elapsed_sec": elapsed_sec,
        "gpu_type": gpu_type,
        "dtype": dtype,
        "generated_tokens": generated_tokens,
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
) -> None:
    """Write run_metadata.json.

    Contains everything needed to reproduce the run: library versions,
    hardware, eval configuration, and the git SHA so you can trace back
    to the exact code state.

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
    """
    import transformers

    # CUDA version: only meaningful when CUDA is available.
    cuda_version = torch.version.cuda if torch.cuda.is_available() else "N/A"

    # Git short SHA: wrap in try/except so a detached HEAD or missing git
    # binary does not abort the evaluation.
    try:
        git_sha = (
            subprocess.check_output(
                ["git", "rev-parse", "--short", "HEAD"],
                stderr=subprocess.DEVNULL,
            )
            .decode()
            .strip()
        )
    except Exception:
        git_sha = "unknown"

    data = {
        "device": device,
        "dtype": dtype,
        "torch_version": torch.__version__,
        "transformers_version": transformers.__version__,
        "cuda_version": cuda_version,
        "seed": seed,
        "eval_n": eval_n,
        "git_sha": git_sha,
        "generation_settings": generation_settings,
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

    Uses ``use_safetensors=True`` by default, consistent with the
    project's SECURITY.md policy: refuse pickle-format weight files.

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
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    # Left-padding: decoder-only models must have the last prompt token
    # at the right edge of the input so the first generated token
    # attends to the correct context.
    tokenizer.padding_side = "left"

    # Many GPT-style tokenizers have no pad_token. Reuse eos_token as pad.
    # The attention_mask ensures the model ignores the padding positions.
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ``use_safetensors=True`` refuses to load pickle-format .bin weights.
    # For tiny-gpt2 (which ships .bin only), we fall back to the default
    # auto-detection by catching the error. Real Phase 1 models (Qwen) ship
    # safetensors and succeed on the first try.
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            dtype=dtype,
            use_safetensors=True,
        ).to(device)
    except Exception:
        # Fallback for models that ship only legacy .bin weights (e.g. tiny-gpt2
        # used for unit tests). The fallback is intentional and documented.
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            dtype=dtype,
        ).to(device)

    model.eval()
    return model, tokenizer


def _unload_model(model: Any) -> None:
    """Delete model weights and release GPU memory.

    Called between checkpoints so only one model occupies VRAM at a time.
    """
    del model
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
) -> tuple[list[dict[str, Any]], int]:
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
    (detail_rows, total_new_tokens)
        ``detail_rows`` is a list of per-example dicts ready to write to CSV.
        ``total_new_tokens`` is the count of newly generated tokens.
    """
    prompts = [ex.prompt for ex in examples]

    # Generate all continuations for this source in one batched call.
    # _generate_batch handles the chunking and OOM fallback internally.
    generated_texts = _generate_batch(
        model=model,
        tokenizer=tokenizer,
        prompts=prompts,
        batch_size=batch_size,
        max_new_tokens=source.default_max_new_tokens,
        device=device,
    )

    # Count newly generated tokens for cost tracking.
    # We re-tokenize to get an accurate token count (decode→encode is the
    # safest way; tokenizer.encode does not add special tokens by default).
    total_new_tokens = sum(
        len(tokenizer.encode(gen, add_special_tokens=False))
        for gen in generated_texts
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

    return detail_rows, total_new_tokens


# =============================================================================
# Top-level run_eval function (callable from tests and the CLI)
# =============================================================================


def run_eval(
    checkpoints: dict[str, str],
    sources: list[str],
    n: int,
    seed: int,
    out_dir: Path,
    batch_sizes: dict[str, int],
    gpu_cost_per_hour: float | None,
    device: str,
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
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

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
        all_examples = source.load_examples()
        sampled = _sample_examples(all_examples, n=n, seed=seed)
        sampled_examples[source.name] = sampled
        print(f"  {source.name}: {len(sampled)} examples (from {len(all_examples)} total)")

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
                detail_rows, new_tokens = _evaluate_one_source(
                    model=model,
                    tokenizer=tokenizer,
                    source=source,
                    examples=examples,
                    batch_size=batch_size,
                    device=device,
                )
                elapsed = time.monotonic() - t0

                tracker.add_generated_tokens(new_tokens)

                # Compute aggregate metrics.
                n_evaluated = len(detail_rows)
                n_correct = sum(1 for row in detail_rows if row["is_correct"])
                n_parsed = sum(1 for row in detail_rows if row["parse_success"])
                accuracy = n_correct / n_evaluated if n_evaluated > 0 else 0.0
                parse_success_rate = n_parsed / n_evaluated if n_evaluated > 0 else 0.0

                print(
                    f"    accuracy={accuracy:.3f}  parse_success_rate={parse_success_rate:.3f}"
                    f"  elapsed={elapsed:.1f}s  tokens={new_tokens}"
                )

                # Write per-example details CSV.
                _write_details_csv(
                    rows=detail_rows,
                    checkpoint_name=ckpt_name,
                    source_name=source.name,
                    out_dir=out_dir,
                )

                # Accumulate for the summary files.
                summary_rows.append({
                    "checkpoint": ckpt_name,
                    "source": source.name,
                    "n": n_evaluated,
                    "accuracy": accuracy,
                    "parse_success_rate": parse_success_rate,
                    "generated_tokens": new_tokens,
                    "elapsed_sec": round(elapsed, 3),
                })

            # Done with this checkpoint. Free GPU memory before the next one.
            print(f"  Unloading {ckpt_name!r}")
            _unload_model(model)

    # Write the three summary files (happens outside the tracker context manager,
    # after tracker.elapsed_sec has been set by __exit__).
    _write_accuracy_summary(rows=summary_rows, out_dir=out_dir)

    # Write cost_summary.json directly rather than via tracker.write(), so
    # we can supply the correct dtype string (only known after model load).
    tokens_per_second = (
        tracker.total_generated_tokens / tracker.elapsed_sec
        if tracker.elapsed_sec > 0
        else 0.0
    )
    estimated_cost_usd: float | None = None
    if gpu_cost_per_hour is not None:
        estimated_cost_usd = (tracker.elapsed_sec / 3600.0) * gpu_cost_per_hour

    _write_cost_summary(
        out_dir=out_dir,
        run_name=run_name,
        start_time=tracker.start_time,
        end_time=tracker.end_time,
        elapsed_sec=tracker.elapsed_sec,
        gpu_type=tracker.gpu_type,
        dtype=dtype_str,
        generated_tokens=tracker.total_generated_tokens,
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
    )

    print(f"\n[eval_exact] Done. Results written to: {out_dir}")


# =============================================================================
# Argument parsing
# =============================================================================


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
        If the value does not contain exactly one ``=`` separator.
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
            "--checkpoints base=Qwen/Qwen2.5-0.5B combined=/tmp/step-00003000"
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
        help="Number of examples per source (seeded random sample of the test split).",
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
        help=(
            "Device to run generation on. Defaults to 'cuda' if available, "
            "else 'cpu'."
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

    # Per-source batch size flags. One flag per registered source so the
    # help text is explicit. Defaults are T4-safe starting points; both will
    # halve automatically on OOM. gsm8k=8 (256-token budget), math=4
    # (768-token budget — much tighter memory per sample).
    _batch_size_defaults: dict[str, int] = {"gsm8k": 8, "math": 4}
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

    run_eval(
        checkpoints=checkpoints,
        sources=args.sources,
        n=args.n,
        seed=args.seed,
        out_dir=args.out_dir,
        batch_sizes=batch_sizes,
        gpu_cost_per_hour=args.gpu_cost_per_hour,
        device=device,
    )


# =============================================================================
# Module entry point
# =============================================================================


if __name__ == "__main__":
    main()
