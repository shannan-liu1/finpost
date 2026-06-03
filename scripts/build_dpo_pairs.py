"""Build offline preference pairs for DPO.

The script samples multiple completions from an HF-format causal LM checkpoint,
grades each completion with the same exact-answer source registry used by eval,
and writes JSONL preference pairs where a correct completion is chosen over an
incorrect completion from the same prompt.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import torch
from tqdm.auto import tqdm

from finpost.data.finchain_dataset import load_finchain, resolve_finchain_path
from finpost.data.schema import Example, Source
from finpost.evals.sources import REGISTRY
from finpost.safety import safe_load_model, safe_load_tokenizer
from finpost.training.dataset import serialize_prompt

FINCHAIN_FULL_DPO_PAIR_QUALITY_THRESHOLDS: dict[str, float | int | bool] = {
    "min_pair_count": 200,
    "min_unique_prompt_count": 200,
    "min_training_prompt_coverage_rate": 1.0,
    "min_pairable_prompt_rate": 1.0,
    "max_pairs_for_one_prompt": 1,
    "max_duplicate_exact_pair_count": 0,
    "max_rejected_parse_failure_rate": 0.50,
    "max_single_prompt_pair_share": 0.02,
    "min_unique_topic_count": 25,
    "min_unique_difficulty_count": 3,
    "require_full_prompt_coverage": True,
    "require_one_pair_per_prompt": True,
    "require_prompt_metadata": True,
}


@dataclass(frozen=True)
class CompletionRecord:
    """One sampled completion plus exact-answer grade metadata."""

    prompt_id: str
    prompt: str
    source: Source
    gold_answer: str
    sample_index: int
    completion: str
    predicted_answer: str | None
    correct: bool
    difficulty: int | None = None
    domain: str | None = None
    topic: str | None = None
    subtopic: str | None = None


def _load_train_examples(source: Source) -> list[Example]:
    if source == "finchain":
        return load_finchain("train")
    raise ValueError(f"unsupported source: {source}")


def select_train_prompts(
    *,
    sources: list[Source],
    heldout_train_n: int,
    seed: int,
) -> list[Example]:
    """Select a deterministic, source-balanced prompt subset from train splits."""
    if heldout_train_n <= 0:
        raise ValueError("heldout_train_n must be positive")
    if not sources:
        raise ValueError("at least one source is required")

    per_source = heldout_train_n // len(sources)
    remainder = heldout_train_n % len(sources)
    selected: list[Example] = []
    for idx, source in enumerate(sources):
        target = per_source + (1 if idx < remainder else 0)
        examples = _load_train_examples(source)
        if target > len(examples):
            raise ValueError(
                f"requested {target} {source} prompts, but only {len(examples)} train records exist"
            )
        rng = random.Random(f"{seed}:{source}:dpo")
        source_examples = list(examples)
        rng.shuffle(source_examples)
        selected.extend(source_examples[:target])

    rng = random.Random(f"{seed}:combined:dpo")
    rng.shuffle(selected)
    return selected


def shard_examples(
    examples: list[Example],
    *,
    shard_id: int,
    num_shards: int,
) -> list[Example]:
    """Return the deterministic prompt slice assigned to one rollout worker."""
    if num_shards <= 0:
        raise ValueError("num_shards must be positive")
    if not 0 <= shard_id < num_shards:
        raise ValueError("shard_id must satisfy 0 <= shard_id < num_shards")
    return examples[shard_id::num_shards]


def _completion_grade(example: Example, completion: str) -> tuple[str | None, bool]:
    source = REGISTRY[example.source]
    predicted = source.extract_answer(completion)
    return predicted, source.score(predicted, example.final_answer)


def _to_pair(
    *,
    prompt_id: str,
    prompt: str,
    source: Source,
    gold_answer: str,
    chosen: CompletionRecord,
    rejected: CompletionRecord,
) -> dict[str, Any]:
    return {
        "prompt": prompt,
        "chosen": chosen.completion,
        "rejected": rejected.completion,
        "source": source,
        "prompt_id": prompt_id,
        "chosen_grade": {
            "sample_index": chosen.sample_index,
            "predicted_answer": chosen.predicted_answer,
            "correct": chosen.correct,
        },
        "rejected_grade": {
            "sample_index": rejected.sample_index,
            "predicted_answer": rejected.predicted_answer,
            "correct": rejected.correct,
        },
        "metadata": {
            "gold_answer": gold_answer,
            "pair_strategy": "correct_vs_incorrect_same_prompt",
            "difficulty": chosen.difficulty,
            "domain": chosen.domain,
            "topic": chosen.topic,
            "subtopic": chosen.subtopic,
        },
    }


def build_pairs_from_completions(
    records: list[CompletionRecord],
    *,
    max_pairs_per_prompt: int | None = None,
    seed: int = 0,
) -> list[dict[str, Any]]:
    """Create chosen/rejected JSON records from graded completions."""
    if max_pairs_per_prompt is not None and max_pairs_per_prompt <= 0:
        raise ValueError("max_pairs_per_prompt must be positive when provided")

    grouped: dict[str, list[CompletionRecord]] = {}
    for record in records:
        grouped.setdefault(record.prompt_id, []).append(record)

    pairs: list[dict[str, Any]] = []
    for prompt_id in sorted(grouped):
        group = sorted(grouped[prompt_id], key=lambda row: row.sample_index)
        correct = [row for row in group if row.correct]
        incorrect = [row for row in group if not row.correct]
        if not correct or not incorrect:
            continue

        # Cross-product chosen x incorrect, but skip pairs where the
        # generated text is identical between sides. A tied pair carries
        # zero preference signal (the DPO loss reduces to 0) and just
        # wastes a training-batch slot. At temp=0.8 / top_p=0.95 with
        # 1.5B-class models this is rare, but cheap to defend against.
        prompt_pairs = [
            _to_pair(
                prompt_id=prompt_id,
                prompt=chosen.prompt,
                source=chosen.source,
                gold_answer=chosen.gold_answer,
                chosen=chosen,
                rejected=rejected,
            )
            for chosen in correct
            for rejected in incorrect
            if chosen.completion != rejected.completion
        ]
        deduped_prompt_pairs: list[dict[str, Any]] = []
        seen_exact_pairs: set[tuple[str, str]] = set()
        for pair in prompt_pairs:
            key = (str(pair["chosen"]), str(pair["rejected"]))
            if key in seen_exact_pairs:
                continue
            seen_exact_pairs.add(key)
            deduped_prompt_pairs.append(pair)
        prompt_pairs = deduped_prompt_pairs
        if max_pairs_per_prompt is not None and len(prompt_pairs) > max_pairs_per_prompt:
            rng = random.Random(f"{seed}:{prompt_id}:pairs")
            rng.shuffle(prompt_pairs)
            prompt_pairs = prompt_pairs[:max_pairs_per_prompt]
        pairs.extend(prompt_pairs)

    deduped_pairs: list[dict[str, Any]] = []
    seen_exact_rows: set[tuple[str, str, str]] = set()
    for pair in pairs:
        key = (
            str(pair.get("prompt_id") or pair.get("prompt") or ""),
            str(pair.get("chosen", "")),
            str(pair.get("rejected", "")),
        )
        if key in seen_exact_rows:
            continue
        seen_exact_rows.add(key)
        deduped_pairs.append(pair)
    return deduped_pairs


def summarize_pair_rows(pairs: list[dict[str, Any]]) -> dict[str, int]:
    """Expose prompt coverage and exact duplicate rate for DPO artifacts."""
    prompt_keys = [
        str(pair.get("prompt_id") or pair.get("prompt") or "")
        for pair in pairs
    ]
    exact_pair_keys = [
        (
            str(pair.get("prompt_id") or pair.get("prompt") or ""),
            str(pair.get("chosen", "")),
            str(pair.get("rejected", "")),
        )
        for pair in pairs
    ]
    prompt_counts = Counter(prompt_keys)
    return {
        "pair_count": len(pairs),
        "unique_prompt_count": len(prompt_counts),
        "unique_exact_pair_count": len(set(exact_pair_keys)),
        "duplicate_exact_pair_count": len(pairs) - len(set(exact_pair_keys)),
        "max_pairs_for_one_prompt": max(prompt_counts.values(), default=0),
    }


def _metadata_value(pair: dict[str, Any], key: str) -> Any:
    metadata = pair.get("metadata")
    if isinstance(metadata, dict) and metadata.get(key) is not None:
        return metadata[key]
    return pair.get(key)


def _rate(numerator: int | float, denominator: int | float) -> float:
    if denominator == 0:
        return 0.0
    return round(float(numerator) / float(denominator), 6)


def summarize_pair_quality(
    pairs: list[dict[str, Any]],
    *,
    group_summary: dict[str, int] | None = None,
) -> dict[str, Any]:
    """Summarize whether DPO pairs are broad, parseable, and train-worthy."""
    row_summary = summarize_pair_rows(pairs)
    prompt_count = int((group_summary or {}).get("prompt_count", 0))
    pairable_prompt_count = int((group_summary or {}).get("pairable_prompt_count", 0))
    rejected_parse_failure_count = sum(
        1
        for pair in pairs
        if isinstance(pair.get("rejected_grade"), dict)
        and pair["rejected_grade"].get("predicted_answer") is None
    )
    chosen_parse_failure_count = sum(
        1
        for pair in pairs
        if isinstance(pair.get("chosen_grade"), dict)
        and pair["chosen_grade"].get("predicted_answer") is None
    )
    missing_metadata_count = sum(
        1
        for pair in pairs
        if any(_metadata_value(pair, key) is None for key in ("difficulty", "topic", "domain"))
    )
    topic_counts = Counter(
        str(_metadata_value(pair, "topic"))
        for pair in pairs
        if _metadata_value(pair, "topic") is not None
    )
    difficulty_counts = Counter(
        str(_metadata_value(pair, "difficulty"))
        for pair in pairs
        if _metadata_value(pair, "difficulty") is not None
    )
    source_counts = Counter(str(pair.get("source", "")) for pair in pairs)
    pair_count = len(pairs)
    max_pairs_for_one_prompt = int(row_summary["max_pairs_for_one_prompt"])

    return {
        **row_summary,
        "sampled_prompt_count": prompt_count,
        "pairable_prompt_count": pairable_prompt_count,
        "pairable_prompt_rate": _rate(pairable_prompt_count, prompt_count),
        "training_prompt_coverage_rate": _rate(
            row_summary["unique_prompt_count"],
            prompt_count,
        ),
        "chosen_parse_failure_count": chosen_parse_failure_count,
        "rejected_parse_failure_count": rejected_parse_failure_count,
        "rejected_parse_failure_rate": _rate(rejected_parse_failure_count, pair_count),
        "missing_prompt_metadata_count": missing_metadata_count,
        "max_single_prompt_pair_share": _rate(max_pairs_for_one_prompt, pair_count),
        "unique_topic_count": len(topic_counts),
        "unique_difficulty_count": len(difficulty_counts),
        "source_counts": dict(sorted(source_counts.items())),
        "topic_counts": dict(sorted(topic_counts.items())),
        "difficulty_counts": dict(sorted(difficulty_counts.items())),
    }


def build_pair_quality_gate_report(
    quality: dict[str, Any],
    *,
    thresholds: dict[str, float | int | bool],
) -> dict[str, Any]:
    """Return pass/fail details for DPO pair quality thresholds."""
    failures: list[str] = []

    minimum_fields = {
        "pair_count": "min_pair_count",
        "unique_prompt_count": "min_unique_prompt_count",
        "training_prompt_coverage_rate": "min_training_prompt_coverage_rate",
        "pairable_prompt_rate": "min_pairable_prompt_rate",
        "unique_topic_count": "min_unique_topic_count",
        "unique_difficulty_count": "min_unique_difficulty_count",
    }
    for field, threshold_key in minimum_fields.items():
        threshold = thresholds.get(threshold_key)
        if threshold is not None and quality.get(field, 0) < threshold:
            failures.append(
                f"{field}={quality.get(field, 0)} below {threshold_key}={threshold}"
            )

    maximum_fields = {
        "duplicate_exact_pair_count": "max_duplicate_exact_pair_count",
        "rejected_parse_failure_rate": "max_rejected_parse_failure_rate",
        "max_single_prompt_pair_share": "max_single_prompt_pair_share",
        "max_pairs_for_one_prompt": "max_pairs_for_one_prompt",
    }
    for field, threshold_key in maximum_fields.items():
        threshold = thresholds.get(threshold_key)
        if threshold is not None and quality.get(field, 0) > threshold:
            failures.append(
                f"{field}={quality.get(field, 0)} above {threshold_key}={threshold}"
            )

    if thresholds.get("require_prompt_metadata") and quality.get(
        "missing_prompt_metadata_count",
        0,
    ):
        failures.append(
            "missing_prompt_metadata_count="
            f"{quality.get('missing_prompt_metadata_count')} but metadata is required"
        )

    if thresholds.get("require_full_prompt_coverage"):
        sampled_prompt_count = int(quality.get("sampled_prompt_count", 0) or 0)
        unique_prompt_count = int(quality.get("unique_prompt_count", 0) or 0)
        pairable_prompt_count = int(quality.get("pairable_prompt_count", 0) or 0)
        if sampled_prompt_count <= 0:
            failures.append("sampled_prompt_count is required for full prompt coverage")
        elif unique_prompt_count != sampled_prompt_count:
            failures.append(
                "unique_prompt_count="
                f"{unique_prompt_count} does not equal sampled_prompt_count="
                f"{sampled_prompt_count} required for full prompt coverage"
            )
        if sampled_prompt_count > 0 and pairable_prompt_count != sampled_prompt_count:
            failures.append(
                "pairable_prompt_count="
                f"{pairable_prompt_count} does not equal sampled_prompt_count="
                f"{sampled_prompt_count} required for full prompt coverage"
            )

    if thresholds.get("require_one_pair_per_prompt"):
        pair_count = int(quality.get("pair_count", 0) or 0)
        unique_prompt_count = int(quality.get("unique_prompt_count", 0) or 0)
        if pair_count != unique_prompt_count:
            failures.append(
                "pair_count="
                f"{pair_count} does not equal unique_prompt_count="
                f"{unique_prompt_count} required for one pair per prompt"
            )

    return {
        "passed": not failures,
        "failures": failures,
        "thresholds": dict(thresholds),
    }


def write_pair_review_sample(
    path: Path,
    pairs: list[dict[str, Any]],
    *,
    sample_size: int = 50,
    seed: int = 42,
) -> None:
    """Write a small CSV for manual inspection before paying for DPO training."""
    rng = random.Random(f"{seed}:pair-review-sample")
    sample = list(pairs)
    rng.shuffle(sample)
    sample = sample[: min(sample_size, len(sample))]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fp:
        writer = csv.DictWriter(
            fp,
            fieldnames=[
                "prompt_id",
                "source",
                "difficulty",
                "domain",
                "topic",
                "subtopic",
                "chosen_predicted_answer",
                "rejected_predicted_answer",
                "prompt",
                "chosen",
                "rejected",
            ],
        )
        writer.writeheader()
        for pair in sample:
            chosen_grade = pair.get("chosen_grade") or {}
            rejected_grade = pair.get("rejected_grade") or {}
            writer.writerow(
                {
                    "prompt_id": pair.get("prompt_id"),
                    "source": pair.get("source"),
                    "difficulty": _metadata_value(pair, "difficulty"),
                    "domain": _metadata_value(pair, "domain"),
                    "topic": _metadata_value(pair, "topic"),
                    "subtopic": _metadata_value(pair, "subtopic"),
                    "chosen_predicted_answer": chosen_grade.get("predicted_answer"),
                    "rejected_predicted_answer": rejected_grade.get("predicted_answer"),
                    "prompt": str(pair.get("prompt", ""))[:500],
                    "chosen": str(pair.get("chosen", ""))[:500],
                    "rejected": str(pair.get("rejected", ""))[:500],
                }
            )


def summarize_completion_groups(records: list[CompletionRecord]) -> dict[str, int]:
    """Count prompt groups by whether they can produce DPO pairs."""
    grouped: dict[str, list[CompletionRecord]] = {}
    for record in records:
        grouped.setdefault(record.prompt_id, []).append(record)

    summary = {
        "prompt_count": len(grouped),
        "pairable_prompt_count": 0,
        "all_correct_prompt_count": 0,
        "all_incorrect_prompt_count": 0,
        "empty_prompt_count": 0,
    }
    for group in grouped.values():
        if not group:
            summary["empty_prompt_count"] += 1
            continue
        correct_count = sum(record.correct for record in group)
        if correct_count == len(group):
            summary["all_correct_prompt_count"] += 1
        elif correct_count == 0:
            summary["all_incorrect_prompt_count"] += 1
        else:
            summary["pairable_prompt_count"] += 1
    return summary


def resolve_max_new_tokens_by_source(
    *,
    sources: list[Source],
    max_new_tokens: int,
    max_new_tokens_finchain: int | None = None,
) -> dict[Source, int]:
    """Resolve per-source generation budgets from CLI overrides."""
    overrides = {
        "finchain": max_new_tokens_finchain,
    }
    resolved: dict[Source, int] = {}
    for source in sources:
        budget = overrides[source] if overrides[source] is not None else max_new_tokens
        if budget <= 0:
            raise ValueError(f"max_new_tokens for {source} must be positive")
        resolved[source] = budget
    return resolved


def _batched(items: list[Any], batch_size: int) -> list[list[Any]]:
    return [
        items[start : start + batch_size]
        for start in range(0, len(items), batch_size)
    ]


def sample_completions(
    *,
    model: Any,
    tokenizer: Any,
    examples: list[Example],
    samples_per_prompt: int,
    generation_batch_size: int,
    max_new_tokens_by_source: dict[Source, int],
    temperature: float,
    top_p: float,
    device: torch.device,
) -> list[CompletionRecord]:
    """Generate and grade sampled completions."""
    if samples_per_prompt <= 0:
        raise ValueError("samples_per_prompt must be positive")
    if generation_batch_size <= 0:
        raise ValueError("generation_batch_size must be positive")
    if temperature <= 0.0:
        raise ValueError("temperature must be positive for DPO pair sampling")

    items_by_source: dict[Source, list[tuple[Example, int]]] = {}
    for example in examples:
        for sample_idx in range(samples_per_prompt):
            items_by_source.setdefault(example.source, []).append((example, sample_idx))

    records: list[CompletionRecord] = []
    model.eval()
    model.config.use_cache = True
    if getattr(tokenizer, "pad_token_id", None) is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    for source in sorted(items_by_source):
        items = items_by_source[source]
        max_new_tokens = max_new_tokens_by_source[source]
        for batch in tqdm(
            _batched(items, generation_batch_size),
            desc=f"sampling {source} completions",
            total=(len(items) + generation_batch_size - 1) // generation_batch_size,
        ):
            records.extend(
                _sample_batch_with_oom_fallback(
                    model=model,
                    tokenizer=tokenizer,
                    batch=batch,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    device=device,
                )
            )
    return records


def _is_cuda_oom(exc: RuntimeError) -> bool:
    message = str(exc).lower()
    return "out of memory" in message or "cuda error: out of memory" in message


def _sample_batch_with_oom_fallback(
    *,
    model: Any,
    tokenizer: Any,
    batch: list[tuple[Example, int]],
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    device: torch.device,
) -> list[CompletionRecord]:
    try:
        return _sample_batch(
            model=model,
            tokenizer=tokenizer,
            batch=batch,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            device=device,
        )
    except RuntimeError as exc:
        if not _is_cuda_oom(exc) or len(batch) == 1:
            raise
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        midpoint = len(batch) // 2
        print(
            "[pairs] CUDA OOM while sampling batch "
            f"size {len(batch)}; retrying as {midpoint}+{len(batch) - midpoint}"
        )
        return [
            *_sample_batch_with_oom_fallback(
                model=model,
                tokenizer=tokenizer,
                batch=batch[:midpoint],
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                device=device,
            ),
            *_sample_batch_with_oom_fallback(
                model=model,
                tokenizer=tokenizer,
                batch=batch[midpoint:],
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                device=device,
            ),
        ]


def _sample_batch(
    *,
    model: Any,
    tokenizer: Any,
    batch: list[tuple[Example, int]],
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    device: torch.device,
) -> list[CompletionRecord]:
    prompts = [serialize_prompt(example.prompt) for example, _ in batch]
    encoded = tokenizer(prompts, return_tensors="pt", padding=True)
    encoded = {key: value.to(device) for key, value in encoded.items()}
    input_width = encoded["input_ids"].shape[1]
    with torch.inference_mode():
        output_ids = model.generate(
            **encoded,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            max_new_tokens=max_new_tokens,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            use_cache=True,
        )

    records: list[CompletionRecord] = []
    for row_idx, (example, sample_idx) in enumerate(batch):
        generated_ids = output_ids[row_idx, input_width:]
        completion = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
        predicted, correct = _completion_grade(example, completion)
        records.append(
            CompletionRecord(
                prompt_id=example.id,
                prompt=example.prompt,
                source=example.source,
                gold_answer=example.final_answer,
                sample_index=sample_idx,
                completion=completion,
                predicted_answer=predicted,
                correct=correct,
                difficulty=example.difficulty,
                domain=example.domain,
                topic=example.topic,
                subtopic=example.subtopic,
            )
        )
    return records


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as fp:
        for row in rows:
            fp.write(json.dumps(row, ensure_ascii=True) + "\n")


def _sha256_file(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as fp:
        for chunk in iter(lambda: fp.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def _fingerprint_path(path: Path) -> str | None:
    if not path.exists():
        return None
    hasher = hashlib.sha256()
    if path.is_file():
        files = [path]
        root = path.parent
    else:
        files = sorted(item for item in path.rglob("*") if item.is_file())
        root = path
    for item in files:
        stat = item.stat()
        relative = item.relative_to(root).as_posix()
        hasher.update(f"{relative}:{stat.st_size}:{stat.st_mtime_ns}\n".encode())
    return hasher.hexdigest()


def data_provenance_for_sources(sources: list[Source]) -> dict[str, Any]:
    """Record local data paths/checksums that make pair shards comparable."""
    data_paths: dict[str, str] = {}
    data_checksums: dict[str, str] = {}
    if "finchain" in sources:
        path = resolve_finchain_path("train")
        data_paths["finchain_train"] = str(path)
        if path.exists():
            data_checksums["finchain_train"] = _sha256_file(path)
    return {
        "data_paths": data_paths,
        "data_checksums_sha256": data_checksums,
    }


def quality_thresholds_by_name(name: str) -> dict[str, float | int | bool] | None:
    if name == "none":
        return None
    if name == "finchain-full":
        return FINCHAIN_FULL_DPO_PAIR_QUALITY_THRESHOLDS
    raise ValueError(f"unsupported quality gate: {name}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model-checkpoint",
        "--sft-checkpoint",
        dest="model_checkpoint",
        type=Path,
        required=True,
        help=(
            "HF-format causal LM checkpoint to sample from. "
            "--sft-checkpoint is kept as a backward-compatible alias."
        ),
    )
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument(
        "--sources",
        nargs="+",
        choices=["finchain"],
        default=["finchain"],
    )
    parser.add_argument("--heldout-train-n", type=int, default=2000)
    parser.add_argument("--samples-per-prompt", type=int, default=8)
    parser.add_argument("--generation-batch-size", type=int, default=64)
    parser.add_argument("--max-new-tokens", type=int, default=768)
    parser.add_argument("--max-new-tokens-finchain", type=int, default=None)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--max-pairs-per-prompt", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--shard-id",
        type=int,
        default=0,
        help="Zero-based rollout shard id. Use with --num-shards for multi-GPU sampling.",
    )
    parser.add_argument(
        "--num-shards",
        type=int,
        default=1,
        help="Total number of rollout shards. Each shard writes its own out-dir.",
    )
    parser.add_argument(
        "--dtype",
        choices=["float32", "bfloat16", "float16"],
        default="bfloat16",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="'auto', 'cpu', 'cuda', or a torch device",
    )
    parser.add_argument(
        "--quality-gate",
        choices=["none", "finchain-full"],
        default="none",
        help="Optional DPO pair quality gate to enforce after pair generation.",
    )
    parser.add_argument(
        "--allow-empty-pairs",
        action="store_true",
        help=(
            "Allow a smoke-test run to succeed when no correct-vs-incorrect "
            "pairs are produced. Do not use with serious DPO pair generation."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
        torch.set_float32_matmul_precision("high")

    device_name = (
        "cuda"
        if args.device == "auto" and torch.cuda.is_available()
        else args.device
    )
    if args.device == "auto" and not torch.cuda.is_available():
        device_name = "cpu"
    device = torch.device(device_name)
    torch_dtype = getattr(torch, args.dtype)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    completions_path = args.out_dir / "completions.jsonl"
    pairs_path = args.out_dir / "pairs.jsonl"
    review_sample_path = args.out_dir / "pair_review_sample.csv"
    manifest_path = args.out_dir / "manifest.json"

    print(f"[pairs] loading train prompts from {args.sources}")
    examples = select_train_prompts(
        sources=args.sources,
        heldout_train_n=args.heldout_train_n,
        seed=args.seed,
    )
    examples = shard_examples(
        examples,
        shard_id=args.shard_id,
        num_shards=args.num_shards,
    )
    if not examples:
        raise ValueError(
            f"shard {args.shard_id}/{args.num_shards} received zero prompts; "
            "reduce --num-shards or increase --heldout-train-n"
        )
    source_counts = {
        source: sum(ex.source == source for ex in examples)
        for source in args.sources
    }
    print(
        f"[pairs] selected {len(examples)} prompts for shard "
        f"{args.shard_id}/{args.num_shards}: {source_counts}"
    )
    max_new_tokens_by_source = resolve_max_new_tokens_by_source(
        sources=args.sources,
        max_new_tokens=args.max_new_tokens,
        max_new_tokens_finchain=args.max_new_tokens_finchain,
    )
    print(f"[pairs] max_new_tokens_by_source: {max_new_tokens_by_source}")

    print(f"[pairs] loading model from {args.model_checkpoint} on {device} ({args.dtype})")
    tokenizer = safe_load_tokenizer(str(args.model_checkpoint))
    model = safe_load_model(
        str(args.model_checkpoint),
        dtype=torch_dtype,
        use_safetensors=True,
    ).to(device)

    records = sample_completions(
        model=model,
        tokenizer=tokenizer,
        examples=examples,
        samples_per_prompt=args.samples_per_prompt,
        generation_batch_size=args.generation_batch_size,
        max_new_tokens_by_source=max_new_tokens_by_source,
        temperature=args.temperature,
        top_p=args.top_p,
        device=device,
    )
    completion_rows = [asdict(record) for record in records]
    _write_jsonl(completions_path, completion_rows)

    pairs = build_pairs_from_completions(
        records,
        max_pairs_per_prompt=args.max_pairs_per_prompt,
        seed=args.seed,
    )
    if not pairs and not args.allow_empty_pairs:
        raise ValueError(
            "no DPO pairs were produced; increase samples_per_prompt/temperature or inspect "
            f"{completions_path}"
        )
    sampling_metadata = {
        "samples_per_prompt": args.samples_per_prompt,
        "max_new_tokens": args.max_new_tokens,
        "max_new_tokens_by_source": max_new_tokens_by_source,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "max_pairs_per_prompt": args.max_pairs_per_prompt,
    }
    for pair in pairs:
        pair["metadata"].update(
            {
                "source_checkpoint": str(args.model_checkpoint),
                "sampling": sampling_metadata,
                "seed": args.seed,
                "verifier": "finpost.evals.sources.REGISTRY",
            }
        )
    _write_jsonl(pairs_path, pairs)
    write_pair_review_sample(review_sample_path, pairs, seed=args.seed)

    correct_count = sum(record.correct for record in records)
    group_summary = summarize_completion_groups(records)
    pair_summary = summarize_pair_rows(pairs)
    pair_quality = summarize_pair_quality(pairs, group_summary=group_summary)
    quality_thresholds = quality_thresholds_by_name(args.quality_gate)
    quality_gate = None
    if quality_thresholds is not None:
        quality_gate = build_pair_quality_gate_report(
            pair_quality,
            thresholds=quality_thresholds,
        )
    data_provenance = data_provenance_for_sources(args.sources)
    manifest = {
        "created_at": datetime.now(UTC).isoformat(),
        "model_checkpoint": str(args.model_checkpoint),
        "model_checkpoint_fingerprint": _fingerprint_path(args.model_checkpoint),
        "verifier": "finpost.evals.sources.REGISTRY",
        "sources": args.sources,
        **data_provenance,
        "source_counts": source_counts,
        "heldout_train_n": args.heldout_train_n,
        "shard_id": args.shard_id,
        "num_shards": args.num_shards,
        "sharded_prompt_count": len(examples),
        "samples_per_prompt": args.samples_per_prompt,
        "generation_batch_size": args.generation_batch_size,
        "max_new_tokens": args.max_new_tokens,
        "max_new_tokens_by_source": max_new_tokens_by_source,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "max_pairs_per_prompt": args.max_pairs_per_prompt,
        "allow_empty_pairs": args.allow_empty_pairs,
        "seed": args.seed,
        "dtype": args.dtype,
        "device": str(device),
        "completion_count": len(records),
        "correct_completion_count": correct_count,
        "incorrect_completion_count": len(records) - correct_count,
        "pair_count": len(pairs),
        "group_summary": group_summary,
        "pair_summary": pair_summary,
        "pair_quality": pair_quality,
        "quality_gate": quality_gate,
        "completions_path": str(completions_path),
        "pairs_path": str(pairs_path),
        "pair_review_sample_path": str(review_sample_path),
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    if quality_gate is not None and not quality_gate["passed"]:
        failures = "\n".join(f"  - {failure}" for failure in quality_gate["failures"])
        raise ValueError(f"DPO pair quality gate failed:\n{failures}")
    print(f"[pairs] wrote {len(records)} completions to {completions_path}")
    print(f"[pairs] wrote {len(pairs)} preference pairs to {pairs_path}")
    print(f"[pairs] wrote manifest to {manifest_path}")


if __name__ == "__main__":
    main()
