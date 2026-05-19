"""Build offline preference pairs for Phase 1 DPO.

The script samples multiple completions from an SFT checkpoint on training
prompts, grades each completion with the same exact-answer source registry used
by eval, and writes JSONL preference pairs where a correct completion is chosen
over an incorrect completion from the same prompt.
"""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import torch
from tqdm.auto import tqdm

from finpost.data.gsm8k import load_gsm8k
from finpost.data.math_dataset import load_math
from finpost.data.schema import Example, Source
from finpost.evals.sources import REGISTRY
from finpost.safety import safe_load_model, safe_load_tokenizer
from finpost.training.dataset import serialize_prompt


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


def _load_train_examples(source: Source) -> list[Example]:
    if source == "gsm8k":
        return load_gsm8k("train")
    if source == "math":
        return load_math("train")
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
        ]
        if max_pairs_per_prompt is not None and len(prompt_pairs) > max_pairs_per_prompt:
            rng = random.Random(f"{seed}:{prompt_id}:pairs")
            rng.shuffle(prompt_pairs)
            prompt_pairs = prompt_pairs[:max_pairs_per_prompt]
        pairs.extend(prompt_pairs)
    return pairs


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
    max_new_tokens_gsm8k: int | None = None,
    max_new_tokens_math: int | None = None,
) -> dict[Source, int]:
    """Resolve per-source generation budgets from CLI overrides."""
    overrides = {
        "gsm8k": max_new_tokens_gsm8k,
        "math": max_new_tokens_math,
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
            )
        )
    return records


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as fp:
        for row in rows:
            fp.write(json.dumps(row, ensure_ascii=True) + "\n")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model-checkpoint",
        "--sft-checkpoint",
        dest="model_checkpoint",
        type=Path,
        required=True,
        help="HF-format SFT checkpoint to sample from.",
    )
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument(
        "--sources",
        nargs="+",
        choices=["gsm8k", "math"],
        default=["gsm8k", "math"],
    )
    parser.add_argument("--heldout-train-n", type=int, default=2000)
    parser.add_argument("--samples-per-prompt", type=int, default=8)
    parser.add_argument("--generation-batch-size", type=int, default=64)
    parser.add_argument("--max-new-tokens", type=int, default=768)
    parser.add_argument("--max-new-tokens-gsm8k", type=int, default=None)
    parser.add_argument("--max-new-tokens-math", type=int, default=None)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--max-pairs-per-prompt", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
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
    manifest_path = args.out_dir / "manifest.json"

    print(f"[pairs] loading train prompts from {args.sources}")
    examples = select_train_prompts(
        sources=args.sources,
        heldout_train_n=args.heldout_train_n,
        seed=args.seed,
    )
    source_counts = {
        source: sum(ex.source == source for ex in examples)
        for source in args.sources
    }
    print(f"[pairs] selected {len(examples)} prompts: {source_counts}")
    max_new_tokens_by_source = resolve_max_new_tokens_by_source(
        sources=args.sources,
        max_new_tokens=args.max_new_tokens,
        max_new_tokens_gsm8k=args.max_new_tokens_gsm8k,
        max_new_tokens_math=args.max_new_tokens_math,
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
    if not pairs:
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

    correct_count = sum(record.correct for record in records)
    group_summary = summarize_completion_groups(records)
    manifest = {
        "created_at": datetime.now(UTC).isoformat(),
        "model_checkpoint": str(args.model_checkpoint),
        "verifier": "finpost.evals.sources.REGISTRY",
        "sources": args.sources,
        "source_counts": source_counts,
        "heldout_train_n": args.heldout_train_n,
        "samples_per_prompt": args.samples_per_prompt,
        "generation_batch_size": args.generation_batch_size,
        "max_new_tokens": args.max_new_tokens,
        "max_new_tokens_by_source": max_new_tokens_by_source,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "max_pairs_per_prompt": args.max_pairs_per_prompt,
        "seed": args.seed,
        "dtype": args.dtype,
        "device": str(device),
        "completion_count": len(records),
        "correct_completion_count": correct_count,
        "incorrect_completion_count": len(records) - correct_count,
        "pair_count": len(pairs),
        "group_summary": group_summary,
        "completions_path": str(completions_path),
        "pairs_path": str(pairs_path),
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(f"[pairs] wrote {len(records)} completions to {completions_path}")
    print(f"[pairs] wrote {len(pairs)} preference pairs to {pairs_path}")
    print(f"[pairs] wrote manifest to {manifest_path}")


if __name__ == "__main__":
    main()
