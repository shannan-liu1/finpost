"""Run FinChain evals in parallel across GPUs.

The underlying eval CLI is intentionally single-process and deterministic: one
checkpoint, one device, one output directory. This wrapper keeps that contract
and launches independent eval processes in one of two modes:

- ``--parallelism checkpoints`` assigns checkpoint runs to GPU ids round-robin.
- ``--parallelism examples`` evaluates one checkpoint at a time, splitting the
  seeded example sample across GPU subprocesses and merging artifacts.

Example:

    python scripts/run_finchain_eval_parallel.py \
      --gpus 0 1 \
      --parallelism examples \
      --finchain-split test \
      --checkpoints \
        sft=results/checkpoints/qwen25-1p5b-finchain-v2-trl-sft-2gpu/selected \
        dpo=results/checkpoints/qwen25-1p5b-finchain-v2-dpo/selected \
      --n 870 \
      --out-dir results/evals/finchain_v2_template_disjoint/test_selected_chaineval
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

from finpost.evals.eval_exact import _parse_checkpoint_pair


@dataclass(frozen=True)
class EvalShard:
    name: str
    checkpoint_path: str
    gpu: str
    out_dir: Path
    sample_shard_id: int | None = None
    num_sample_shards: int | None = None


_SUMMARY_BASE_FIELDS = [
    "checkpoint",
    "source",
    "n",
    "accuracy",
    "parse_success_rate",
    "generated_tokens",
    "generated_tokens_decoded",
    "elapsed_sec",
]


def assign_eval_shards(
    *,
    checkpoint_pairs: list[str],
    gpus: list[str],
    out_dir: Path,
) -> list[EvalShard]:
    """Assign checkpoint eval jobs to GPU ids round-robin."""
    if not gpus:
        raise ValueError("at least one GPU id is required")
    shards: list[EvalShard] = []
    seen_names: set[str] = set()
    for idx, pair in enumerate(checkpoint_pairs):
        name, checkpoint_path = _parse_checkpoint_pair(pair)
        if name in seen_names:
            raise ValueError(f"duplicate checkpoint name: {name}")
        seen_names.add(name)
        shards.append(
            EvalShard(
                name=name,
                checkpoint_path=checkpoint_path,
                gpu=gpus[idx % len(gpus)],
                out_dir=out_dir / name,
            )
        )
    return shards


def build_eval_command(
    shard: EvalShard,
    *,
    n: int,
    seed: int,
    finchain_split: str,
    batch_size_finchain: int,
    gpu_cost_per_hour: float | None,
    enable_chaineval: bool = False,
    chaineval_batch_size: int = 8,
) -> list[str]:
    """Build the subprocess argv for one checkpoint eval.

    ``enable_chaineval`` and ``chaineval_batch_size`` are forwarded to
    each spawned eval process. Each subprocess inherits the flag via
    argv, so the underlying CLI's own ChainEvalNotAvailable check fires
    in the subprocess if the extra is not installed.
    """
    command = [
        sys.executable,
        "scripts/run_finchain_eval.py",
        "--checkpoints",
        f"{shard.name}={shard.checkpoint_path}",
        "--n",
        str(n),
        "--seed",
        str(seed),
        "--finchain-split",
        finchain_split,
        "--out-dir",
        str(shard.out_dir),
        "--device",
        "cuda",
        "--batch-size-finchain",
        str(batch_size_finchain),
    ]
    if gpu_cost_per_hour is not None:
        command.extend(["--gpu-cost-per-hour", str(gpu_cost_per_hour)])
    if enable_chaineval:
        command.extend(
            [
                "--enable-chaineval",
                "--chaineval-batch-size",
                str(chaineval_batch_size),
            ]
        )
    if shard.sample_shard_id is not None and shard.num_sample_shards is not None:
        command.extend(
            [
                "--sample-shard-id",
                str(shard.sample_shard_id),
                "--num-sample-shards",
                str(shard.num_sample_shards),
            ]
        )
    return command


def _write_summary_rows(rows: list[dict[str, object]], out_dir: Path) -> None:
    extra_fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in _SUMMARY_BASE_FIELDS and key not in extra_fields:
                extra_fields.append(key)
    fieldnames = [*_SUMMARY_BASE_FIELDS, *extra_fields]
    with (out_dir / "accuracy_summary.json").open("w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2)
    with (out_dir / "accuracy_summary.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _merge_summary_rows(shard_dirs: list[Path]) -> list[dict[str, object]]:
    grouped: dict[tuple[str, str], list[dict[str, object]]] = {}
    order: list[tuple[str, str]] = []
    for shard_dir in shard_dirs:
        rows = json.loads((shard_dir / "accuracy_summary.json").read_text(encoding="utf-8"))
        for row in rows:
            key = (str(row["checkpoint"]), str(row["source"]))
            if key not in grouped:
                order.append(key)
                grouped[key] = []
            grouped[key].append(row)

    merged_rows: list[dict[str, object]] = []
    for key in order:
        rows = grouped[key]
        total_n = sum(int(row["n"]) for row in rows)
        merged: dict[str, object] = {
            "checkpoint": key[0],
            "source": key[1],
            "n": total_n,
            "accuracy": (
                sum(float(row["accuracy"]) * int(row["n"]) for row in rows) / total_n
                if total_n
                else 0.0
            ),
            "parse_success_rate": (
                sum(float(row["parse_success_rate"]) * int(row["n"]) for row in rows)
                / total_n
                if total_n
                else 0.0
            ),
            "generated_tokens": sum(int(row["generated_tokens"]) for row in rows),
            "generated_tokens_decoded": sum(
                int(row["generated_tokens_decoded"]) for row in rows
            ),
            "elapsed_sec": max(float(row["elapsed_sec"]) for row in rows),
        }
        extra_fields = [
            field
            for field in rows[0]
            if field not in _SUMMARY_BASE_FIELDS
        ]
        for field in extra_fields:
            merged[field] = (
                sum(float(row[field]) * int(row["n"]) for row in rows) / total_n
                if total_n
                else 0.0
            )
        merged_rows.append(merged)
    return merged_rows


def _read_csv_rows(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return list(reader.fieldnames or []), list(reader)


def _interleave_detail_rows(shard_rows: list[list[dict[str, str]]]) -> list[dict[str, str]]:
    merged: list[dict[str, str]] = []
    max_len = max((len(rows) for rows in shard_rows), default=0)
    for row_idx in range(max_len):
        for rows in shard_rows:
            if row_idx < len(rows):
                merged.append(rows[row_idx])
    return merged


def _merge_details(checkpoint_name: str, shard_dirs: list[Path], out_dir: Path) -> None:
    detail_paths = sorted(shard_dirs[0].glob(f"details_{checkpoint_name}_*.csv"))
    for first_path in detail_paths:
        fieldnames, first_rows = _read_csv_rows(first_path)
        shard_rows = [first_rows]
        for shard_dir in shard_dirs[1:]:
            path = shard_dir / first_path.name
            next_fieldnames, rows = _read_csv_rows(path)
            if next_fieldnames != fieldnames:
                raise ValueError(f"detail schema mismatch in {path}")
            shard_rows.append(rows)

        with (out_dir / first_path.name).open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(_interleave_detail_rows(shard_rows))


def _merge_cost_summary(shard_dirs: list[Path], out_dir: Path) -> None:
    costs = [
        json.loads((shard_dir / "cost_summary.json").read_text(encoding="utf-8"))
        for shard_dir in shard_dirs
    ]
    generated_tokens = sum(int(cost["generated_tokens"]) for cost in costs)
    generated_tokens_decoded = sum(
        int(cost["generated_tokens_decoded"]) for cost in costs
    )
    generation_seconds = max(float(cost["generation_seconds"]) for cost in costs)
    estimated_values = [cost.get("estimated_cost_usd") for cost in costs]
    merged = {
        "run_name": out_dir.name,
        "start_time": min(str(cost["start_time"]) for cost in costs),
        "end_time": max(str(cost["end_time"]) for cost in costs),
        "elapsed_sec": max(float(cost["elapsed_sec"]) for cost in costs),
        "generation_seconds": generation_seconds,
        "gpu_type": ", ".join(sorted({str(cost["gpu_type"]) for cost in costs})),
        "dtype": costs[0]["dtype"],
        "generated_tokens": generated_tokens,
        "generated_tokens_decoded": generated_tokens_decoded,
        "tokens_per_second": (
            generated_tokens / generation_seconds if generation_seconds > 0 else 0.0
        ),
        "estimated_cost_usd": (
            sum(float(value) for value in estimated_values)
            if all(value is not None for value in estimated_values)
            else None
        ),
    }
    with (out_dir / "cost_summary.json").open("w", encoding="utf-8") as f:
        json.dump(merged, f, indent=2)


def _merge_run_metadata(
    *,
    shard_dirs: list[Path],
    out_dir: Path,
    sample_shard_count: int,
) -> None:
    metadata = json.loads(
        (shard_dirs[0] / "run_metadata.json").read_text(encoding="utf-8")
    )
    metadata["sample_sharding"] = {
        "num_sample_shards": sample_shard_count,
        "strategy": "round_robin_after_seeded_sample",
        "shard_dirs": [str(path) for path in shard_dirs],
    }
    with (out_dir / "run_metadata.json").open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)


def merge_example_shard_outputs(
    *,
    checkpoint_name: str,
    shard_dirs: list[Path],
    out_dir: Path,
    sample_shard_count: int,
) -> None:
    """Merge example-sharded subprocess artifacts into one checkpoint eval."""
    if not shard_dirs:
        raise ValueError("at least one shard directory is required")
    out_dir.mkdir(parents=True, exist_ok=True)
    _write_summary_rows(_merge_summary_rows(shard_dirs), out_dir)
    _merge_details(checkpoint_name, shard_dirs, out_dir)
    _merge_cost_summary(shard_dirs, out_dir)
    _merge_run_metadata(
        shard_dirs=shard_dirs,
        out_dir=out_dir,
        sample_shard_count=sample_shard_count,
    )
    manifest = {
        "checkpoint": checkpoint_name,
        "num_sample_shards": sample_shard_count,
        "shards": [str(path) for path in shard_dirs],
    }
    with (out_dir / "shard_manifest.json").open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoints", nargs="+", required=True, metavar="NAME=PATH")
    parser.add_argument("--gpus", nargs="+", required=True, help="GPU ids, e.g. 0 1")
    parser.add_argument(
        "--parallelism",
        choices=["checkpoints", "examples"],
        default="checkpoints",
        help=(
            "checkpoints: assign checkpoint evals round-robin across GPUs. "
            "examples: split each checkpoint's sampled examples across GPUs "
            "and merge artifacts."
        ),
    )
    parser.add_argument("--n", type=int, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--finchain-split",
        choices=["validation", "test"],
        default="test",
        help="Use validation for checkpoint selection and test only for final reporting.",
    )
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--batch-size-finchain", type=int, default=4)
    parser.add_argument("--gpu-cost-per-hour", type=float, default=None)
    parser.add_argument(
        "--enable-chaineval",
        action="store_true",
        help=(
            "Forward --enable-chaineval to every spawned eval subprocess. "
            'Requires the [chaineval] extra in each subprocess environment.'
        ),
    )
    parser.add_argument(
        "--chaineval-batch-size",
        type=int,
        default=8,
        help="Forwarded to every spawned eval subprocess. Default: %(default)s.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    if args.parallelism == "checkpoints":
        shards = assign_eval_shards(
            checkpoint_pairs=args.checkpoints,
            gpus=[str(gpu) for gpu in args.gpus],
            out_dir=args.out_dir,
        )

        processes: list[tuple[EvalShard, subprocess.Popen]] = []
        for shard in shards:
            shard.out_dir.mkdir(parents=True, exist_ok=True)
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = shard.gpu
            command = build_eval_command(
                shard,
                n=args.n,
                seed=args.seed,
                finchain_split=args.finchain_split,
                batch_size_finchain=args.batch_size_finchain,
                gpu_cost_per_hour=args.gpu_cost_per_hour,
                enable_chaineval=args.enable_chaineval,
                chaineval_batch_size=args.chaineval_batch_size,
            )
            print(f"[parallel-eval] GPU {shard.gpu}: {' '.join(command)}")
            processes.append((shard, subprocess.Popen(command, env=env)))

        failures: list[tuple[str, int]] = []
        for shard, process in processes:
            code = process.wait()
            if code != 0:
                failures.append((shard.name, code))

        if failures:
            for name, code in failures:
                print(f"[parallel-eval] {name} failed with exit {code}", file=sys.stderr)
            raise SystemExit(1)
    else:
        gpus = [str(gpu) for gpu in args.gpus]
        for pair in args.checkpoints:
            name, checkpoint_path = _parse_checkpoint_pair(pair)
            checkpoint_out_dir = args.out_dir / name
            shard_root = checkpoint_out_dir / "_shards"
            shards = [
                EvalShard(
                    name=name,
                    checkpoint_path=checkpoint_path,
                    gpu=gpu,
                    out_dir=shard_root / f"shard-{idx:02d}-of-{len(gpus):02d}",
                    sample_shard_id=idx,
                    num_sample_shards=len(gpus),
                )
                for idx, gpu in enumerate(gpus)
            ]

            processes: list[tuple[EvalShard, subprocess.Popen]] = []
            for shard in shards:
                shard.out_dir.mkdir(parents=True, exist_ok=True)
                env = os.environ.copy()
                env["CUDA_VISIBLE_DEVICES"] = shard.gpu
                command = build_eval_command(
                    shard,
                    n=args.n,
                    seed=args.seed,
                    finchain_split=args.finchain_split,
                    batch_size_finchain=args.batch_size_finchain,
                    gpu_cost_per_hour=args.gpu_cost_per_hour,
                    enable_chaineval=args.enable_chaineval,
                    chaineval_batch_size=args.chaineval_batch_size,
                )
                print(
                    f"[parallel-eval] {name} shard "
                    f"{shard.sample_shard_id}/{shard.num_sample_shards} "
                    f"on GPU {shard.gpu}: {' '.join(command)}"
                )
                processes.append((shard, subprocess.Popen(command, env=env)))

            failures: list[tuple[str, int]] = []
            for shard, process in processes:
                code = process.wait()
                if code != 0:
                    failures.append((f"{shard.name}:shard-{shard.sample_shard_id}", code))

            if failures:
                for label, code in failures:
                    print(f"[parallel-eval] {label} failed with exit {code}", file=sys.stderr)
                raise SystemExit(1)

            merge_example_shard_outputs(
                checkpoint_name=name,
                shard_dirs=[shard.out_dir for shard in shards],
                out_dir=checkpoint_out_dir,
                sample_shard_count=len(gpus),
            )

    print(f"[parallel-eval] wrote per-checkpoint evals under {args.out_dir}")


if __name__ == "__main__":
    main()
