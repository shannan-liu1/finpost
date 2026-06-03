"""Run the minimum FinChain GPU readiness checks and emit a JSON report."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

DEFAULT_OUT = Path("artifacts/preflight/preflight_report.json")


def _command_specs() -> list[tuple[str, list[str]]]:
    python = sys.executable
    return [
        (
            "public_smoke_pytest",
            [
                python,
                "-m",
                "pytest",
                "tests",
                "-q",
            ],
        ),
        ("ruff", [python, "-m", "ruff", "check", "."]),
        ("git_diff_check", ["git", "diff", "--check"]),
        (
            "dataset_manifest_audit",
            [
                python,
                "scripts/audit_finchain_template_disjoint_manifest.py",
                "--out",
                "artifacts/preflight/finchain_v2_manifest_audit.json",
            ],
        ),
        (
            "eval_exact_help",
            [python, "-m", "finpost.evals.eval_exact", "--help"],
        ),
        (
            "eval_validation_dry_run",
            [
                python,
                "-m",
                "finpost.evals.eval_exact",
                "--checkpoints",
                "base=Qwen/Qwen2.5-1.5B",
                "--sources",
                "finchain",
                "--finchain-split",
                "validation",
                "--n",
                "1",
                "--out-dir",
                "artifacts/preflight/eval_validation_dry_run",
                "--device",
                "cpu",
                "--dry-run",
            ],
        ),
        (
            "eval_test_dry_run",
            [
                python,
                "-m",
                "finpost.evals.eval_exact",
                "--checkpoints",
                "base=Qwen/Qwen2.5-1.5B",
                "--sources",
                "finchain",
                "--finchain-split",
                "test",
                "--n",
                "1",
                "--out-dir",
                "artifacts/preflight/eval_test_dry_run",
                "--device",
                "cpu",
                "--dry-run",
            ],
        ),
        ("eval_parallel_help", [python, "scripts/run_finchain_eval_parallel.py", "--help"]),
        ("dpo_help", [python, "scripts/train_finchain_trl_dpo.py", "--help"]),
        (
            "dpo_trl_config_smoke",
            [
                python,
                "-c",
                (
                    "import argparse; "
                    "from trl import DPOConfig; "
                    "from scripts.train_finchain_trl_dpo import _config_kwargs; "
                    "args=argparse.Namespace("
                    "output_dir='artifacts/preflight/dpo_config_smoke', "
                    "max_steps=1, learning_rate=5e-6, "
                    "per_device_train_batch_size=2, "
                    "gradient_accumulation_steps=8, beta=0.1, "
                    "max_length=1536, max_prompt_length=768, "
                    "save_steps=50, logging_steps=5, report_to='none', "
                    "run_name='dpo-config-smoke', seed=42); "
                    "DPOConfig(**_config_kwargs(args, config_cls=DPOConfig)); "
                    "print('DPOConfig OK')"
                ),
            ],
        ),
        ("grpo_help", [python, "scripts/train_finchain_trl_grpo.py", "--help"]),
        (
            "grpo_trl_config_smoke",
            [
                python,
                "-c",
                (
                    "import argparse; "
                    "from trl import GRPOConfig; "
                    "from scripts.train_finchain_trl_grpo import _config_kwargs; "
                    "args=argparse.Namespace("
                    "output_dir='artifacts/preflight/grpo_config_smoke', "
                    "max_steps=1, learning_rate=5e-7, "
                    "per_device_train_batch_size=1, "
                    "gradient_accumulation_steps=2, num_generations=2, "
                    "max_completion_length=64, temperature=0.8, top_p=0.95, "
                    "beta=0.02, save_steps=50, logging_steps=5, "
                    "report_to='none', run_name='grpo-config-smoke', "
                    "use_vllm=False, vllm_gpu_memory_utilization=0.3, seed=42); "
                    "GRPOConfig(**_config_kwargs(args, config_cls=GRPOConfig)); "
                    "print('GRPOConfig OK')"
                ),
            ],
        ),
        ("gkd_help", [python, "-m", "finpost.training.gkd_train", "--help"]),
        (
            "gkd_config_smoke",
            [
                python,
                "-c",
                (
                    "from finpost.training.gkd_train import GKDConfig; "
                    "GKDConfig.model_validate({"
                    "'model': {'student_checkpoint': "
                    "'results/checkpoints/qwen25-1p5b-finchain-v2-trl-sft-2gpu/selected', "
                    "'teacher_checkpoint': 'Qwen/Qwen2.5-7B-Instruct'}, "
                    "'training': {'max_steps': 2, 'warmup_steps': 0, 'lr': 5e-6}, "
                    "'checkpointing': {'save_dir': 'artifacts/preflight/gkd_config_smoke'}"
                    "}); "
                    "print('GKDConfig OK')"
                ),
            ],
        ),
        (
            "notebook_smoke",
            [
                python,
                "scripts/smoke_finchain_notebooks.py",
                "--out",
                "artifacts/preflight/notebook_smoke.json",
            ],
        ),
    ]


def _run_one(name: str, command: list[str], timeout_sec: int) -> dict[str, Any]:
    start = time.perf_counter()
    started_at = datetime.now(UTC).isoformat()
    proc = subprocess.run(
        command,
        text=True,
        capture_output=True,
        timeout=timeout_sec,
        check=False,
    )
    ended_at = datetime.now(UTC).isoformat()
    return {
        "name": name,
        "command": command,
        "returncode": proc.returncode,
        "ok": proc.returncode == 0,
        "started_at": started_at,
        "ended_at": ended_at,
        "elapsed_sec": round(time.perf_counter() - start, 3),
        "stdout_tail": proc.stdout[-4000:],
        "stderr_tail": proc.stderr[-4000:],
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--timeout-sec", type=int, default=600)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    results = [
        _run_one(name, command, args.timeout_sec)
        for name, command in _command_specs()
    ]
    report = {
        "ok": all(result["ok"] for result in results),
        "generated_at": datetime.now(UTC).isoformat(),
        "results": results,
    }
    args.out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    if not report["ok"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
