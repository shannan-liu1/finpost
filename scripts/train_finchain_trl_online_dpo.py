"""Train FinChain online DPO / OPD with Hugging Face TRL.

This is the teacher/student-style OPD path: the trainer samples current-policy
responses, scores them with the repo's deterministic FinChain verifier reward,
and updates the student online. It is intentionally separate from the offline
DPO notebook, which trains on fixed pre-generated pairs.

Single GPU:

    python scripts/train_finchain_trl_online_dpo.py \
      --model Qwen/Qwen2.5-1.5B \
      --train-n 512 \
      --output-dir results/checkpoints/qwen25-1p5b-finchain-online-dpo-canary

Two GPUs:

    accelerate launch --num_processes 2 scripts/train_finchain_trl_online_dpo.py \
      --model Qwen/Qwen2.5-1.5B \
      --train-n 2000 \
      --output-dir results/checkpoints/qwen25-1p5b-finchain-online-dpo-2gpu
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import torch
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from finpost.data.finchain import load_finchain
from finpost.posttraining.finchain_rlvr import (
    build_finchain_prompt_rows,
    deterministic_sample,
    finchain_binary_rewards,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="Qwen/Qwen2.5-1.5B")
    parser.add_argument("--split", choices=["train", "validation", "test"], default="train")
    parser.add_argument("--train-n", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-steps", type=int, default=100)
    parser.add_argument("--learning-rate", type=float, default=5.0e-7)
    parser.add_argument("--per-device-train-batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--missing-eos-penalty", type=float, default=1.0)
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--save-steps", type=int, default=50)
    parser.add_argument("--logging-steps", type=int, default=5)
    parser.add_argument("--report-to", default="none")
    parser.add_argument("--run-name", default=None)
    return parser.parse_args()


def _dataset_from_args(args: argparse.Namespace) -> Dataset:
    examples = deterministic_sample(
        load_finchain(args.split),
        n=args.train_n,
        seed=args.seed,
    )
    return Dataset.from_list(build_finchain_prompt_rows(examples))


def _config_kwargs(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "output_dir": str(args.output_dir),
        "max_steps": args.max_steps,
        "learning_rate": args.learning_rate,
        "per_device_train_batch_size": args.per_device_train_batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "max_new_tokens": args.max_new_tokens,
        "temperature": args.temperature,
        "missing_eos_penalty": args.missing_eos_penalty,
        "beta": args.beta,
        "bf16": True,
        "tf32": True,
        "gradient_checkpointing": True,
        "logging_steps": args.logging_steps,
        "save_steps": args.save_steps,
        "save_total_limit": 3,
        "report_to": args.report_to,
        "run_name": args.run_name,
        "remove_unused_columns": False,
        "seed": args.seed,
    }


def main() -> None:
    args = _parse_args()
    try:
        from trl.experimental.online_dpo import OnlineDPOConfig, OnlineDPOTrainer
    except ImportError as exc:  # pragma: no cover - depends on optional RunPod extra
        raise SystemExit(
            "TRL online DPO is required for this script. On RunPod run: "
            'pip install -e ".[dev,rlvr]"'
        ) from exc

    dataset = _dataset_from_args(args)
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=dtype)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    training_args = OnlineDPOConfig(**_config_kwargs(args))
    trainer = OnlineDPOTrainer(
        model=model,
        reward_funcs=finchain_binary_rewards,
        args=training_args,
        processing_class=tokenizer,
        train_dataset=dataset,
    )
    trainer.train()
    trainer.save_model(str(args.output_dir / "final"))


if __name__ == "__main__":
    main()
