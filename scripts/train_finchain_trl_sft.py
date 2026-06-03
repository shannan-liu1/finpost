"""Train FinChain SFT with Hugging Face TRL.

This is the distributed TRL path for the same FinChain examples used by
the repo-native SFT trainer. Use ``finpost.training.train`` when you want to
inspect response masking, packing, checkpointing, and optimizer plumbing from
first principles. Use this script when you want Accelerate/TRL to handle
multi-GPU launch, mixed precision, packing, and checkpoint plumbing.

Single GPU:

    python scripts/train_finchain_trl_sft.py \
      --model Qwen/Qwen2.5-1.5B \
      --train-n 512 \
      --output-dir results/checkpoints/qwen25-1p5b-finchain-v2-trl-sft-canary

Two GPUs:

    accelerate launch --num_processes 2 scripts/train_finchain_trl_sft.py \
      --model Qwen/Qwen2.5-1.5B \
      --train-n 2320 \
      --output-dir results/checkpoints/qwen25-1p5b-finchain-v2-trl-sft-2gpu
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from finpost.data.schema import Example
from finpost.training.dataset import serialize_prompt, serialize_response
from scripts.finchain_trl_common import (
    add_finchain_sampling_args,
    load_tokenizer_with_padding,
    sampled_finchain_dataset,
    trl_training_config,
)


def finchain_examples_to_sft_rows(examples: list[Example]) -> list[dict[str, Any]]:
    """Convert FinChain examples into TRL prompt-completion rows.

    TRL applies completion-only loss automatically for prompt-completion
    datasets when ``completion_only_loss=True``. Keeping prompt and completion
    separate preserves the repo's "do not train on prompt tokens" contract in
    the distributed trainer path.
    """
    return [
        {
            "prompt": serialize_prompt(example.prompt),
            "completion": serialize_response(example.response),
            "source": example.source,
            "prompt_id": example.id,
            "gold_answer": example.final_answer,
        }
        for example in examples
    ]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="Qwen/Qwen2.5-1.5B")
    add_finchain_sampling_args(parser, default_train_n=2000)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-steps", type=int, default=250)
    parser.add_argument("--learning-rate", type=float, default=2.0e-5)
    parser.add_argument("--per-device-train-batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=1536)
    parser.add_argument("--save-steps", type=int, default=50)
    parser.add_argument("--logging-steps", type=int, default=5)
    parser.add_argument("--report-to", default="none")
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--packing", action="store_true")
    parser.add_argument("--attn-implementation", default=None)
    return parser.parse_args()


def _validate_runtime_args(args: argparse.Namespace) -> None:
    """Fail before model load for GPU options with expensive failure modes."""
    if args.packing and args.attn_implementation != "flash_attention_2":
        raise SystemExit(
            "TRL SFT packing uses the padding-free/BFD path in recent TRL versions. "
            "Run without --packing, or install flash-attn and pass "
            "--attn-implementation flash_attention_2 after a small packed canary."
        )


def _config_kwargs(args: argparse.Namespace) -> dict[str, Any]:
    model_init_kwargs: dict[str, Any] = {"dtype": "bfloat16"}
    if args.attn_implementation:
        model_init_kwargs["attn_implementation"] = args.attn_implementation
    return trl_training_config(
        args,
        max_length=args.max_length,
        packing=args.packing,
        completion_only_loss=True,
        model_init_kwargs=model_init_kwargs,
    )


def main() -> None:
    args = _parse_args()
    _validate_runtime_args(args)
    try:
        from trl import SFTConfig, SFTTrainer
    except ImportError as exc:  # pragma: no cover - depends on optional RLVR extra
        raise SystemExit(
            "TRL SFT is required for this script. In a GPU environment run: "
            'pip install -e ".[dev,rlvr]"'
        ) from exc

    dataset = sampled_finchain_dataset(args, finchain_examples_to_sft_rows)
    tokenizer = load_tokenizer_with_padding(args.model)

    trainer = SFTTrainer(
        model=args.model,
        args=SFTConfig(**_config_kwargs(args)),
        processing_class=tokenizer,
        train_dataset=dataset,
    )
    trainer.train()
    trainer.save_model(str(args.output_dir / "final"))


if __name__ == "__main__":
    main()
