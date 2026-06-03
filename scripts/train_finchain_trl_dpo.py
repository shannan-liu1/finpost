"""Train FinChain offline DPO with Hugging Face TRL.

This is the distributed TRL path for the same fixed preference-pair
JSONL consumed by ``finpost.training.dpo_train``. Use the repo-native trainer
when you want to inspect the DPO math and cached reference-logp mechanics. Use
this script when you want Accelerate/TRL to handle multi-GPU launch, mixed
precision, and checkpoint plumbing.

``--model`` and optional ``--ref-model`` accept any compatible HF-format
causal LM checkpoint. The public FinChain run uses the validation-selected SFT
checkpoint for both policy initialization and reference model to keep DPO a
fair continuation baseline.

Single GPU:

    python scripts/train_finchain_trl_dpo.py \
      --pairs-path results/finchain_pairs/v2_template_disjoint/merged/pairs.jsonl \
      --model results/checkpoints/qwen25-1p5b-finchain-v2-trl-sft-2gpu/selected \
      --output-dir results/checkpoints/qwen25-1p5b-finchain-v2-dpo-canary

Two GPUs:

    accelerate launch --num_processes 2 scripts/train_finchain_trl_dpo.py \
      --pairs-path results/finchain_pairs/v2_template_disjoint/merged/pairs.jsonl \
      --model results/checkpoints/qwen25-1p5b-finchain-v2-trl-sft-2gpu/selected \
      --output-dir results/checkpoints/qwen25-1p5b-finchain-v2-dpo
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from datasets import Dataset

from finpost.safety import safe_load_model
from finpost.training.dataset import serialize_prompt, serialize_response
from finpost.training.preference_data import load_preference_pairs
from scripts.finchain_trl_common import (
    filter_trl_config_kwargs,
    load_tokenizer_with_padding,
    trl_training_config,
)

OPTIONAL_DPO_CONFIG_FIELDS = {"max_prompt_length"}


def preference_pairs_to_trl_rows(path: str | Path) -> list[dict[str, Any]]:
    """Convert repo-native DPO JSONL into TRL DPOTrainer rows."""
    rows: list[dict[str, Any]] = []
    for example in load_preference_pairs(path):
        rows.append(
            {
                "prompt": serialize_prompt(example.prompt),
                "chosen": serialize_response(example.chosen),
                "rejected": serialize_response(example.rejected),
                "source": example.source,
                "prompt_id": example.prompt_id,
            }
        )
    return rows


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pairs-path", type=Path, required=True)
    parser.add_argument(
        "--model",
        default="results/checkpoints/qwen25-1p5b-finchain-v2-trl-sft-2gpu/selected",
    )
    parser.add_argument("--ref-model", default=None)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-steps", type=int, default=100)
    parser.add_argument("--learning-rate", type=float, default=5.0e-6)
    parser.add_argument("--per-device-train-batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--max-length", type=int, default=1536)
    parser.add_argument("--max-prompt-length", type=int, default=768)
    parser.add_argument("--save-steps", type=int, default=50)
    parser.add_argument("--logging-steps", type=int, default=5)
    parser.add_argument("--report-to", default="none")
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def _config_kwargs(
    args: argparse.Namespace,
    config_cls: type[Any] | None = None,
) -> dict[str, Any]:
    kwargs = trl_training_config(
        args,
        beta=args.beta,
        max_length=args.max_length,
        max_prompt_length=args.max_prompt_length,
    )
    if config_cls is None:
        return kwargs
    return filter_trl_config_kwargs(
        kwargs,
        config_cls,
        optional_fields=OPTIONAL_DPO_CONFIG_FIELDS,
    )


def _trainer_ref_model(args: argparse.Namespace) -> Any | None:
    """Return the TRL ref_model argument, loading only explicit different refs."""
    if args.ref_model is None or args.ref_model == args.model:
        return None

    import torch

    return safe_load_model(
        args.ref_model,
        torch_dtype=torch.bfloat16,
    )


def main() -> None:
    args = _parse_args()
    try:
        from trl import DPOConfig, DPOTrainer
    except ImportError as exc:  # pragma: no cover - depends on optional RLVR extra
        raise SystemExit(
            "TRL DPO is required for this script. In a GPU environment run: "
            'pip install -e ".[dev,rlvr]"'
        ) from exc

    dataset = Dataset.from_list(preference_pairs_to_trl_rows(args.pairs_path))
    tokenizer = load_tokenizer_with_padding(args.model)

    trainer = DPOTrainer(
        model=args.model,
        ref_model=_trainer_ref_model(args),
        args=DPOConfig(**_config_kwargs(args, config_cls=DPOConfig)),
        processing_class=tokenizer,
        train_dataset=dataset,
    )
    trainer.train()
    trainer.save_model(str(args.output_dir / "final"))


if __name__ == "__main__":
    main()
