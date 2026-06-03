"""Train FinChain GRPO with Hugging Face TRL.

This is the TRL trainer path for GPU runs. It deliberately reuses the
repo's FinChain loader and verifier reward, while delegating distributed
training, rollout generation, and optimizer plumbing to TRL/Accelerate.

Single GPU:

    python scripts/train_finchain_trl_grpo.py \
      --model Qwen/Qwen2.5-1.5B \
      --train-n 512 \
      --per-device-train-batch-size 4 \
      --output-dir results/checkpoints/qwen25-1p5b-finchain-v2-grpo-canary

Two GPUs:

    accelerate launch --num_processes 2 scripts/train_finchain_trl_grpo.py \
      --model Qwen/Qwen2.5-1.5B \
      --train-n 2320 \
      --per-device-train-batch-size 2 \
      --output-dir results/checkpoints/qwen25-1p5b-finchain-v2-grpo
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any

from finpost.training.finchain_rlvr import (
    build_finchain_prompt_rows,
    finchain_binary_rewards,
)
from finpost.safety import safe_load_model
from scripts.finchain_trl_common import (
    add_finchain_sampling_args,
    filter_trl_config_kwargs,
    load_tokenizer_with_padding,
    sampled_finchain_dataset,
    trl_training_config,
)

OPTIONAL_GRPO_CONFIG_FIELDS = {
    "generation_kwargs",
    "use_vllm",
    "vllm_gpu_memory_utilization",
}

STABLE_GENERATION_KWARGS = {
    # GRPO samples during training. On some Qwen/TRL/Transformers stacks the
    # first CUDA rollout can produce NaN/Inf logits and crash multinomial.
    "remove_invalid_values": True,
    "renormalize_logits": True,
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="Qwen/Qwen2.5-1.5B")
    add_finchain_sampling_args(parser, default_train_n=512)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-steps", type=int, default=100)
    parser.add_argument("--learning-rate", type=float, default=5.0e-7)
    parser.add_argument("--per-device-train-batch-size", type=int, default=4)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--num-generations", type=int, default=4)
    parser.add_argument("--max-completion-length", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--beta", type=float, default=0.02)
    parser.add_argument("--save-steps", type=int, default=50)
    parser.add_argument("--logging-steps", type=int, default=5)
    parser.add_argument("--report-to", default="none")
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--use-vllm", action="store_true")
    parser.add_argument("--vllm-gpu-memory-utilization", type=float, default=0.3)
    return parser.parse_args()


def _world_size_from_env() -> int:
    raw = os.environ.get("WORLD_SIZE", "1")
    try:
        world_size = int(raw)
    except ValueError as exc:
        raise SystemExit(f"WORLD_SIZE must be an integer; got {raw!r}") from exc
    if world_size <= 0:
        raise SystemExit(f"WORLD_SIZE must be positive; got {world_size}")
    return world_size


def _validate_runtime_args(args: argparse.Namespace, *, world_size: int | None = None) -> None:
    """Fail before model load for TRL GRPO grouped-rollout constraints."""
    if args.num_generations < 2:
        raise SystemExit("GRPO requires --num-generations >= 2")
    if args.per_device_train_batch_size <= 0:
        raise SystemExit("GRPO requires --per-device-train-batch-size to be positive")
    resolved_world_size = _world_size_from_env() if world_size is None else world_size
    if resolved_world_size <= 0:
        raise SystemExit(f"WORLD_SIZE must be positive; got {resolved_world_size}")
    global_prompt_batch = args.per_device_train_batch_size * resolved_world_size
    if global_prompt_batch % args.num_generations != 0:
        raise SystemExit(
            "TRL GRPO requires the global prompt batch "
            "(per_device_train_batch_size * world_size) to be divisible by "
            "num_generations. Got "
            f"{args.per_device_train_batch_size} * {resolved_world_size} = "
            f"{global_prompt_batch}, num_generations={args.num_generations}."
        )


def _align_model_with_tokenizer(model: Any, tokenizer: Any) -> None:
    """Keep explicit GRPO model/tokenizer loading safe for generation."""
    tokenizer.padding_side = "left"

    embedding_count = model.get_input_embeddings().num_embeddings
    token_count = len(tokenizer)
    if token_count > embedding_count:
        model.resize_token_embeddings(token_count)

    for token_attr in ("pad_token_id", "eos_token_id", "bos_token_id"):
        token_id = getattr(tokenizer, token_attr, None)
        if token_id is None:
            continue
        setattr(model.config, token_attr, token_id)
        generation_config = getattr(model, "generation_config", None)
        if generation_config is not None:
            setattr(generation_config, token_attr, token_id)


def _config_kwargs(
    args: argparse.Namespace,
    config_cls: type[Any] | None = None,
) -> dict[str, Any]:
    kwargs = trl_training_config(
        args,
        num_generations=args.num_generations,
        max_completion_length=args.max_completion_length,
        temperature=args.temperature,
        top_p=args.top_p,
        beta=args.beta,
        generation_kwargs=STABLE_GENERATION_KWARGS,
        use_vllm=args.use_vllm,
        vllm_gpu_memory_utilization=args.vllm_gpu_memory_utilization,
    )
    if config_cls is None:
        return kwargs
    return filter_trl_config_kwargs(
        kwargs,
        config_cls,
        optional_fields=OPTIONAL_GRPO_CONFIG_FIELDS,
    )


def main() -> None:
    args = _parse_args()
    _validate_runtime_args(args)
    try:
        from trl import GRPOConfig, GRPOTrainer
    except ImportError as exc:  # pragma: no cover - depends on optional RLVR extra
        raise SystemExit(
            "TRL is required for this script. In a GPU environment run: "
            'pip install -e ".[dev,rlvr]"'
        ) from exc

    import torch

    dataset = sampled_finchain_dataset(args, build_finchain_prompt_rows)
    tokenizer = load_tokenizer_with_padding(args.model)
    model = safe_load_model(
        args.model,
        torch_dtype=torch.bfloat16,
        attn_implementation="eager",
    )
    _align_model_with_tokenizer(model, tokenizer)
    training_args = GRPOConfig(**_config_kwargs(args, config_cls=GRPOConfig))
    trainer = GRPOTrainer(
        model=model,
        reward_funcs=finchain_binary_rewards,
        args=training_args,
        train_dataset=dataset,
        processing_class=tokenizer,
    )
    trainer.train()
    trainer.save_model(str(args.output_dir / "final"))


if __name__ == "__main__":
    main()
