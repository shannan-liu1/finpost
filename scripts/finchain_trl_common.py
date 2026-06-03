"""Shared helpers for FinChain TRL adapter scripts."""

from __future__ import annotations

import argparse
import inspect
from collections.abc import Callable
from typing import Any

from datasets import Dataset

from finpost.data.finchain_dataset import load_finchain
from finpost.training.finchain_rlvr import deterministic_sample
from finpost.safety import safe_load_tokenizer

FINCHAIN_SPLITS = ("train", "validation", "test")


def add_finchain_sampling_args(parser: argparse.ArgumentParser, *, default_train_n: int) -> None:
    parser.add_argument("--split", choices=FINCHAIN_SPLITS, default="train")
    parser.add_argument("--train-n", type=int, default=default_train_n)
    parser.add_argument("--seed", type=int, default=42)


def sampled_finchain_dataset(
    args: argparse.Namespace,
    row_builder: Callable[..., list[dict[str, Any]]],
    **row_builder_kwargs: Any,
) -> Dataset:
    examples = deterministic_sample(
        load_finchain(args.split),
        n=args.train_n,
        seed=args.seed,
    )
    return Dataset.from_list(row_builder(examples, **row_builder_kwargs))


def trl_training_config(args: argparse.Namespace, **extra: Any) -> dict[str, Any]:
    config: dict[str, Any] = {
        "output_dir": str(args.output_dir),
        "max_steps": args.max_steps,
        "learning_rate": args.learning_rate,
        "per_device_train_batch_size": args.per_device_train_batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
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
    config.update(extra)
    return config


def filter_trl_config_kwargs(
    kwargs: dict[str, Any],
    config_cls: type[Any],
    *,
    optional_fields: set[str],
) -> dict[str, Any]:
    """Drop known-optional TRL config kwargs when the installed TRL rejects them."""
    signature = inspect.signature(config_cls.__init__)
    parameters = signature.parameters
    if any(param.kind is inspect.Parameter.VAR_KEYWORD for param in parameters.values()):
        return kwargs

    accepted = set(parameters)
    unsupported = set(kwargs) - accepted
    non_optional = unsupported - optional_fields
    if non_optional:
        raise TypeError(
            f"Installed {config_cls.__name__} does not accept required kwargs: "
            f"{sorted(non_optional)}. Accepted kwargs include: {sorted(accepted)}"
        )
    return {
        key: value
        for key, value in kwargs.items()
        if key not in unsupported
    }


def load_tokenizer_with_padding(model: str):
    """Load a tokenizer and make padding explicit for TRL trainers."""
    tokenizer = safe_load_tokenizer(model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer
