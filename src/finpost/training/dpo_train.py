"""Command-line DPO trainer for Phase 1 preference optimization."""

from __future__ import annotations

import argparse
import os
import random
import shutil
import time
from pathlib import Path
from typing import IO, Any, NamedTuple

import numpy as np
import safetensors.torch
import torch
import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator
from torch.utils.data import DataLoader

import wandb
from finpost.safety import safe_load_model, safe_load_tokenizer
from finpost.training._guards import check_finite_loss
from finpost.training.checkpoint import apply_retention_policy
from finpost.training.config import CheckpointConfig, DType, LoggingConfig, PackingConfig
from finpost.training.dpo import dpo_loss_from_logits
from finpost.training.masking import IGNORE_INDEX
from finpost.training.optim import build_lr_scheduler, build_optimizer
from finpost.training.preference_data import (
    DPOCollator,
    load_or_build_tokenized_preference_dataset,
)

_FROZEN_FORBID = ConfigDict(frozen=True, extra="forbid")
_STEP_PREFIX = "step-"
_STEP_DIGITS = 8
_MODEL_FILENAME = "model.safetensors"
_STATE_FILENAME = "state.pt"


class DPOModelConfig(BaseModel):
    model_config = _FROZEN_FORBID

    policy_checkpoint: Path = Field(
        ...,
        description="HF-format SFT checkpoint to optimize.",
    )
    reference_checkpoint: Path = Field(
        ...,
        description="Frozen HF-format reference checkpoint.",
    )
    dtype: DType = Field(default="bfloat16", description="Compute dtype for policy/reference.")
    use_safetensors: bool = Field(default=True, description="Require safetensors weights.")


class DPODataConfig(BaseModel):
    model_config = _FROZEN_FORBID

    pairs_path: Path = Field(..., description="JSONL preference pairs for DPO.")
    manifest_path: Path | None = Field(default=None, description="Pair-generation manifest path.")
    tokenized_cache_path: Path | None = Field(
        default=None,
        description="Optional torch cache for tokenized DPO pairs.",
    )
    rebuild_tokenized_cache: bool = Field(
        default=False,
        description="Force rebuilding tokenized_cache_path even if metadata matches.",
    )
    seed: int = Field(default=42, ge=0, description="Seed for shuffling and training RNGs.")


class DPOLossConfig(BaseModel):
    model_config = _FROZEN_FORBID

    beta: float = Field(default=0.1, gt=0.0, description="DPO KL temperature.")


class DPOTrainingConfig(BaseModel):
    model_config = _FROZEN_FORBID

    max_steps: int = Field(..., gt=0)
    warmup_steps: int = Field(default=100, ge=0)
    lr: float = Field(..., gt=0.0)
    weight_decay: float = Field(default=0.01, ge=0.0)
    grad_accum_steps: int = Field(default=1, ge=1)
    grad_clip: float = Field(default=1.0, gt=0.0)
    checkpoint_every_n_steps: int = Field(default=250, gt=0)
    per_device_pair_batch_size: int = Field(default=2, ge=1)
    dataloader_num_workers: int = Field(default=0, ge=0)
    pin_memory: bool = Field(default=True)

    @model_validator(mode="after")
    def _warmup_must_be_less_than_max(self) -> DPOTrainingConfig:
        if self.warmup_steps >= self.max_steps:
            raise ValueError(
                f"warmup_steps ({self.warmup_steps}) must be < max_steps ({self.max_steps})"
            )
        return self


class DPOConfig(BaseModel):
    """Top-level DPO training config."""

    model_config = _FROZEN_FORBID

    model: DPOModelConfig
    data: DPODataConfig
    dpo: DPOLossConfig = Field(default_factory=DPOLossConfig)
    training: DPOTrainingConfig
    packing: PackingConfig = Field(default_factory=PackingConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    checkpointing: CheckpointConfig = Field(default_factory=CheckpointConfig)

    @classmethod
    def from_yaml(cls, path: str | Path) -> DPOConfig:
        path = Path(path)
        with path.open("r", encoding="utf-8") as fp:
            raw = yaml.safe_load(fp)
        if not isinstance(raw, dict):
            raise ValueError(
                f"YAML root must be a mapping; got {type(raw).__name__} from {path}"
            )
        return cls.model_validate(raw)


class DPOCheckpointState(NamedTuple):
    """Read-only DPO checkpoint payload."""

    model_state_dict: dict[str, torch.Tensor]
    optimizer_state_dict: dict[str, Any]
    scheduler_state_dict: dict[str, Any]
    step: int
    rng_states: dict[str, Any]
    config: DPOConfig


def _capture_rng_states() -> dict[str, Any]:
    return {
        "torch": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
        "numpy": np.random.get_state(),
        "python": random.getstate(),
    }


def _apply_rng_states(rng_states: dict[str, Any]) -> None:
    torch_state = rng_states.get("torch")
    if torch_state is not None:
        torch.set_rng_state(torch_state)
    if torch.cuda.is_available() and rng_states.get("torch_cuda"):
        torch.cuda.set_rng_state_all(rng_states["torch_cuda"])
    numpy_state = rng_states.get("numpy")
    if numpy_state is not None:
        np.random.set_state(numpy_state)
    python_state = rng_states.get("python")
    if python_state is not None:
        random.setstate(python_state)


def _seed_everything(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.set_float32_matmul_precision("high")
    np.random.seed(seed)
    random.seed(seed)


def save_dpo_checkpoint(
    *,
    directory: Path,
    step: int,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    rng_states: dict[str, Any],
    config: DPOConfig,
) -> Path:
    """Atomically write a DPO checkpoint using the repo checkpoint layout."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    final_dir = directory / f"{_STEP_PREFIX}{step:0{_STEP_DIGITS}d}"
    tmp_dir = directory / f"{_STEP_PREFIX}{step:0{_STEP_DIGITS}d}.tmp"
    tmp_dir.mkdir(parents=True)

    safetensors.torch.save_model(model, str(tmp_dir / _MODEL_FILENAME))
    torch.save(
        {
            "step": step,
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "rng_states": rng_states,
            "config": config.model_dump(mode="json"),
        },
        tmp_dir / _STATE_FILENAME,
    )
    os.replace(tmp_dir, final_dir)
    return final_dir


def load_dpo_checkpoint(path: Path) -> DPOCheckpointState:
    """Read a DPO checkpoint without applying it to live objects."""
    path = Path(path)
    model_state_dict = safetensors.torch.load_file(str(path / _MODEL_FILENAME))
    payload = torch.load(path / _STATE_FILENAME, weights_only=False)
    return DPOCheckpointState(
        model_state_dict=model_state_dict,
        optimizer_state_dict=payload["optimizer"],
        scheduler_state_dict=payload["scheduler"],
        step=payload["step"],
        rng_states=payload["rng_states"],
        config=DPOConfig.model_validate(payload["config"]),
    )


def _resolve_device(device: str | None) -> torch.device:
    if device is None or device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def _metric_to_float(value: Any) -> float:
    if isinstance(value, torch.Tensor):
        return float(value.detach().float().item())
    return float(value)


class DPOTrainer:
    """Minimal full fine-tune DPO loop over offline preference pairs."""

    def __init__(self, config: DPOConfig, *, device: str | None = None) -> None:
        self.config = config
        self.device = _resolve_device(device)
        self.policy: torch.nn.Module | None = None
        self.reference: torch.nn.Module | None = None
        self.tokenizer: Any = None
        self.train_loader: DataLoader | None = None
        self.optimizer: torch.optim.Optimizer | None = None
        self.scheduler: torch.optim.lr_scheduler.LRScheduler | None = None
        self.global_step = 0

    def train(self) -> None:
        self._setup()
        self._run_training_loop()
        self._teardown()

    def _setup(self) -> None:
        _seed_everything(self.config.data.seed)

        dtype = getattr(torch, self.config.model.dtype)
        self.tokenizer = safe_load_tokenizer(str(self.config.model.policy_checkpoint))
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.policy = safe_load_model(
            str(self.config.model.policy_checkpoint),
            dtype=dtype,
            use_safetensors=self.config.model.use_safetensors,
        ).to(self.device)
        self.reference = safe_load_model(
            str(self.config.model.reference_checkpoint),
            dtype=dtype,
            use_safetensors=self.config.model.use_safetensors,
        ).to(self.device)
        self.policy.train()
        self.policy.config.use_cache = False
        self.reference.eval()
        self.reference.requires_grad_(False)
        self.reference.config.use_cache = False

        dataset = load_or_build_tokenized_preference_dataset(
            pairs_path=self.config.data.pairs_path,
            tokenizer=self.tokenizer,
            max_seq_len=self.config.packing.max_seq_len,
            cache_path=self.config.data.tokenized_cache_path,
            rebuild_cache=self.config.data.rebuild_tokenized_cache,
        )
        generator = torch.Generator()
        generator.manual_seed(self.config.data.seed)
        num_workers = self.config.training.dataloader_num_workers
        pin_memory = self.config.training.pin_memory and self.device.type == "cuda"
        dataloader_kwargs: dict[str, Any] = {}
        if num_workers > 0:
            dataloader_kwargs["persistent_workers"] = True
            dataloader_kwargs["prefetch_factor"] = 2
        self.train_loader = DataLoader(
            dataset,
            batch_size=self.config.training.per_device_pair_batch_size,
            shuffle=True,
            generator=generator,
            num_workers=num_workers,
            pin_memory=pin_memory,
            collate_fn=DPOCollator(
                tokenizer=None,
                max_seq_len=self.config.packing.max_seq_len,
                pad_token_id=self.tokenizer.pad_token_id,
            ),
            **dataloader_kwargs,
        )

        self.optimizer = build_optimizer(
            self.policy,
            lr=self.config.training.lr,
            weight_decay=self.config.training.weight_decay,
        )
        self.scheduler = build_lr_scheduler(
            self.optimizer,
            total_steps=self.config.training.max_steps,
            warmup_steps=self.config.training.warmup_steps,
        )

        if self.config.checkpointing.resume_from is not None:
            self._load_resume(self.config.checkpointing.resume_from)

        wandb.init(
            project=self.config.logging.wandb_project,
            name=self.config.logging.run_name,
            config=self.config.model_dump(mode="json"),
            mode=os.environ.get("WANDB_MODE"),
        )

    def _load_resume(self, resume_from: Path) -> None:
        assert self.policy is not None and self.optimizer is not None and self.scheduler is not None
        state = load_dpo_checkpoint(resume_from)
        self.policy.load_state_dict(state.model_state_dict, strict=False)
        self.optimizer.load_state_dict(state.optimizer_state_dict)
        self.scheduler.load_state_dict(state.scheduler_state_dict)
        _apply_rng_states(state.rng_states)
        self.global_step = state.step

    def _run_training_loop(self) -> None:
        assert (
            self.train_loader is not None
            and self.optimizer is not None
            and self.scheduler is not None
        )
        assert self.policy is not None

        max_steps = self.config.training.max_steps
        grad_accum = self.config.training.grad_accum_steps
        grad_clip = self.config.training.grad_clip
        ckpt_every = self.config.training.checkpoint_every_n_steps

        accumulated_loss = 0.0
        accumulated_metrics: dict[str, float] = {}
        micro_step = 0
        window_tokens = 0
        window_start = time.perf_counter()
        loader_iter = iter(self.train_loader)

        while self.global_step < max_steps:
            try:
                batch = next(loader_iter)
            except StopIteration:
                loader_iter = iter(self.train_loader)
                batch = next(loader_iter)

            loss, metrics = self._forward_loss(batch)
            check_finite_loss(loss, self.global_step)
            (loss / grad_accum).backward()

            accumulated_loss += float(loss.detach().float().item()) / grad_accum
            for key, value in metrics.items():
                if key == "loss":
                    continue
                accumulated_metrics[key] = accumulated_metrics.get(key, 0.0) + (
                    _metric_to_float(value) / grad_accum
                )

            useful_tokens = int((batch["chosen_labels"] != IGNORE_INDEX).sum().item())
            useful_tokens += int((batch["rejected_labels"] != IGNORE_INDEX).sum().item())
            window_tokens += useful_tokens
            micro_step += 1

            if micro_step % grad_accum != 0:
                continue

            grad_norm = torch.nn.utils.clip_grad_norm_(self.policy.parameters(), grad_clip)
            self.optimizer.step()
            self.scheduler.step()
            self.optimizer.zero_grad()
            self.global_step += 1

            self._log_train_metrics(
                step=self.global_step,
                loss=accumulated_loss,
                grad_norm=float(grad_norm),
                metrics=accumulated_metrics,
            )

            if self.global_step % 10 == 0:
                elapsed = time.perf_counter() - window_start
                if elapsed > 0:
                    wandb.log(
                        {"train/tokens_per_sec": window_tokens / elapsed},
                        step=self.global_step,
                    )
                window_tokens = 0
                window_start = time.perf_counter()

            accumulated_loss = 0.0
            accumulated_metrics = {}
            micro_step = 0

            if self.global_step % ckpt_every == 0:
                self._save_checkpoint()

    def _forward_loss(self, batch: dict[str, Any]) -> tuple[torch.Tensor, dict[str, Any]]:
        assert self.policy is not None and self.reference is not None
        chosen_input_ids = batch["chosen_input_ids"].to(self.device, non_blocking=True)
        chosen_attention_mask = batch["chosen_attention_mask"].to(
            self.device,
            non_blocking=True,
        )
        chosen_labels = batch["chosen_labels"].to(self.device, non_blocking=True)
        rejected_input_ids = batch["rejected_input_ids"].to(self.device, non_blocking=True)
        rejected_attention_mask = batch["rejected_attention_mask"].to(
            self.device,
            non_blocking=True,
        )
        rejected_labels = batch["rejected_labels"].to(self.device, non_blocking=True)

        input_ids = torch.cat([chosen_input_ids, rejected_input_ids], dim=0)
        attention_mask = torch.cat([chosen_attention_mask, rejected_attention_mask], dim=0)
        pair_count = chosen_input_ids.size(0)

        policy_logits = self.policy(
            input_ids=input_ids,
            attention_mask=attention_mask,
        ).logits
        policy_chosen_logits, policy_rejected_logits = policy_logits.split(pair_count, dim=0)
        del policy_logits

        with torch.no_grad():
            ref_logits = self.reference(
                input_ids=input_ids,
                attention_mask=attention_mask,
            ).logits
        ref_chosen_logits, ref_rejected_logits = ref_logits.split(pair_count, dim=0)

        return dpo_loss_from_logits(
            policy_chosen_logits=policy_chosen_logits,
            policy_rejected_logits=policy_rejected_logits,
            ref_chosen_logits=ref_chosen_logits,
            ref_rejected_logits=ref_rejected_logits,
            chosen_labels=chosen_labels,
            rejected_labels=rejected_labels,
            beta=self.config.dpo.beta,
        )

    def _log_train_metrics(
        self,
        *,
        step: int,
        loss: float,
        grad_norm: float,
        metrics: dict[str, float],
    ) -> None:
        assert self.optimizer is not None
        payload = {
            "train/loss": loss,
            "train/lr": self.optimizer.param_groups[0]["lr"],
            "train/grad_norm": grad_norm,
        }
        payload.update({f"train/{key}": value for key, value in metrics.items()})
        wandb.log(payload, step=step)

    def _save_checkpoint(self) -> Path:
        assert (
            self.policy is not None
            and self.optimizer is not None
            and self.scheduler is not None
        )
        path = save_dpo_checkpoint(
            directory=Path(self.config.checkpointing.save_dir),
            step=self.global_step,
            model=self.policy,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            rng_states=_capture_rng_states(),
            config=self.config,
        )
        apply_retention_policy(
            directory=Path(self.config.checkpointing.save_dir),
            last_n=self.config.checkpointing.retention_last_n,
            best_so_far=None,
        )
        return path

    def _teardown(self) -> None:
        ckpt_every = self.config.training.checkpoint_every_n_steps
        if self.global_step > 0 and self.global_step % ckpt_every != 0:
            self._save_checkpoint()
        wandb.finish()


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python -m finpost.training.dpo_train",
        description="Run Phase 1 Direct Preference Optimization from a YAML config.",
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--resume-from", type=Path, default=None)
    return parser.parse_args(argv)


def resolve_config(
    *,
    config_path: Path,
    max_steps_override: int | None,
    resume_from_override: Path | None,
) -> DPOConfig:
    with config_path.open("r", encoding="utf-8") as fp:
        raw = yaml.safe_load(fp)
    if not isinstance(raw, dict):
        raise ValueError(
            f"YAML root must be a mapping; got {type(raw).__name__} from {config_path}"
        )

    if max_steps_override is not None:
        training = raw.setdefault("training", {})
        training["max_steps"] = max_steps_override
        if training.get("warmup_steps", 0) >= max_steps_override:
            training["warmup_steps"] = max(max_steps_override - 1, 0)

    if resume_from_override is not None:
        raw.setdefault("checkpointing", {})["resume_from"] = str(resume_from_override)

    return DPOConfig.model_validate(raw)


def print_effective_config(
    config: DPOConfig,
    *,
    stream: IO[str] | None = None,
) -> None:
    out = stream if stream is not None else None
    effective_pair_batch = (
        config.training.per_device_pair_batch_size * config.training.grad_accum_steps
    )
    lines = [
        "Effective DPO config:",
        f"  policy_checkpoint:       {config.model.policy_checkpoint}",
        f"  reference_checkpoint:    {config.model.reference_checkpoint}",
        f"  dtype:                   {config.model.dtype}",
        f"  pairs_path:              {config.data.pairs_path}",
        f"  tokenized_cache_path:    {config.data.tokenized_cache_path}",
        f"  beta:                    {config.dpo.beta}",
        f"  max_steps:               {config.training.max_steps}",
        f"  warmup_steps:            {config.training.warmup_steps}",
        f"  lr:                      {config.training.lr}",
        f"  per_device_pair_batch:   {config.training.per_device_pair_batch_size}",
        f"  grad_accum_steps:        {config.training.grad_accum_steps}",
        f"  effective pair batch:    {effective_pair_batch}",
        f"  dataloader_num_workers:  {config.training.dataloader_num_workers}",
        f"  pin_memory:              {config.training.pin_memory}",
        f"  max_seq_len:             {config.packing.max_seq_len}",
        f"  save_dir:                {config.checkpointing.save_dir}",
        f"  resume_from:             {config.checkpointing.resume_from}",
        f"  run_name:                {config.logging.run_name}",
    ]
    text = "\n".join(lines) + "\n"
    if out is None:
        print(text, end="")
    else:
        out.write(text)
        out.flush()


def cleanup_failed_tmp_checkpoints(directory: Path) -> None:
    """Remove stale atomic-write temp dirs before a fresh DPO run."""
    directory = Path(directory)
    if not directory.exists():
        return
    for path in directory.glob("step-*.tmp"):
        if path.is_dir():
            shutil.rmtree(path)


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    config = resolve_config(
        config_path=args.config,
        max_steps_override=args.max_steps,
        resume_from_override=args.resume_from,
    )
    print_effective_config(config)
    cleanup_failed_tmp_checkpoints(Path(config.checkpointing.save_dir))
    DPOTrainer(config, device=args.device).train()


if __name__ == "__main__":
    main()
