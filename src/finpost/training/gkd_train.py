"""Command-line GKD trainer for on-policy distillation on FinChain.

Implements one training step of Agarwal et al. 2023 (arXiv 2306.13649)
on-policy GKD with full-distribution JSD(beta) divergence:

  1. Student generates K completions per prompt under ``no_grad`` with
     temperature 1 (paper default).
  2. Teacher does a single ``no_grad`` forward over (prompt + completion).
  3. Student does a forward over (prompt + completion) WITH gradient.
  4. ``gkd_jsd_loss`` averages the per-token divergence over response
     tokens. Gradient flows through the student term only. The
     divergence can be chunked over sequence positions to reduce peak
     log-softmax memory without changing the objective.
  5. AdamW step.

The trainer supports multi-GPU DDP through Hugging Face Accelerate when
launched with ``accelerate launch --num_processes N``. LoRA, 4-bit teacher
loading, and the lambda mixed-policy fraction remain explicit future work.

Pad / EOS contract
------------------

The trainer requires ``tokenizer.pad_token_id != tokenizer.eos_token_id``
so that post-EOS padding can be masked out of the label tensor
without also masking the EOS token itself. Qwen2.5 tokenizers
satisfy this by default (pad = ``<|endoftext|>``, eos = ``<|im_end|>``).
For models that share the two tokens (e.g. tiny-gpt2), set a distinct
pad token before constructing the trainer.
"""

from __future__ import annotations

import argparse
import os
import random
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import IO, Any, NamedTuple

import numpy as np
import safetensors.torch
import torch
import wandb
import yaml
from accelerate import Accelerator
from pydantic import BaseModel, ConfigDict, Field, model_validator
from torch.utils.data import DataLoader, Dataset

from finpost.data.finchain_dataset import load_finchain
from finpost.data.schema import Example
from finpost.training.gkd import gkd_jsd_loss
from finpost.safety import safe_load_model, safe_load_tokenizer
from finpost.training._guards import check_finite_loss
from finpost.training.checkpoint import apply_retention_policy
from finpost.training.config import CheckpointConfig, DType, LoggingConfig
from finpost.training.dataset import serialize_prompt
from finpost.training.masking import mask_prompt_tokens
from finpost.training.optim import build_8bit_adamw, build_lr_scheduler, build_optimizer

_FROZEN_FORBID = ConfigDict(frozen=True, extra="forbid")
_STEP_PREFIX = "step-"
_STEP_DIGITS = 8
_MODEL_FILENAME = "model.safetensors"
_STATE_FILENAME = "state.pt"


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


class GKDModelConfig(BaseModel):
    model_config = _FROZEN_FORBID

    student_checkpoint: Path = Field(
        ..., description="HF-format student checkpoint (for this run, the selected FinChain SFT model)."
    )
    teacher_checkpoint: Path = Field(
        ..., description="HF-format teacher checkpoint (e.g. Qwen2.5-7B-Instruct)."
    )
    student_dtype: DType = Field(default="bfloat16")
    teacher_dtype: DType = Field(default="bfloat16")
    use_safetensors: bool = Field(default=True)
    gradient_checkpointing: bool = Field(
        default=True,
        description="Enable gradient checkpointing on the student to bound activation memory.",
    )
    use_8bit_optimizer: bool = Field(
        default=False,
        description=(
            "Use bitsandbytes AdamW8bit instead of the standard fp32 AdamW. "
            "Reduces optimizer state from ~12 GB to ~3 GB on Qwen2.5-1.5B. "
            "Requires bitsandbytes (install with `pip install -e .[rlvr]`). "
            "Recommended when running with both student and Qwen2.5-7B teacher on a single A40."
        ),
    )
    attn_implementation: str | None = Field(
        default=None,
        description=(
            "Attention backend forwarded to HF `from_pretrained` for both "
            "student and teacher. None preserves the HF default (typically "
            "`sdpa`). Set to `flash_attention_2` in a CUDA environment with "
            "`flash-attn` installed for ~1.5-2x generation throughput."
        ),
    )


class GKDDataConfig(BaseModel):
    model_config = _FROZEN_FORBID

    source: str = Field(default="finchain", description="Prompt source name.")
    split: str = Field(default="train", description="Split passed to the loader.")
    max_prompt_len: int = Field(default=512, gt=0)
    require_full_prompt_coverage: bool = Field(
        default=True,
        description=(
            "Raise if tokenization drops any prompt. Set false only for explicit "
            "length-ablation runs where a smaller training distribution is intended."
        ),
    )
    seed: int = Field(default=42, ge=0)


class GKDLossConfig(BaseModel):
    model_config = _FROZEN_FORBID

    beta: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description=(
            "JSD(beta) interpolation. 0.0 = forward KL (paper default for arithmetic reasoning). "
            "0.5 = symmetric JSD. 1.0 = exact reverse KL (full-vocab)."
        ),
    )
    sequence_chunk_size: int | None = Field(
        default=64,
        gt=0,
        description=(
            "Shifted sequence positions per full-vocab divergence chunk. "
            "Keeps the exact JSD(beta) loss while reducing peak log-softmax VRAM."
        ),
    )


class GKDTrainingConfig(BaseModel):
    model_config = _FROZEN_FORBID

    max_steps: int = Field(..., gt=0)
    warmup_steps: int = Field(default=100, ge=0)
    lr: float = Field(..., gt=0.0)
    weight_decay: float = Field(default=0.01, ge=0.0)
    grad_accum_steps: int = Field(default=1, ge=1)
    grad_clip: float = Field(default=1.0, gt=0.0)
    checkpoint_every_n_steps: int = Field(default=250, gt=0)
    per_device_prompt_batch_size: int = Field(default=1, ge=1)
    rollouts_per_prompt: int = Field(
        default=4,
        ge=1,
        description="K in the paper. How many completions to sample per prompt per step.",
    )
    max_completion_length: int = Field(
        default=512,
        gt=0,
        description="Maximum generated completion tokens per OPD/GKD rollout.",
    )
    student_temperature: float = Field(
        default=1.0,
        gt=0.0,
        description="gamma in the paper; 1.0 is the paper-wide default during training.",
    )
    dataloader_num_workers: int = Field(default=0, ge=0)
    pin_memory: bool = Field(default=True)

    @model_validator(mode="after")
    def _warmup_check(self) -> GKDTrainingConfig:
        if self.warmup_steps >= self.max_steps:
            raise ValueError(
                f"warmup_steps ({self.warmup_steps}) must be < max_steps ({self.max_steps})"
            )
        return self


class GKDConfig(BaseModel):
    model_config = _FROZEN_FORBID

    model: GKDModelConfig
    data: GKDDataConfig = Field(default_factory=GKDDataConfig)
    gkd: GKDLossConfig = Field(default_factory=GKDLossConfig)
    training: GKDTrainingConfig
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    checkpointing: CheckpointConfig = Field(default_factory=CheckpointConfig)

    @classmethod
    def from_yaml(cls, path: str | Path) -> GKDConfig:
        path = Path(path)
        with path.open("r", encoding="utf-8") as fp:
            raw = yaml.safe_load(fp)
        if not isinstance(raw, dict):
            raise ValueError(
                f"YAML root must be a mapping; got {type(raw).__name__} from {path}"
            )
        return cls.model_validate(raw)


class GKDCheckpointState(NamedTuple):
    model_state_dict: dict[str, torch.Tensor]
    optimizer_state_dict: dict[str, Any]
    scheduler_state_dict: dict[str, Any]
    step: int
    rng_states: dict[str, Any]
    config: GKDConfig


# ---------------------------------------------------------------------------
# RNG / checkpoint helpers (parallel structure to dpo_train.py)
# ---------------------------------------------------------------------------


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


def save_gkd_checkpoint(
    *,
    directory: Path,
    step: int,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    rng_states: dict[str, Any],
    config: GKDConfig,
) -> Path:
    """Atomically write a GKD student checkpoint to ``directory/step-<n>/``."""
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


def load_gkd_checkpoint(path: Path) -> GKDCheckpointState:
    path = Path(path)
    model_state_dict = safetensors.torch.load_file(str(path / _MODEL_FILENAME))
    payload = torch.load(path / _STATE_FILENAME, weights_only=False)
    return GKDCheckpointState(
        model_state_dict=model_state_dict,
        optimizer_state_dict=payload["optimizer"],
        scheduler_state_dict=payload["scheduler"],
        step=payload["step"],
        rng_states=payload["rng_states"],
        config=GKDConfig.model_validate(payload["config"]),
    )


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TokenizedPrompt:
    input_ids: torch.Tensor  # (prompt_len,)
    attention_mask: torch.Tensor
    prompt_id: str | None = None


class TokenizedPromptDataset(Dataset):
    def __init__(self, prompts: list[TokenizedPrompt]) -> None:
        if not prompts:
            raise ValueError("GKD prompt dataset is empty")
        self.prompts = prompts

    def __len__(self) -> int:
        return len(self.prompts)

    def __getitem__(self, idx: int) -> TokenizedPrompt:
        return self.prompts[idx]


def tokenize_prompts_for_gkd(
    examples: list[Example],
    *,
    tokenizer: Any,
    max_prompt_len: int,
    require_full_prompt_coverage: bool = True,
) -> list[TokenizedPrompt]:
    """Tokenize ``Example`` records into GKD prompts using the chat template.

    Prompts that exceed ``max_prompt_len`` after tokenization are
    skipped — truncating a chat-template prompt corrupts the role
    markers and produces ungenerable inputs. With the default coverage guard,
    any such skip raises instead of quietly shrinking the OPD/GKD train set.
    """
    tokenized: list[TokenizedPrompt] = []
    dropped_prompt_ids: list[str] = []
    for example in examples:
        text = serialize_prompt(example.prompt)
        encoded = tokenizer(text, add_special_tokens=False)
        ids = encoded["input_ids"]
        if isinstance(ids, torch.Tensor):
            ids = ids.flatten().tolist()
        if not ids:
            dropped_prompt_ids.append(example.id)
            continue
        if len(ids) > max_prompt_len:
            dropped_prompt_ids.append(example.id)
            continue
        input_ids = torch.tensor(ids, dtype=torch.long)
        attention_mask = torch.ones_like(input_ids)
        tokenized.append(
            TokenizedPrompt(
                input_ids=input_ids,
                attention_mask=attention_mask,
                prompt_id=example.id,
            )
        )
    if dropped_prompt_ids and require_full_prompt_coverage:
        sample = ", ".join(dropped_prompt_ids[:5])
        suffix = "..." if len(dropped_prompt_ids) > 5 else ""
        raise ValueError(
            "GKD prompt coverage check failed: "
            f"dropped {len(dropped_prompt_ids)}/{len(examples)} prompts at "
            f"max_prompt_len={max_prompt_len}. First dropped prompt ids: "
            f"{sample}{suffix}. Increase max_prompt_len or set "
            "data.require_full_prompt_coverage=false for an explicit ablation."
        )
    if not tokenized:
        raise ValueError("No GKD prompts produced; check max_prompt_len and source data.")
    return tokenized


class GKDPromptCollator:
    """Left-pad a batch of prompts for batched generation.

    Left-padding is the convention required by Hugging Face's
    ``generate()`` API so all sequences in the batch produce new
    tokens starting at the same position.
    """

    def __init__(self, *, pad_token_id: int) -> None:
        self.pad_token_id = int(pad_token_id)

    def __call__(self, batch: list[TokenizedPrompt]) -> dict[str, Any]:
        width = max(len(p.input_ids) for p in batch)
        input_rows: list[torch.Tensor] = []
        mask_rows: list[torch.Tensor] = []
        for prompt in batch:
            n = len(prompt.input_ids)
            pad_len = width - n
            input_rows.append(
                torch.cat(
                    [
                        torch.full((pad_len,), self.pad_token_id, dtype=torch.long),
                        prompt.input_ids,
                    ]
                )
            )
            mask_rows.append(
                torch.cat(
                    [
                        torch.zeros((pad_len,), dtype=torch.long),
                        prompt.attention_mask,
                    ]
                )
            )
        return {
            "prompt_input_ids": torch.stack(input_rows),
            "prompt_attention_mask": torch.stack(mask_rows),
            "prompt_ids": [p.prompt_id for p in batch],
        }


def _load_finchain_prompts(data_cfg: GKDDataConfig) -> list[Example]:
    if data_cfg.source != "finchain":
        raise ValueError(
            f"unsupported prompt source {data_cfg.source!r}; only 'finchain' is wired up"
        )
    return load_finchain(split=data_cfg.split)


# ---------------------------------------------------------------------------
# Training step (free function for testability)
# ---------------------------------------------------------------------------


def build_gkd_labels(
    *,
    full_input_ids: torch.Tensor,
    padded_prompt_len: int,
    pad_token_id: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build labels and forward-attention mask for a GKD batch.

    ``full_input_ids`` is the output of ``student.generate``: shape
    ``(B*K, padded_prompt_len + new_tokens)``. The function returns

      labels:   shape ``(B*K, T)``. ``IGNORE_INDEX`` at prompt
                positions (including their left-padding) and at any
                position whose token is ``pad_token_id`` (the
                post-EOS padding region produced by early-stopped
                generation). Real generated token IDs elsewhere.
      attention_mask: shape ``(B*K, T)``. ``1`` for every non-pad
                position, ``0`` for pad. Pass this to both the
                teacher and student forward passes so attention
                ignores padding correctly.

    Caller must guarantee ``pad_token_id != eos_token_id`` (the
    trainer enforces this); otherwise EOS would be incorrectly
    masked.
    """
    attention_mask = (full_input_ids != pad_token_id).long()
    prompt_lengths = torch.full(
        (full_input_ids.size(0),),
        padded_prompt_len,
        dtype=torch.long,
        device=full_input_ids.device,
    )
    labels = mask_prompt_tokens(
        full_input_ids,
        prompt_lengths,
        attention_mask=attention_mask,
    )
    return labels, attention_mask


def gkd_train_step(
    *,
    student: torch.nn.Module,
    teacher: torch.nn.Module,
    student_generate_model: torch.nn.Module | None = None,
    prompt_input_ids: torch.Tensor,
    prompt_attention_mask: torch.Tensor,
    pad_token_id: int,
    max_completion_length: int,
    student_temperature: float,
    rollouts_per_prompt: int,
    beta: float,
    sequence_chunk_size: int | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """One GKD training step (forward and loss only; caller handles backward).

    Returns the scalar loss tensor (with grad) and the metrics dict
    from ``gkd_jsd_loss``.

    Memory note: the model forwards still return full ``(B*K, T, V)``
    logits, but the JSD loss can process those logits in sequence chunks
    so it does not also materialize full-sequence float32 log-probability
    buffers for both models at once.
    """
    if rollouts_per_prompt > 1:
        prompt_input_ids = prompt_input_ids.repeat_interleave(rollouts_per_prompt, dim=0)
        prompt_attention_mask = prompt_attention_mask.repeat_interleave(
            rollouts_per_prompt, dim=0
        )
    padded_prompt_len = prompt_input_ids.size(-1)

    # Step 1: student generates K samples per prompt under no_grad.
    # We switch to eval() so dropout is off — the paper's framework
    # assumes p_S is a fixed distribution at sampling time, which
    # requires dropout off.
    generate_model = student_generate_model or student
    was_training = student.training
    was_generation_training = generate_model.training
    generate_model.eval()
    try:
        with torch.no_grad():
            outputs = generate_model.generate(
                input_ids=prompt_input_ids,
                attention_mask=prompt_attention_mask,
                max_new_tokens=max_completion_length,
                do_sample=True,
                temperature=student_temperature,
                top_p=1.0,
                top_k=0,
                pad_token_id=pad_token_id,
                use_cache=True,
            )
    finally:
        if was_generation_training:
            generate_model.train()
        if was_training:
            student.train()

    # Step 2: labels + attention mask for the dual forward.
    labels, attention_mask = build_gkd_labels(
        full_input_ids=outputs,
        padded_prompt_len=padded_prompt_len,
        pad_token_id=pad_token_id,
    )

    # Step 3: teacher forward (no_grad). The teacher receives the
    # student-sampled trajectory and scores it.
    with torch.no_grad():
        teacher_logits = teacher(
            input_ids=outputs,
            attention_mask=attention_mask,
        ).logits

    # Step 4: student forward (with grad).
    student_logits = student(
        input_ids=outputs,
        attention_mask=attention_mask,
    ).logits

    # Step 5: per-token JSD(beta) averaged over response tokens.
    return gkd_jsd_loss(
        student_logits=student_logits,
        teacher_logits=teacher_logits,
        labels=labels,
        beta=beta,
        sequence_chunk_size=sequence_chunk_size,
    )


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------


def _resolve_device(device: str | None) -> torch.device:
    if device is None or device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def _accelerate_launch_requested() -> bool:
    """Return True when this process was launched as part of a distributed job."""
    try:
        return int(os.environ.get("WORLD_SIZE", "1")) > 1
    except ValueError:
        return False


def _metric_to_float(value: Any) -> float:
    if isinstance(value, torch.Tensor):
        return float(value.detach().float().item())
    return float(value)


class GKDTrainer:
    """Full-fine-tune GKD trainer over FinChain prompts.

    Normal ``python -m finpost.training.gkd_train`` runs single-process.
    ``accelerate launch --num_processes N`` wraps the student, optimizer,
    scheduler, and prompt dataloader for DDP. Each rank keeps a local frozen
    teacher copy because the teacher has no gradients to synchronize.
    """

    def __init__(self, config: GKDConfig, *, device: str | None = None) -> None:
        self.config = config
        self.accelerator: Accelerator | None = None
        self.requested_device = device
        self.device = _resolve_device(device)
        self.student: torch.nn.Module | None = None
        self.teacher: torch.nn.Module | None = None
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
        if _accelerate_launch_requested():
            if self.requested_device not in (None, "auto"):
                raise ValueError(
                    "Do not pass --device with a distributed Accelerate launch; "
                    "Accelerate assigns one device per rank."
                )
            self.accelerator = Accelerator()
            self.device = self.accelerator.device

        _seed_everything(self.config.data.seed)

        student_dtype = getattr(torch, self.config.model.student_dtype)
        teacher_dtype = getattr(torch, self.config.model.teacher_dtype)

        self.tokenizer = safe_load_tokenizer(str(self.config.model.student_checkpoint))
        if self.tokenizer.pad_token_id is None:
            raise ValueError(
                "GKD trainer requires the tokenizer to provide pad_token_id. Configure "
                "tokenizer.pad_token before saving the student checkpoint."
            )
        if self.tokenizer.eos_token_id is not None and (
            self.tokenizer.pad_token_id == self.tokenizer.eos_token_id
        ):
            raise ValueError(
                "GKD trainer requires pad_token_id != eos_token_id so that post-EOS "
                "padding can be masked without also masking the EOS token. Configure "
                "a distinct pad_token on the student tokenizer."
            )

        self.student = safe_load_model(
            str(self.config.model.student_checkpoint),
            dtype=student_dtype,
            use_safetensors=self.config.model.use_safetensors,
            attn_implementation=self.config.model.attn_implementation,
        ).to(self.device)
        self.teacher = safe_load_model(
            str(self.config.model.teacher_checkpoint),
            dtype=teacher_dtype,
            use_safetensors=self.config.model.use_safetensors,
            attn_implementation=self.config.model.attn_implementation,
        ).to(self.device)

        self.student.train()
        self.student.config.use_cache = False
        if self.config.model.gradient_checkpointing:
            self.student.gradient_checkpointing_enable()
            # Required for grads to flow when gradient checkpointing
            # is on and the model has frozen input embeddings.
            if hasattr(self.student, "enable_input_require_grads"):
                self.student.enable_input_require_grads()

        self.teacher.eval()
        self.teacher.requires_grad_(False)
        self.teacher.config.use_cache = False

        examples = _load_finchain_prompts(self.config.data)
        prompts = tokenize_prompts_for_gkd(
            examples,
            tokenizer=self.tokenizer,
            max_prompt_len=self.config.data.max_prompt_len,
            require_full_prompt_coverage=self.config.data.require_full_prompt_coverage,
        )
        dataset = TokenizedPromptDataset(prompts)

        generator = torch.Generator()
        generator.manual_seed(self.config.data.seed)
        num_workers = self.config.training.dataloader_num_workers
        pin_memory = self.config.training.pin_memory and self.device.type == "cuda"
        loader_kwargs: dict[str, Any] = {}
        if num_workers > 0:
            loader_kwargs["persistent_workers"] = True
            loader_kwargs["prefetch_factor"] = 2

        self.train_loader = DataLoader(
            dataset,
            batch_size=self.config.training.per_device_prompt_batch_size,
            shuffle=True,
            generator=generator,
            num_workers=num_workers,
            pin_memory=pin_memory,
            collate_fn=GKDPromptCollator(pad_token_id=self.tokenizer.pad_token_id),
            **loader_kwargs,
        )

        if self.config.model.use_8bit_optimizer:
            self.optimizer = build_8bit_adamw(
                self.student,
                lr=self.config.training.lr,
                weight_decay=self.config.training.weight_decay,
            )
        else:
            self.optimizer = build_optimizer(
                self.student,
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

        if self.accelerator is not None:
            (
                self.student,
                self.optimizer,
                self.train_loader,
                self.scheduler,
            ) = self.accelerator.prepare(
                self.student,
                self.optimizer,
                self.train_loader,
                self.scheduler,
            )

        if self._is_main_process:
            wandb.init(
                project=self.config.logging.wandb_project,
                name=self.config.logging.run_name,
                config=self.config.model_dump(mode="json")
                | {"distributed_world_size": self.world_size},
                mode=os.environ.get("WANDB_MODE"),
            )

    def _load_resume(self, resume_from: Path) -> None:
        assert self.student is not None
        assert self.optimizer is not None
        assert self.scheduler is not None
        state = load_gkd_checkpoint(resume_from)
        self.student.load_state_dict(state.model_state_dict, strict=False)
        self.optimizer.load_state_dict(state.optimizer_state_dict)
        self.scheduler.load_state_dict(state.scheduler_state_dict)
        _apply_rng_states(state.rng_states)
        self.global_step = state.step

    def _run_training_loop(self) -> None:
        assert (
            self.train_loader is not None
            and self.optimizer is not None
            and self.scheduler is not None
            and self.student is not None
            and self.teacher is not None
        )

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
            scaled_loss = loss / grad_accum
            if self.accelerator is not None:
                self.accelerator.backward(scaled_loss)
            else:
                scaled_loss.backward()

            accumulated_loss += float(loss.detach().float().item()) / grad_accum
            for key, value in metrics.items():
                if key == "loss":
                    continue
                accumulated_metrics[key] = accumulated_metrics.get(key, 0.0) + (
                    _metric_to_float(value) / grad_accum
                )

            window_tokens += int(metrics["response_tokens"].item())
            micro_step += 1

            if micro_step % grad_accum != 0:
                continue

            if self.accelerator is not None:
                grad_norm = self.accelerator.clip_grad_norm_(
                    self.student.parameters(), grad_clip
                )
            else:
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    self.student.parameters(), grad_clip
                )
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
                    self._wandb_log(
                        {"train/response_tokens_per_sec": window_tokens / elapsed},
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
        assert self.student is not None and self.teacher is not None
        prompt_input_ids = batch["prompt_input_ids"].to(self.device, non_blocking=True)
        prompt_attention_mask = batch["prompt_attention_mask"].to(
            self.device, non_blocking=True
        )
        student_generate_model = (
            self.accelerator.unwrap_model(self.student)
            if self.accelerator is not None
            else self.student
        )
        return gkd_train_step(
            student=self.student,
            teacher=self.teacher,
            student_generate_model=student_generate_model,
            prompt_input_ids=prompt_input_ids,
            prompt_attention_mask=prompt_attention_mask,
            pad_token_id=int(self.tokenizer.pad_token_id),
            max_completion_length=self.config.training.max_completion_length,
            student_temperature=self.config.training.student_temperature,
            rollouts_per_prompt=self.config.training.rollouts_per_prompt,
            beta=self.config.gkd.beta,
            sequence_chunk_size=self.config.gkd.sequence_chunk_size,
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
        self._wandb_log(payload, step=step)

    def _save_checkpoint(self) -> Path:
        assert (
            self.student is not None
            and self.optimizer is not None
            and self.scheduler is not None
        )
        if self.accelerator is not None:
            self.accelerator.wait_for_everyone()
        path = Path(self.config.checkpointing.save_dir) / (
            f"{_STEP_PREFIX}{self.global_step:0{_STEP_DIGITS}d}"
        )
        if self._is_main_process:
            model_to_save = (
                self.accelerator.unwrap_model(self.student)
                if self.accelerator is not None
                else self.student
            )
            path = save_gkd_checkpoint(
                directory=Path(self.config.checkpointing.save_dir),
                step=self.global_step,
                model=model_to_save,
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
        if self.accelerator is not None:
            self.accelerator.wait_for_everyone()
        return path

    def _teardown(self) -> None:
        ckpt_every = self.config.training.checkpoint_every_n_steps
        if self.global_step > 0 and self.global_step % ckpt_every != 0:
            self._save_checkpoint()
        if self._is_main_process:
            wandb.finish()
        if self.accelerator is not None:
            self.accelerator.wait_for_everyone()

    @property
    def _is_main_process(self) -> bool:
        return self.accelerator is None or self.accelerator.is_main_process

    @property
    def world_size(self) -> int:
        if self.accelerator is not None:
            return self.accelerator.num_processes
        return 1

    def _wandb_log(self, payload: dict[str, Any], *, step: int) -> None:
        if self._is_main_process:
            wandb.log(payload, step=step)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python -m finpost.training.gkd_train",
        description="Run on-policy distillation (GKD) from a YAML config.",
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
) -> GKDConfig:
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
    return GKDConfig.model_validate(raw)


def print_effective_config(config: GKDConfig, *, stream: IO[str] | None = None) -> None:
    lines = [
        "Effective GKD config:",
        f"  student_checkpoint:      {config.model.student_checkpoint}",
        f"  teacher_checkpoint:      {config.model.teacher_checkpoint}",
        f"  student_dtype:           {config.model.student_dtype}",
        f"  teacher_dtype:           {config.model.teacher_dtype}",
        f"  gradient_checkpointing:  {config.model.gradient_checkpointing}",
        f"  beta:                    {config.gkd.beta}",
        f"  sequence_chunk_size:     {config.gkd.sequence_chunk_size}",
        f"  max_steps:               {config.training.max_steps}",
        f"  warmup_steps:            {config.training.warmup_steps}",
        f"  lr:                      {config.training.lr}",
        f"  prompt_batch:            {config.training.per_device_prompt_batch_size}",
        f"  rollouts_per_prompt:     {config.training.rollouts_per_prompt}",
        f"  max_completion_length:   {config.training.max_completion_length}",
        f"  student_temperature:     {config.training.student_temperature}",
        f"  grad_accum_steps:        {config.training.grad_accum_steps}",
        f"  max_prompt_len:          {config.data.max_prompt_len}",
        f"  full_prompt_coverage:    {config.data.require_full_prompt_coverage}",
        f"  save_dir:                {config.checkpointing.save_dir}",
        f"  resume_from:             {config.checkpointing.resume_from}",
        f"  run_name:                {config.logging.run_name}",
    ]
    text = "\n".join(lines) + "\n"
    if stream is None:
        print(text, end="")
    else:
        stream.write(text)
        stream.flush()


def cleanup_failed_tmp_checkpoints(directory: Path) -> None:
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
    GKDTrainer(config, device=args.device).train()


if __name__ == "__main__":
    main()
