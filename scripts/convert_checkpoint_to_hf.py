"""Convert a finpost training checkpoint to Hugging Face format for eval.

The training checkpoint directory written by ``finpost.training.checkpoint``
contains:

  - ``model.safetensors``  — trained weights only
  - ``state.pt``           — optimizer + scheduler + RNG state

The exact-answer eval CLI (``finpost.evals.eval_exact``) loads checkpoints
via ``finpost.safety.safe_load_model`` and ``safe_load_tokenizer``, which
both expect a Hugging Face-format directory containing at minimum:

  - ``config.json``
  - tokenizer files (``tokenizer.json`` etc.)
  - ``model.safetensors`` (or ``pytorch_model.bin``)

This script bridges the two. It loads the base model architecture and
tokenizer from the Hub, swaps in the trained weights from
``model.safetensors``, then writes the result to a new directory via
``save_pretrained``. The output is a self-contained HF directory ready
for the eval CLI.

Usage::

    python scripts/convert_checkpoint_to_hf.py \\
        --checkpoint-dir results/checkpoints/qwen-gsm8k_only-1000s/step-00001000 \\
        --out-dir results/checkpoints/qwen-gsm8k_only-1000s-hf
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from safetensors.torch import load_file

from finpost.safety import safe_load_model, safe_load_tokenizer


def convert(
    checkpoint_dir: Path,
    base_model_id: str,
    out_dir: Path,
    dtype: str = "float32",
) -> None:
    """Convert a finpost training checkpoint to HF format.

    Parameters
    ----------
    checkpoint_dir
        Path to a ``step-XXXXXXXX`` directory containing ``model.safetensors``.
    base_model_id
        Hugging Face Hub id of the architecture (e.g. ``"Qwen/Qwen2.5-0.5B"``).
        Must match the architecture that was trained.
    out_dir
        Destination directory. Created if missing.
    dtype
        Compute dtype to load the base architecture as. Should match what
        the training run used (the saved weights are stored at training
        precision and will load into whichever dtype the model is in).
    """
    weights_path = checkpoint_dir / "model.safetensors"
    if not weights_path.exists():
        raise FileNotFoundError(
            f"No model.safetensors at {weights_path}. "
            "Pass the step-XXXXXXXX directory, not the run root."
        )

    torch_dtype = getattr(torch, dtype)
    print(f"[convert] loading base architecture: {base_model_id} ({dtype})")
    model = safe_load_model(base_model_id, dtype=torch_dtype)
    tokenizer = safe_load_tokenizer(base_model_id)

    print(f"[convert] loading trained weights from {weights_path}")
    state_dict = load_file(str(weights_path))
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        print(
            f"[convert] WARNING: {len(missing)} missing keys "
            f"(first 3: {missing[:3]})"
        )
    if unexpected:
        print(
            f"[convert] WARNING: {len(unexpected)} unexpected keys "
            f"(first 3: {unexpected[:3]})"
        )

    out_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out_dir)
    tokenizer.save_pretrained(out_dir)
    print(f"[convert] wrote HF-format checkpoint to {out_dir}")
    print(f"[convert] contents: {sorted(p.name for p in out_dir.iterdir())}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert a finpost training checkpoint to HF format."
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        required=True,
        help=(
            "Path to a step-XXXXXXXX directory containing model.safetensors "
            "(NOT the run-root directory)."
        ),
    )
    parser.add_argument(
        "--base-model-id",
        type=str,
        default="Qwen/Qwen2.5-0.5B",
        help="HF Hub id of the base architecture. Default: %(default)s",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        required=True,
        help="Destination directory for the HF-format checkpoint.",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="float32",
        choices=["float32", "bfloat16", "float16"],
        help="Compute dtype for loading. Default: %(default)s",
    )
    args = parser.parse_args()
    convert(
        checkpoint_dir=args.checkpoint_dir,
        base_model_id=args.base_model_id,
        out_dir=args.out_dir,
        dtype=args.dtype,
    )


if __name__ == "__main__":
    main()
