"""Smoke test for the Supervised Fine-Tuning trainer.

Loads a model, builds a tiny hand-written batch of (prompt, response)
pairs, runs ``--steps`` training steps, prints the loss after each.
The point is to verify three things end-to-end:

1. The trainer runs without exception.
2. Loss values are finite (not NaN, not inf).
3. Loss decreases over a small number of steps on this toy data.

By default uses ``sshleifer/tiny-gpt2`` (~1MB, no auth required) so
the smoke test runs in seconds on CPU. Pass ``--no-tiny-model`` to
swap in the Phase 1 Qwen 0.5B base model.

Usage:
    python scripts/sft_smoke.py --tiny-model --device cpu
"""

from __future__ import annotations

import argparse

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from finpost.training.masking import mask_prompt_tokens
from finpost.training.sft import train_step

# ~1MB GPT-2 with reduced layers/dim; ships with the Hugging Face test
# suite. Exists specifically so smoke tests like this one have a
# downloadable, license-free CausalLM to load on CPU in seconds.
_TINY_MODEL = "sshleifer/tiny-gpt2"
_REAL_MODEL = "Qwen/Qwen2.5-0.5B"


def main() -> None:
    parser = argparse.ArgumentParser(description="SFT trainer smoke test.")
    parser.add_argument(
        "--tiny-model",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=f"Use {_TINY_MODEL} (~1MB, no auth). Pass --no-tiny-model for {_REAL_MODEL}.",
    )
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument(
        "--dtype",
        choices=("float32", "bfloat16"),
        default=None,
        help="Defaults: float32 on CPU, bfloat16 on CUDA.",
    )
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-4)
    args = parser.parse_args()

    # Default dtype based on device. CPU bf16 is supported but slow and
    # rarely worth it; float32 is the right default there.
    if args.dtype is not None:
        dtype = getattr(torch, args.dtype)
    elif args.device == "cuda":
        dtype = torch.bfloat16
    else:
        dtype = torch.float32

    model_name = _TINY_MODEL if args.tiny_model else _REAL_MODEL
    print(f"Model:  {model_name}")
    print(f"Device: {args.device}")
    print(f"Dtype:  {dtype}")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    # Tokenizers without an explicit pad token cannot batch with padding.
    # Reuse the EOS token as the pad — standard convention. The
    # attention_mask we build below tells the model where padding is.
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # For the real Phase 1 model, enforce safetensors per SECURITY.md.
    # Older toy models (sshleifer/tiny-gpt2) ship as .bin and don't
    # have safetensors weights — don't enforce there or the load fails.
    load_kwargs = {"dtype": dtype}
    if not args.tiny_model:
        load_kwargs["use_safetensors"] = True

    model = AutoModelForCausalLM.from_pretrained(model_name, **load_kwargs).to(args.device)
    model.train()

    # Three trivial QA pairs. The model should overfit them within a
    # few steps; that's what makes this a meaningful smoke test (loss
    # should noticeably drop).
    examples = [
        ("What is 2+2?", "The answer is 4."),
        ("What is 3*5?", "The answer is 15."),
        ("What is 10-7?", "The answer is 3."),
    ]
    full_texts = [f"{p} {r}" for p, r in examples]
    prompt_texts = [f"{p} " for p, _ in examples]

    # Tokenize the FULL sequences (prompt + response) batched with padding.
    encoded = tokenizer(full_texts, return_tensors="pt", padding=True)
    input_ids = encoded["input_ids"].to(args.device)
    attention_mask = encoded["attention_mask"].to(args.device)

    # Tokenize prompts only to find each example's prompt length.
    # add_special_tokens=False because the full-sequence tokenization
    # is what determines special-token placement; we only want the
    # prompt's content-token count for the mask boundary.
    prompt_lengths = torch.tensor(
        [len(tokenizer(p, add_special_tokens=False)["input_ids"]) for p in prompt_texts],
        device=args.device,
    )

    labels = mask_prompt_tokens(input_ids, prompt_lengths, attention_mask)

    print(f"Batch shape:    {tuple(input_ids.shape)}")
    print(f"Prompt lengths: {prompt_lengths.tolist()}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    print(f"\nRunning {args.steps} training steps:")
    losses: list[float] = []
    for step in range(args.steps):
        loss = train_step(model, input_ids, labels, optimizer)
        losses.append(loss)
        print(f"  step {step}: loss = {loss:.4f}")

    print()
    if any(not torch.isfinite(torch.tensor(loss_value)).item() for loss_value in losses):
        print(f"FAIL: at least one loss is NaN or inf: {losses}")
        return

    if losses[-1] < losses[0]:
        delta = losses[0] - losses[-1]
        print(f"OK: loss decreased {losses[0]:.4f} -> {losses[-1]:.4f} (drop {delta:.4f})")
    else:
        print(
            f"WARN: loss did NOT decrease over {args.steps} steps "
            f"({losses[0]:.4f} -> {losses[-1]:.4f}). Try more steps or higher --lr."
        )


if __name__ == "__main__":
    main()
