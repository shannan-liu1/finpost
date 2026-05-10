"""Local SFT training smoke test on real GSM8K data with packing.

End-to-end wire-up of the Phase 1 SFT stack:

  - tiny-gpt2 (~1MB, CPU-friendly placeholder model)
  - real GSM8K examples loaded via PhasedSFTDataset
  - PackingCollator: prompt/response serialization, tokenization,
    prompt-token masking, padding, and (optional) document packing
  - masked cross-entropy loss in finpost.training.sft

Why this exists alongside scripts/sft_smoke.py:
  sft_smoke.py    — 3 hand-written QA pairs, no packing, no real
                    dataset. Tests only the loss path. Runs in seconds.
  this script     — real GSM8K data, real packing collator, real
                    DataLoader. Tests the full data pipeline.

The success criterion is the same in both: loss is finite and
decreases over a small number of steps. If that holds here, the
Phase 1 trainer plumbing is correct end-to-end and we can promote
the same code path to Colab with Qwen 2.5-0.5B on a real GPU.

Usage:
    python scripts/sft_local_train.py --steps 10
"""

from __future__ import annotations

import argparse
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from finpost.training.config import (
    Config,
    DataConfig,
    ModelConfig,
    PackingConfig,
    TrainingConfig,
)
from finpost.training.dataset import make_loaders
from finpost.training.sft import compute_masked_ce_loss

# tiny-gpt2: ~1MB GPT-2 with reduced layers/dim. Ships with the
# HuggingFace test suite. No auth, downloads in seconds, runs on CPU.
# Same identifier loads both the tokenizer and the model weights.
_TINY_MODEL = "sshleifer/tiny-gpt2"


def build_config(batch_size: int, max_seq_len: int) -> Config:
    """Build a smoke-test Config programmatically (no YAML on disk).

    We construct the Config object directly in Python instead of
    loading a YAML for two reasons:

      1. Self-contained — running the script requires no extra files.
      2. Pedagogical — every training knob is visible inline, with
         a comment next to it explaining why it has the value it does.

    For real training runs, use Config.from_yaml() with a versioned
    YAML config so you can reproduce a run by replaying its config.
    """
    return Config(
        model=ModelConfig(
            base_model_id=_TINY_MODEL,
            # float32 on CPU. bfloat16 is technically supported on
            # modern CPUs (AVX512_BF16) but PyTorch's CPU bf16 kernels
            # are slow; for a tiny model the memory savings are not
            # worth the throughput hit.
            dtype="float32",
            # tiny-gpt2 ships only the legacy pickle (.bin) weights;
            # it has no safetensors variant. Pickle weights are a
            # security risk for untrusted models — but tiny-gpt2 is
            # a known HF artifact, so we accept .bin here. The real
            # Phase 1 Qwen model has safetensors and we use them.
            use_safetensors=False,
        ),
        data=DataConfig(
            # Phase 1 starts with GSM8K only. MATH gets layered in
            # later as a curriculum step.
            sources=["gsm8k"],
            # 5% held out as validation, stratified by source. The
            # split is deterministic given the seed, so val/train
            # never overlap across runs of this script.
            val_split_pct=5.0,
            seed=42,
        ),
        training=TrainingConfig(
            # Required by the schema. We override the actual loop
            # length with --steps; the value here just needs to be
            # large enough to satisfy the warmup_steps < max_steps
            # cross-field validator.
            max_steps=10_000,
            warmup_steps=10,
            lr=1e-4,
            per_device_batch_size=batch_size,
        ),
        packing=PackingConfig(
            max_seq_len=max_seq_len,
            # Cross-document attention isolation requires the 4D
            # attention mask path. Most HF models accept it, but the
            # interaction with their internal causal-mask construction
            # is subtle and varies by model. For this smoke test we
            # use the simpler 2D padding mask (isolate_documents=False)
            # and accept that response tokens of document B in a
            # packed row can attend to document A.
            #
            # Why this is acceptable here:
            #   - loss is still masked on prompt tokens (no spurious
            #     gradient flows through the leakage),
            #   - the EOS separator gives a strong "new document"
            #     signal in the input stream,
            #   - the goal is verifying loss decreases, which it does
            #     even with imperfect packing.
            #
            # For production training, set isolate_documents=True and
            # wire the 4D mask into the trainer carefully.
            isolate_documents=False,
        ),
    )


def train_packed_step(
    model: torch.nn.Module,
    batch: dict[str, Any],
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    """One training step over a packed batch from the collator.

    Differs from finpost.training.sft.train_step in one critical way:
    that helper takes only (input_ids, labels) and never passes an
    attention_mask. That works for the unpacked smoke test where every
    sequence is the same length and there is no padding. With packing,
    rows of different effective lengths are padded to a common width,
    so we MUST pass the attention_mask — without it, the model would
    attend to padding positions as if they were real content and the
    loss would be wrong.
    """
    # batch is a dict of CPU tensors. We move each tensor to the
    # training device individually because dicts have no .to() method
    # that recurses. .to(device) is a no-op when the tensor is already
    # on the target device.
    input_ids = batch["input_ids"].to(device)
    labels = batch["labels"].to(device)
    attention_mask = batch["attention_mask"].to(device)

    # zero_grad: reset .grad on every parameter to zero. PyTorch
    # accumulates gradients across successive .backward() calls — that
    # is desirable for gradient accumulation but a bug for a vanilla
    # step. Always zero before computing fresh gradients.
    optimizer.zero_grad()

    # Forward pass.
    #   input_ids:       (batch, seq_len) integer token IDs to embed
    #   attention_mask:  (batch, seq_len) 1 = real, 0 = padding
    # The model returns a ModelOutput whose .logits has shape
    # (batch, seq_len, vocab_size). Note we deliberately do NOT pass
    # labels= to the model — that would trigger HF's internal loss
    # path, which knows nothing about our prompt masking. We compute
    # loss ourselves below using the labels tensor the collator built.
    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
    )

    # compute_masked_ce_loss does three things at once:
    #   1. shifts logits and labels by one position (the logit at
    #      position t predicts the token at position t+1),
    #   2. flattens (batch, seq_len-1, vocab_size) to a 2D matrix
    #      so cross_entropy treats each (batch, position) as one
    #      independent classification problem,
    #   3. excludes IGNORE_INDEX (-100) positions from both the loss
    #      sum and the mean denominator — so prompt and padding
    #      tokens contribute zero gradient.
    loss = compute_masked_ce_loss(outputs.logits, labels)

    # backward: walks the autograd graph from `loss` back through
    # every operation that produced it, accumulating gradients into
    # each parameter's .grad tensor via chain rule.
    loss.backward()

    # step: applies the AdamW update rule. For each parameter:
    #   - update first/second moment estimates of the gradient,
    #   - bias-correct them,
    #   - subtract lr * (corrected_first / sqrt(corrected_second + eps)),
    #   - apply decoupled weight decay.
    optimizer.step()

    # .item() pulls the Python float off the 0-dim loss tensor. This
    # also detaches from the autograd graph, so we can log it
    # without holding references that prevent graph deallocation.
    return loss.item()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Local SFT smoke test on real GSM8K with packing.",
    )
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument(
        "--batch-size",
        type=int,
        default=2,
        help="Packed rows per step. CPU is happy with 1-2; T4 GPU 4-8.",
    )
    parser.add_argument(
        "--max-seq-len",
        type=int,
        default=1024,
        help="Token cap per packed row. tiny-gpt2 max position is 1024.",
    )
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    args = parser.parse_args()

    device = torch.device(args.device)
    config = build_config(batch_size=args.batch_size, max_seq_len=args.max_seq_len)

    print(f"Model:   {config.model.base_model_id}")
    print(f"Device:  {device}")
    print(f"Steps:   {args.steps}")
    print(f"Batch:   {config.training.per_device_batch_size}")
    print(f"MaxLen:  {config.packing.max_seq_len}")

    print("\nLoading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(config.model.base_model_id)

    # GPT-style tokenizers usually do not define a pad token because
    # GPT was trained without padding. Reuse EOS as the pad token —
    # standard convention. The attention mask the collator produces
    # tells the model which positions are padding, so the actual
    # token id sitting at padding positions never matters.
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("Loading model...")
    dtype = getattr(torch, config.model.dtype)
    model = AutoModelForCausalLM.from_pretrained(
        config.model.base_model_id,
        dtype=dtype,
        use_safetensors=config.model.use_safetensors,
    ).to(device)
    # train() flips dropout / batchnorm-ish layers into training mode.
    # tiny-gpt2 has attention and residual dropout; both are no-ops
    # if model.eval() were called instead.
    model.train()

    print("Building data loaders (this loads GSM8K)...")
    # make_loaders performs, in order:
    #   1. PhasedSFTDataset(split="train") — loads GSM8K train, drops
    #      val_split_pct% as held-out val using a deterministic
    #      seeded shuffle stratified by source.
    #   2. PhasedSFTDataset(split="val")   — same deterministic
    #      shuffle, but takes the val portion instead.
    #   3. Wraps both in DataLoader with PackingCollator as
    #      collate_fn. The collator runs in this process here
    #      (num_workers=0). For real GPU runs you'd parallelize.
    train_loader, _val_loader = make_loaders(config, tokenizer)

    print(f"Train examples (post-split): {len(train_loader.dataset)}\n")

    # AdamW = Adam with decoupled weight decay (Loshchilov & Hutter,
    # 2019). The default optimizer for transformer fine-tuning. lr=1e-4
    # is a mid-range fine-tuning choice; smaller models tolerate higher
    # LR but 1e-4 is safe for tiny-gpt2.
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.training.lr,
        weight_decay=config.training.weight_decay,
    )

    print(f"Running {args.steps} steps:\n")
    losses: list[float] = []

    # Manual iterator instead of `for batch in train_loader`: we want
    # to stop after exactly --steps batches regardless of how many
    # batches the loader actually has, AND we want to wrap around if
    # the loader runs out before we hit our step count.
    loader_iter = iter(train_loader)

    for step in range(args.steps):
        try:
            batch = next(loader_iter)
        except StopIteration:
            # Loader exhausted — start a new epoch. For a real
            # training run you'd track epoch counts and re-seed the
            # generator; for a smoke test we just loop.
            loader_iter = iter(train_loader)
            batch = next(loader_iter)

        loss = train_packed_step(model, batch, optimizer, device)
        losses.append(loss)

        # Log the packing geometry so we can see the collator at work.
        # IGNORE_INDEX = -100; response tokens are everything else.
        rows, seq_len = batch["input_ids"].shape
        response_tokens = (batch["labels"] != -100).sum().item()
        print(
            f"  step {step:2d}: loss={loss:.4f}  "
            f"shape=({rows}x{seq_len})  resp_tokens={response_tokens}"
        )

    print()
    # Sanity check: any NaN or inf is a hard fail. Common causes are
    # bad dtype (fp16 underflow), exploded gradients, or a bug in the
    # mask that ignores ALL positions and divides by zero.
    if any(not torch.isfinite(torch.tensor(v)).item() for v in losses):
        print(f"FAIL: NaN or inf loss in {losses}")
        return

    if losses[-1] < losses[0]:
        drop = losses[0] - losses[-1]
        print(f"OK: loss {losses[0]:.4f} -> {losses[-1]:.4f} (drop {drop:.4f})")
    else:
        print(
            f"WARN: loss did not decrease over {args.steps} steps "
            f"({losses[0]:.4f} -> {losses[-1]:.4f}). "
            f"Try more steps, higher --lr, or a larger model."
        )


if __name__ == "__main__":
    main()
