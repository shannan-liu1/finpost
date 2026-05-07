# Primer: where training memory actually goes

A reference for "why does training a 1-billion-parameter model need an A100 when inference fits on a phone?" Written so a future reader (you, in three months) can re-derive the budget without help.

This primer covers four interlocking topics:

1. **The bytes-per-parameter accounting** during full fine-tuning.
2. **AdamW optimizer state** — what the "first moment" and "second moment" actually are, and why they cost what they cost.
3. **Activation memory** — the part that scales with batch size and sequence length.
4. **The unlocks** — FlashAttention (free), gradient checkpointing (cheap), 8-bit optimizers, LoRA, QLoRA.

Worked example throughout: Gemma 3 1B, mixed-precision training (bf16 forward, fp32 master copy), AdamW.

---

## 1. The five copies of every parameter

Full fine-tuning under mixed precision keeps **five distinct tensors per parameter** in GPU memory at all times. This is the part most people don't internalize.

| Copy | Precision | Bytes per parameter | Why it exists |
|------|-----------|---------------------|---------------|
| **bf16 weight** (used in compute) | bf16 | 2 | The actual weight matrix that participates in forward and backward. Bf16 is fast on modern GPUs (Tensor Cores) and uses half the memory of fp32. |
| **fp32 master weight** | fp32 | 4 | A high-precision copy that is the "ground truth" the optimizer updates. Required because tiny updates (η × g ≈ 1e-8) get lost in bf16's limited mantissa. The bf16 copy is reconstituted from the fp32 master each step. |
| **gradient** | bf16 | 2 | Computed during backward. Stored at the same precision as the bf16 weights. |
| **AdamW first moment (m)** | fp32 | 4 | Running average of past gradients. See section 2. |
| **AdamW second moment (v)** | fp32 | 4 | Running average of past squared gradients. See section 2. |
| **Total** | | **16 bytes per parameter** | |

For 1 billion parameters: **16 GB of memory**, *before* you have processed a single training example.

### Why fp32 for the master weight and the optimizer state

Two related issues:

- **Limited bf16 precision.** Bf16 has 7 mantissa bits — about 3 decimal digits of precision around any given value. A weight of magnitude 1.0 cannot represent updates smaller than ~1e-3 reliably. With learning rate 1e-5 and gradient 1e-3, the per-step update is 1e-8 — far below bf16's resolution. The update would round to zero. The fp32 master copy lets updates accumulate at full precision; only the bf16 view used for compute is lossy.
- **Long-horizon accumulation.** The first and second moments are exponentially-weighted averages over thousands of steps. Tiny errors per step would compound. Fp32 keeps them faithful.

### What changes if you skip the fp32 master copy

"Pure bf16" training drops the 4-byte master copy. Saves a quarter of the per-parameter cost. Risk: training instability — loss spikes, NaN gradients, the optimizer wandering. Some teams do this; for a learning project the conservative default of mixed precision with fp32 master copy is the right call.

---

## 2. The optimizer state — first moment, second moment, and "wait, isn't that just momentum?"

This is the part that confuses everyone the first time.

### The math

AdamW maintains two running averages per parameter, updated each step:

```
m_t = β₁ · m_{t-1} + (1 − β₁) · g_t            [first moment]
v_t = β₂ · v_{t-1} + (1 − β₂) · g_t²           [second moment]
```

Where `g_t` is the gradient at step `t`, and `β₁` and `β₂` are decay rates (typical defaults: β₁ = 0.9, β₂ = 0.999).

Then the parameter update is (with `m̂` and `v̂` being bias-corrected versions of m and v):

```
θ_t = θ_{t-1} − η · m̂_t / (sqrt(v̂_t) + ε)
```

### What "first moment" and "second moment" mean

The terms come from probability theory.

For a random variable `X`:

- The **first moment** is `E[X]` — its expected value, the mean.
- The **second moment** is `E[X²]` — the expected value of its square. (The variance is `E[X²] − (E[X])²` — a function of both moments.)

When applied to gradients: `m_t` is an estimate of the **mean of recent gradients**, and `v_t` is an estimate of the **mean of recent squared gradients** (which is closely related to gradient variance).

### Yes, the first moment IS momentum — same object, different name

Classical SGD-with-momentum maintains:

```
u_t = μ · u_{t-1} + g_t                        [the "velocity" / "momentum" buffer]
θ_t = θ_{t-1} − η · u_t
```

This `u_t` is **operationally identical** to AdamW's `m_t` (modulo the `(1 − β₁)` scaling factor, which is just a normalization choice). Both are exponentially-weighted moving averages of past gradients. Both serve the same purpose: smooth out gradient noise and let the optimizer "carry" useful direction across steps the way physical momentum carries a moving object.

The naming difference is purely historical and disciplinary:

- **"Momentum"** comes from the physics analogy. SGD with momentum was framed as: gradients are forces, weights have inertia, the velocity is what gets updated.
- **"First moment"** comes from the statistical framing. Adam was sold as "estimate the first and second moments of the gradient distribution and use them to construct an adaptive update."

They are the same concept under two different names. AdamW uses the statistical naming because it ALSO maintains a second moment — and the physics analogy doesn't extend cleanly to "the variance of the velocity."

### Why this costs 8 bytes per parameter

`m` and `v` are both stored, both in fp32, both per parameter:

- m: 4 bytes per parameter
- v: 4 bytes per parameter
- **Total: 8 bytes per parameter for AdamW state.**

For 1 billion parameters: **8 GB** of optimizer state. Notice this is **larger than the bf16 model itself** (which is 2 GB). The optimizer's bookkeeping is the dominant memory cost of training.

This is the single most important number to internalize. **The optimizer state, not the model, is what makes training expensive.**

### Why not store m and v in bf16 too?

You can. Bitsandbytes' "8-bit Adam" stores them in 8-bit (with block-wise quantization to preserve dynamic range), cutting their cost from 8 bytes per parameter to 2 bytes per parameter. Quality is empirically very close to fp32 AdamW for most tasks. This is a free-ish win — and it's why we use `paged_adamw_8bit` in Phase 2's QLoRA configuration.

---

## 3. Activation memory — the part that scales with batch and sequence length

The 16 bytes per parameter from sections 1 and 2 is the **constant** cost. It doesn't depend on batch size, sequence length, or what data you're training on. On top of that, you pay for activations.

### What activations are

When the model does a forward pass, every layer produces intermediate tensors. These are needed during the backward pass for the chain rule — specifically, the gradient at one layer requires the activation values from that layer. Without saving them, you cannot compute gradients without recomputing them.

For a single transformer layer at sequence length `L`, batch size `B`, hidden dimension `H`, intermediate dimension `H_int` (typically 4·H or so), the activations include roughly:

- The layer input: `B × L × H` floats
- The attention output (post-projection): `B × L × H` floats
- The MLP intermediate: `B × L × H_int` floats
- The MLP output: `B × L × H` floats
- Plus several smaller tensors (LayerNorm intermediates, residual sums)

In bf16, each float is 2 bytes. Roughly: ~10 × B × L × H bytes per layer (very approximate).

### Worked estimate for Gemma 3 1B

[NEEDS VERIFICATION on Gemma 3 1B's exact architecture; rough numbers below]

Assume: 26 layers, hidden_dim ≈ 1500, intermediate_dim ≈ 6000.

Per layer at batch=1, seq=4096:
- Hidden-dim activations: ~10 × 1 × 4096 × 1500 × 2 bytes = ~120 MB
- (For comparison: attention scores without FlashAttention would add 1 × 32 heads × 4096² × 2 bytes ≈ 1 GB *per layer*. This is why FlashAttention isn't optional.)

Across 26 layers: ~3 GB of activations at batch=1, seq=4096.

Scales linearly with batch size and sequence length:
- Batch 8, seq 4096: ~24 GB
- Batch 1, seq 16384: ~12 GB
- Batch 8, seq 16384: ~96 GB (well over an A100 80GB)

### Why FlashAttention is mandatory

Vanilla attention computes Q·Kᵀ first, then softmax, then multiplies by V. The intermediate Q·Kᵀ is shape `[B, heads, L, L]` — quadratic in sequence length. At L = 4096 with 32 heads at batch 1: 1 GB per layer. Across 26 layers: 26 GB just for one attention matrix per layer per example.

FlashAttention restructures the computation:

- Tile the attention computation into blocks that fit in GPU SRAM (the small, fast on-chip memory near the compute units).
- Within each block, compute Q·Kᵀ → softmax → · V locally and accumulate the result.
- Maintain running max and running sum to compute softmax incrementally without materializing the full row.
- The full Q·Kᵀ matrix is **never written to HBM** (the slow, large GPU memory).

Result: same numerical output (within floating-point tolerance), **O(L) memory** instead of O(L²), **and faster** because the tiled compute pattern is dramatically more cache-friendly.

This is the "always on" baseline. `transformers` enables it automatically when available. If you're not using it, you're paying memory and time for nothing.

### Gradient checkpointing — recompute instead of store

Even with FlashAttention, you still store hidden-dim activations at every layer. Gradient checkpointing trades compute for memory: keep only the *layer inputs* (one set per layer), not the per-layer intermediates. During the backward pass, recompute the intermediates by running the forward again on each layer.

Tradeoff: ~30–50% reduction in activation memory, ~25% extra training time.

Use it when memory is the constraint. Skip it when you have headroom.

---

## 4. The full picture — putting numbers on it

Full fine-tune of Gemma 3 1B, mixed precision, AdamW, FlashAttention, no gradient checkpointing:

| Component | Per-parameter cost | For 1B params | Notes |
|-----------|-------------------|---------------|-------|
| bf16 weights | 2 B | 2 GB | |
| fp32 master weights | 4 B | 4 GB | |
| bf16 gradients | 2 B | 2 GB | |
| AdamW first moment (fp32) | 4 B | 4 GB | |
| AdamW second moment (fp32) | 4 B | 4 GB | |
| **Static training memory** | **16 B** | **16 GB** | |
| Activations, batch=1, seq=4K | — | ~3 GB | scales linearly with B and L |
| Activations, batch=8, seq=4K | — | ~24 GB | |
| Workspace, fragmentation | — | ~3 GB | rough overhead |
| **Total at batch=8, seq=4K** | | **~43 GB** | tight on A100 40GB, fine on 80GB |

The pattern: **the constant cost (params + grads + optimizer = 16 GB) is roughly equal to the variable cost (activations) at moderate batch sizes.** Both matter. Either one alone fits on a smaller GPU; together they push you to A100-class hardware.

---

## 5. The unlocks — what each one buys you

### LoRA (Low-Rank Adaptation)

Replace the full-rank update `ΔW` to a weight matrix `W` with a low-rank product `B · A`, where `A` and `B` are small matrices (rank `r` ≪ original dimensions). Train only `A` and `B`; freeze `W`.

For a typical setup, the trainable parameter count drops from 1 billion to ~10 million (0.1–1%, depending on rank and which layers get adapters).

**Memory savings:** the 12 bytes per parameter that would have gone to gradients + AdamW state + fp32 master copy ONLY apply to the trainable adapter parameters. The 988 million frozen parameters store only their bf16 weight (no gradient, no optimizer state). Roughly: optimizer-related memory drops by ~99% for these layers.

**What you give up:** the update is constrained to a low-rank subspace. Empirically near-equivalent to full fine-tuning for most tasks; in some cases produces measurably worse results, especially at very small rank or for tasks that require broad capability shifts. For Direct Preference Optimization specifically, LoRA can be subtly tricky — see Phase 2 notes in the project plan.

### QLoRA — quantize the frozen base on top of LoRA

Stack three things:

1. **Base model in 4-bit** (NF4 format from bitsandbytes). The 988 million frozen parameters now use 0.5 bytes each instead of 2 bytes, dropping base model memory from 2 GB to 0.5 GB.
2. **LoRA adapters in bf16.** As above.
3. **8-bit AdamW for the adapters.** Optimizer state for the adapters drops by 4× on top of the LoRA savings.

How the math works during forward pass: the 4-bit base weights are dequantized to bf16 *on the fly* per matrix multiplication, used in compute, then thrown away. The dequantized version never persists in memory. The trick is that dequantization is fast (specialized kernels) and the bandwidth saved is enormous.

**Memory budget for Gemma 3 1B with rank-16 QLoRA on attention + MLP:**

- Base model in 4-bit: ~0.5 GB
- LoRA adapter weights (bf16): ~20 MB
- LoRA gradients (bf16): ~20 MB
- LoRA fp32 master: ~40 MB
- LoRA AdamW state in 8-bit: ~20 MB
- **Static training memory: ~600 MB**
- Activations at batch=1, seq=8K: ~5 GB (still need full activations through all layers — the base being quantized doesn't reduce the activation count)
- **Total: ~6 GB at batch=1, seq=8K**

This fits comfortably on a 24 GB consumer card, with substantial headroom for longer context or larger batch.

### When to use what

| Scenario | Recommended setup |
|----------|-------------------|
| Learning the substrate of optimization | Full fine-tune. Pay the A100 tax for the pedagogy. |
| Production small-model fine-tuning | LoRA. Faster iteration, comparable quality. |
| Memory-constrained, want long context | QLoRA. The dominant approach for 7B+ on consumer hardware. |
| Best possible quality, compute is unconstrained | Full fine-tune. |

---

## Glossary recap

- **First moment (m)** = running mean of past gradients = "momentum." Same object, two names from different traditions.
- **Second moment (v)** = running mean of past squared gradients. Used to scale per-parameter learning rates adaptively (parameters with high gradient variance get smaller effective updates).
- **Optimizer state** = m + v + (in mixed precision) the fp32 master weight. The dominant memory cost of training, larger than the model itself.
- **Activations** = forward-pass intermediates retained for the backward pass. Scale with batch size, sequence length, and model depth.
- **FlashAttention** = an attention implementation that never materializes the full Q·Kᵀ matrix. Always-on baseline.
- **Gradient checkpointing** = trade compute for memory by recomputing activations during backward. ~30–50% activation memory savings, ~25% slowdown.
- **LoRA** = train only low-rank adapter matrices on top of a frozen base. Cuts trainable-parameter count by ~99%.
- **QLoRA** = LoRA + 4-bit quantized frozen base + 8-bit optimizer. Fits 1B–7B-class training on consumer hardware.
