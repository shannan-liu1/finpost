"""One-shot script to add the A40 throughput levers subsection to STUDY.html.

Inserts a new <h3>...</h3> block at the end of section 11 (Hardware and
memory) covering: today's applied recipe, the memory analysis for bs=64
training and bs=256 eval (the answer to whether they fit), the pending
trainer-side improvements with expected impact and risk, a skip list,
and primary sources.

Run once from the repo root, then delete the script.
"""
from __future__ import annotations

import re
from pathlib import Path

STUDY = Path("STUDY.html")

NEW_BLOCK = r"""
<h3>A40 throughput levers &mdash; tomorrow's recipe and what comes next</h3>

<p>This is the running tally for the Phase 1 A40 run. Everything in the <em>applied today</em> block is in the RunPod ablation notebook on disk and validated end-to-end. The <em>pending</em> table is research-backed but requires trainer or eval-CLI changes outside the notebook and is not yet wired up.</p>

<div class="callout">
  <div class="callout-label">Recipe applied today (notebook only)</div>
  <ul style="margin: 0.4rem 0 0; padding-left: 1.2rem;">
    <li><strong>Dtype:</strong> <code>bfloat16</code> throughout. A40 is Ampere, has native bf16, no FP8. Cuts memory ~2&times; vs fp32 and ~1.5&times; throughput at no measured accuracy cost on supervised fine-tuning.</li>
    <li><strong>Per-device micro-batch:</strong> 16 (was 4). At 0.5B parameters on a 48&nbsp;GB card the GPU was idle at batch 4; raising it to 16 raises utilisation. Effective batch under greedy packing at <code>max_seq_len=1024</code> is about 48 documents per optimizer step.</li>
    <li><strong>Packed sequence length:</strong> 1024 (was 512). Covers the MATH train-set p95 of 784 tokens of <code>prompt + response</code>. At 512, roughly 15% of MATH documents had their <code>\boxed{...}</code> answer truncated.</li>
    <li><strong>Eval batch:</strong> 128 for both GSM8K and MATH (was 32). Memory math: model 1&nbsp;GB + key/value cache 128&times;12&nbsp;MB = 1.5&nbsp;GB + transients ~1&nbsp;GB = ~4&nbsp;GB of 48. The eval command-line interface's <code>_generate_chunk_with_oom_fallback</code> halves transparently if a hard chunk blows up.</li>
    <li><strong>Checkpoint cadence:</strong> every 500 steps. <code>INTERMEDIATE_STEPS=[500, 1000, 1500, 2000]</code> and <code>retention_last_n=4</code> so the eval cell can score the full trajectory (12 (arm, step) rows instead of 3 final-only).</li>
    <li><strong>Validation cadence:</strong> every 500 steps, aligned to checkpoint cadence so <code>val_loss</code> curves line up with eval points.</li>
  </ul>
</div>

<h4>Can the batches go even higher? &mdash; batch 64 training, batch 256 eval</h4>

<p>The memory budget says yes; the throughput budget says &ldquo;depends on the collator.&rdquo; First-principles count for one optimizer step at <code>seq_len=1024</code>, <code>bfloat16</code>, with the current trainer that builds a 4D per-document attention mask:</p>

<table>
  <thead>
    <tr>
      <th>Memory component</th>
      <th>bs=16 (today)</th>
      <th>bs=32</th>
      <th>bs=64 (asked)</th>
    </tr>
  </thead>
  <tbody>
    <tr><td>Model weights (0.5B &times; bf16)</td><td>1.0</td><td>1.0</td><td>1.0</td></tr>
    <tr><td>Gradients (bf16)</td><td>1.0</td><td>1.0</td><td>1.0</td></tr>
    <tr><td>AdamW state (fp32 first + second moments)</td><td>4.0</td><td>4.0</td><td>4.0</td></tr>
    <tr><td>Saved activations (~4 buffers &times; 24 layers)</td><td>3.6</td><td>7.2</td><td>14.4</td></tr>
    <tr><td>Attention scores (peak, transient)</td><td>0.5</td><td>0.9</td><td>1.8</td></tr>
    <tr><td>4D document-isolation mask (on-GPU bool)</td><td>0.02</td><td>0.04</td><td>0.06</td></tr>
    <tr><td><strong>Subtotal (GB)</strong></td><td><strong>~10</strong></td><td><strong>~14</strong></td><td><strong>~22</strong></td></tr>
  </tbody>
</table>

<p>All three fit comfortably in 48&nbsp;GB. The constraint is not VRAM &mdash; it is the collator. The packing collator builds the 4D mask on CPU, in the main thread (<code>num_workers=0</code>), and the cost scales with <code>bs &times; seq &times; seq &times; 8 bytes</code> (int64 intermediate). Concretely:</p>

<ul>
  <li>bs=16, seq=1024 &rarr; 128&nbsp;MB CPU work per batch</li>
  <li>bs=32, seq=1024 &rarr; 256&nbsp;MB per batch</li>
  <li>bs=64, seq=1024 &rarr; <strong>512&nbsp;MB per batch</strong></li>
</ul>

<p>At batch 64, the main process spends a non-trivial fraction of each step rebuilding the mask between batches. The GPU sits idle during that time. So &ldquo;memory says it fits&rdquo; is the easy part; throughput is gated on either letting the DataLoader use workers (<code>num_workers&ge;2</code>, which moves the collator off the main thread) or replacing the 4D mask with FlashAttention&nbsp;2 variable-length packed-attention (which has no 4D mask at all). Both are pending trainer changes &mdash; see below.</p>

<p>For eval, the picture is simpler. At batch 256 with max context ~1024 tokens, memory is model (~1&nbsp;GB) + key/value cache (256 &times; ~12&nbsp;MB &asymp; 3&nbsp;GB; Qwen2.5 uses grouped-query attention with two key/value heads, so the cache stays small) + transients ~1&nbsp;GB &asymp; <strong>5&nbsp;GB of 48</strong>. Memory is not the constraint. The constraint is HuggingFace <code>generate()</code> static batching: every sequence in the batch keeps running until the slowest one emits the end-of-sequence token, so finished sequences burn compute as padding. MATH has high output-length variance (some solutions are 50 tokens, some are 700+), so doubling the batch from 128 to 256 produces less than doubled throughput. GSM8K outputs are more uniform; batch 256 there should be closer to a real 2&times;. The real fix for eval throughput is <em>continuous batching</em>: vLLM PagedAttention swaps finished requests out and pulls new ones in mid-batch, so key/value-cache utilisation stays high regardless of length variance.</p>

<div class="callout warn">
  <div class="callout-label">Verdict on the question</div>
  <p style="margin: 0.4rem 0 0;"><strong>Memory-feasible at batch 64 training and batch 256 eval &mdash; yes.</strong> Throughput-feasible &mdash; not without the trainer changes below. The current notebook holds at 16 / 128 because pushing past those without addressing the CPU collator and the static-batching straggler problem produces marginal returns. The right path to maximum A40 utilisation is to land the trainer and eval improvements first, <em>then</em> raise the batches.</p>
</div>

<h4>Pending trainer-side improvements (research-backed, not yet applied)</h4>

<p>These came out of a focused research pass on small-model supervised fine-tuning throughput on Ampere. Sources are listed at the end of this subsection. Each row gives the change, the expected impact, and the risk of getting it wrong.</p>

<table>
  <thead>
    <tr>
      <th>Improvement</th>
      <th>Where it lives</th>
      <th>Expected impact</th>
      <th>Risk</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>FlashAttention&nbsp;2 with <code>cu_seqlens</code> variable-length attention instead of the 4D document-isolation mask</td>
      <td>Trainer: <code>dataset.py</code> collator (drop the 4D mask entirely, emit only <code>{input_ids, labels, position_ids}</code>); model load with <code>attn_implementation="flash_attention_2"</code></td>
      <td>1.4&ndash;2&times; training throughput; removes the CPU collator bottleneck; unlocks batch 32 / 64 safely</td>
      <td><strong>Medium.</strong> The correctness trap is real: if the collator emits <em>any</em> <code>attention_mask</code> key, HuggingFace falls back to the dense-mask path and silently ignores <code>position_ids</code> for cu_seqlens construction. Needs a unit test that asserts the loss on a 2-document packed row equals the mean of the two unpacked single-document losses (within rounding).</td>
    </tr>
    <tr>
      <td>Liger-Kernel via <code>apply_liger_kernel_to_qwen2()</code></td>
      <td>Trainer: one line before model load, plus <code>pip install liger-kernel</code></td>
      <td>~20% training throughput; large drop in activation memory (its <code>FusedLinearCrossEntropy</code> avoids materialising the full <em>batch &times; seq &times; vocab</em> logits tensor)</td>
      <td>Low. Drop-in monkey-patch. Compatible with FlashAttention&nbsp;2.</td>
    </tr>
    <tr>
      <td>DataLoader: <code>num_workers&ge;2</code>, <code>pin_memory=True</code>, <code>persistent_workers=True</code></td>
      <td>Trainer: <code>make_loaders</code> in <code>dataset.py</code></td>
      <td>5&ndash;20% step-time; specifically attacks the 4D-mask-on-CPU stall. Becomes less important if FlashAttention&nbsp;2 lands first (no more 4D mask), but the tokenise/pack/<code>position_ids</code> build still benefits.</td>
      <td>Low. Watch for non-fork-safe state in <code>__getitem__</code>; current collator is pure CPU/Python so it should be safe.</td>
    </tr>
    <tr>
      <td>Fused AdamW: <code>torch.optim.AdamW(..., fused=True)</code></td>
      <td>Trainer: one line in <code>build_optimizer</code></td>
      <td>3&ndash;8% step-time, no memory cost</td>
      <td>Trivial.</td>
    </tr>
    <tr>
      <td>vLLM for the eval command-line interface in place of HuggingFace <code>generate()</code></td>
      <td>Eval CLI: rewrite of <code>eval_exact.py</code> generation path; vLLM has its own model-load entrypoint</td>
      <td>Several&times; eval throughput, especially on MATH; eliminates static-batching straggler waste</td>
      <td>Medium. Tokenisation parity and end-of-sequence handling need to match the existing details-CSV semantics so re-runs stay byte-comparable.</td>
    </tr>
  </tbody>
</table>

<div class="callout">
  <div class="callout-label">Skip these &mdash; commonly recommended, does not apply here</div>
  <ul style="margin: 0.4rem 0 0; padding-left: 1.2rem;">
    <li><strong>Gradient checkpointing.</strong> Costs ~20% throughput to buy activation memory we do not need at 0.5B + batch 16 on 48&nbsp;GB.</li>
    <li><strong>8-bit / paged AdamW (bitsandbytes).</strong> Optimizer state is only ~4&nbsp;GB. Dequantisation overhead costs more than the memory it saves.</li>
    <li><strong><code>torch.compile</code> for training.</strong> There are recurring bfloat16 + HuggingFace Transformers regressions (silent <code>generate()</code> no-ops, dtype-mismatch errors, recompile storms on dynamic shapes). The expected 10&ndash;20% gain is not worth the time-to-debug for a learning project until the items above are exhausted.</li>
    <li><strong>HuggingFace TGI for eval.</strong> TGI is an HTTP serving stack. The eval pass is offline batch over 500 prompts on one A40 &mdash; vLLM <code>LLM(...)</code> offline application programming interface is the right tool.</li>
  </ul>
</div>

<h4>Sources</h4>
<ul>
  <li><a href="https://huggingface.co/blog/packing-with-FA2">HuggingFace blog &mdash; Packing with FA2</a> &mdash; canonical reference for the cu_seqlens path. Reports 1.4&ndash;2&times; gains depending on how short and variable the packed sub-documents are.</li>
  <li><a href="https://arxiv.org/html/2407.09105v4">arXiv 2407.09105 &mdash; Enhancing Training Efficiency Using Packing with Flash Attention</a></li>
  <li><a href="https://github.com/linkedin/Liger-Kernel">Liger-Kernel (LinkedIn)</a></li>
  <li><a href="https://docs.vllm.ai/en/latest/serving/offline_inference.html">vLLM offline inference docs</a> and <a href="https://arxiv.org/abs/2511.17593">arXiv 2511.17593 &mdash; vLLM vs TGI comparison</a></li>
  <li><a href="https://huggingface.co/docs/transformers/main/perf_train_gpu_one">HuggingFace Transformers single-GPU training perf guide</a> (covers fused AdamW, num_workers, pin_memory, persistent_workers, plus the &ldquo;skip these&rdquo; rationale)</li>
  <li><a href="https://github.com/huggingface/transformers/issues/30945">transformers issue #30945 &mdash; <code>torch.compile</code> + bfloat16</a></li>
</ul>
"""


def main() -> None:
    content = STUDY.read_text(encoding="utf-8")

    # Find the end of the Hardware section. The Recommended workflow <ol>
    # ends with </ol> directly before </section>. Anchor on that.
    pattern = re.compile(
        r'(<section id="hardware">.*?</ol>\s*)(</section>)',
        re.DOTALL,
    )
    m = pattern.search(content)
    if not m:
        raise AssertionError("could not locate the hardware section close")

    new_content = content[: m.end(1)] + NEW_BLOCK + m.group(2) + content[m.end(2) :]
    STUDY.write_text(new_content, encoding="utf-8")
    print(f"STUDY.html updated: +{len(NEW_BLOCK):,} chars in section 11")


if __name__ == "__main__":
    main()
