# finpost — execution plan (draft)

This is the working plan. Decisions still open are tagged `[OPEN]`. Resolved decisions are tagged `[DECIDED]` with a one-line justification. Replace `[OPEN]` tags inline as decisions are made.

For terminology, see [`CONTEXT.md`](./CONTEXT.md). Abbreviations are spelled out on first use.

---

## Phase 0 — Setup (≈1–2 days)

- [ ] Repository skeleton
  - `data/` — raw filings, processed examples (gitignored)
  - `src/finpost/` — Python package: data, training, evaluation
  - `experiments/` — configuration files for each run
  - `notebooks/` — exploratory work only, not load-bearing
  - `results/` — checkpoints (gitignored), evaluation outputs
- [ ] Environment
  - Python 3.11+, CUDA 12.x
  - Core libraries: `torch`, `transformers`, `peft`, `bitsandbytes`, `accelerate`, `datasets`, `trl` (only as a reference for our from-scratch loss; not used as the trainer)
  - Eval and analysis: `numpy`, `scipy` (for bootstrapping)
  - Tracking: Weights & Biases (`wandb`)
- [ ] Hardware verification
  - Confirm A100 access (40 GB or 80 GB? `[OPEN]` — affects context budget)
  - Smoke test: load Gemma 3 1B, generate 100 tokens, measure tokens/sec
- [ ] API keys and budgets
  - Hugging Face token (model download)
  - Anthropic and/or OpenAI key (for teacher distillation in Phase 2)
  - Set a **hard** budget cap on the teacher API key (e.g. $300) to avoid runaway costs

---

## Phase 1 — Math post-training stack (≈2–3 weeks)

**Goal:** end-to-end Supervised Fine-Tuning (SFT) and Direct Preference Optimization (DPO) stack with full ablations on free-grader benchmarks. Base model is Gemma 3 1B with full fine-tuning (no LoRA in this phase).

### 1.1 Data preparation

- [ ] Download GSM8K (Cobbe et al., 2021) train and test splits via `datasets`.
- [ ] Download MATH (Hendrycks et al., 2021) train and test splits.
- [ ] Standardize to a single shape:
  ```
  { "id": str, "source": "gsm8k" | "math",
    "difficulty": int | None,
    "prompt": str,
    "response": str,           # gold chain-of-thought + final answer
    "answer": str }            # parsed final answer for grading
  ```
- [ ] Length statistics: token-length histogram; cap or filter extreme outliers.
- [ ] Deduplicate against the test sets (paranoia check; HuggingFace versions are clean but verify).

### 1.2 Supervised Fine-Tuning trainer

- [ ] Implement masked cross-entropy loss in a thin training loop:
  - Apply Gemma's chat template
  - Mask the prompt tokens (loss only on the response)
  - Mixed-precision (bf16), gradient accumulation, cosine learning-rate schedule
- [ ] First sanity run: 100 steps on a small subset; confirm loss decreases monotonically.
- [ ] Full first SFT run on combined GSM8K + MATH at one set of hyperparameters.
- [ ] Evaluate Base vs. SFT on test sets (see 1.5).

`[DECIDED 2026-05-05]` Decision Q-A: **write our own minimal trainers from scratch** for both Supervised Fine-Tuning and Direct Preference Optimization (~300–500 lines each). Justification: the project goal is depth on the mechanics, and the trainer is where the substrate of post-training (masking, gradient accumulation, optimizer dynamics, mixed-precision casts) lives. Use TRL only as a reference for numerical sanity-checking the Direct Preference Optimization loss.

### 1.3 SFT ablation matrix

- [ ] Vary one axis at a time:
  - Data scale: 10%, 50%, 100% of training data
  - Learning rate: 1e-5, 5e-5, 1e-4
  - Epochs: 1, 3
- [ ] Each cell: train, evaluate, log to Weights & Biases.
- [ ] Realistic budget: ~9–12 cells on a single A100 over a week.

### 1.4 Direct Preference Optimization preparation

- [ ] From the best SFT checkpoint, sample N=8 completions per prompt at temperature 0.8 over a held-out set of training prompts (not test).
- [ ] Programmatically grade each completion (final-answer match).
- [ ] Form preference pairs:
  - `[OPEN]` Decision Q-B: how to handle prompts with all-correct or all-incorrect samples? See decisions section.
- [ ] Target: ~5,000 preference pairs.

### 1.5 Direct Preference Optimization implementation (from scratch, no TRL trainer)

- [ ] Implement the DPO loss directly:
  ```
  loss = -log_sigmoid(beta * ((logp_pi(chosen|x) - logp_ref(chosen|x))
                              - (logp_pi(rejected|x) - logp_ref(rejected|x))))
  ```
- [ ] Reference model = SFT checkpoint, frozen, with `requires_grad=False`.
- [ ] Policy model = SFT checkpoint, training.
- [ ] Numerical sanity check: compute the loss on one batch with our implementation and with TRL's `DPOTrainer` loss; values must match within ~1e-5.
- [ ] First DPO run, then ablation matrix:
  - beta: 0.01, 0.1, 0.5
  - learning rate: 1e-6, 5e-6, 1e-5
  - data scale: 25%, 100% of preference pairs

### 1.6 Evaluation harness

- [ ] Per-checkpoint metrics on test sets:
  - GSM8K: final-answer exact-match accuracy
  - MATH: final-answer LaTeX equivalence (use a known normalizer; do not write our own)
  - Per-difficulty breakdown for MATH (1–5)
- [ ] Bootstrapped 95% confidence intervals (10K resamples, paired where comparing two checkpoints on the same prompts)
- [ ] `[OPEN]` Decision Q-C: minimum test-set sample size to discriminate ~5-percentage-point improvements? Power analysis required.
- [ ] Track also: mean response length, mean number of "let me think" patterns (degenerate-CoT detection)

### 1.7 Phase 1 deliverable

- [ ] Three checkpoints saved and versioned: Base, SFT-best, SFT+DPO-best.
- [ ] Ablation tables with confidence intervals.
- [ ] A short writeup (one page) documenting what worked, what didn't.

---

## Phase 2 — Financial domain transfer (≈3–4 weeks)

**Goal:** port the stack to numerical reasoning over real EDGAR filing sections. Switch from full fine-tuning to QLoRA at long context. Use teacher-model distillation with programmatic verification for data.

### 2.1 Corpus acquisition

- [ ] Pick companies. Recommendation: 30–50 large-cap United States companies across sectors (technology, financials, retail, energy, healthcare, industrials). Examples: Apple, Microsoft, Alphabet, Amazon, Meta, JPMorgan, Bank of America, Berkshire Hathaway, Walmart, Costco, Exxon, Chevron, UnitedHealth, Pfizer, Caterpillar, Boeing.
  - `[OPEN]` Decision Q-D: company list — exact set. See decisions section.
- [ ] Time range: most recent 4 fiscal years.
- [ ] Forms: 10-K (annual) and 10-Q (quarterly).
- [ ] Tooling: `edgartools` Python package (handles SEC throttling, parses HTML and XBRL).
- [ ] Estimated count: ~30 companies × 4 years × (1 10-K + 3 10-Q) ≈ 480 filings.
- [ ] Storage: raw HTML and parsed text on disk under `data/raw/<ticker>/<period>/`. Gitignored.

### 2.2 Section extraction

- [ ] Parse each filing into sections by Item heading.
- [ ] Keep high-value sections only:
  - 10-K Item 7 (Management's Discussion and Analysis)
  - 10-K Item 7A (Quantitative and Qualitative Disclosures about Market Risk)
  - 10-K Item 8 (Financial Statements and Supplementary Data — contains the income statement, balance sheet, cash flow statement, and footnotes)
  - 10-Q Item 1 (Financial Statements) and Item 2 (Management's Discussion and Analysis)
- [ ] For each section, store both:
  - Plain text (with table content rendered as readable text using `tabulate` or similar)
  - Structured table extractions (when XBRL is available, use it as the canonical source)
- [ ] Standardized record:
  ```
  { "filing_id": str, "ticker": str, "form": "10-K" | "10-Q",
    "period_end": "YYYY-MM-DD",
    "section_id": str,         # e.g. "10-K-Item-8"
    "section_title": str,
    "section_text": str,       # plain text, with table content
    "tables": [ { "title": str, "rows": [...], "columns": [...] } ],
    "token_count": int }
  ```
- [ ] Filter sections to the training context budget (target ≤ 8K tokens; chunk longer sections at natural boundaries).

### 2.3 Question generation via teacher distillation

- [ ] Pick teacher: `[OPEN]` Decision Q-E (Claude vs. GPT-5; see decisions section).
- [ ] Prompt template (sketch):
  ```
  You are generating training data for a small language model that learns to do
  numerical reasoning over financial filings.

  Below is a section of {ticker}'s {form} for the period ending {period_end}.

  Generate exactly 5 numerical questions answerable using only this section.

  Each question should require either:
  (a) extracting one specific number (label this "extraction"), or
  (b) computing a derived quantity from 2-4 numbers in the section
      (label this "reasoning").

  For each question, return a JSON object with:
    - question: the question text
    - type: "extraction" | "reasoning"
    - cited_line_items: array of verbatim quotes from the section (each is a
      single line of text containing a key number)
    - computation: step-by-step computation (one line per step;
      empty array for extraction questions)
    - final_answer: the answer as a string

  Output a JSON array of 5 such objects. No prose outside the JSON.

  Section:
  ---
  {section_text}
  ---
  ```
- [ ] Run teacher over the corpus. Cost estimate: ~$0.05–0.20 per section × ~500 sections × 5 questions per section ≈ $25–100 for the base run; budget $300 with iteration.

### 2.4 Programmatic verification (the key step that turns an unreliable teacher into a clean dataset)

- [ ] For each generated example:
  - Verify each `cited_line_item` appears in `section_text` (allow normalized whitespace; reject if not found).
  - Re-execute the `computation` programmatically (using `sympy` or a careful eval) and verify it produces `final_answer`.
  - For extraction-type questions: verify `final_answer` appears verbatim in the cited line items.
- [ ] Reject anything that fails verification.
- [ ] Expected pass rate: 60–80%. Track this number — it's a quality signal on the teacher.

### 2.5 Quality filtering and deduplication

- [ ] Trivial-question filter: reject if the answer appears verbatim in the question text.
- [ ] Embedding-based deduplication: embed all questions with a small sentence model (e.g. `bge-small`), remove pairs with cosine similarity ≥ 0.92.
- [ ] Difficulty cap: enforce a target distribution (e.g. 30% extraction, 70% reasoning) by downsampling overrepresented categories.
- [ ] Final target: ~5,000 SFT examples and ~2,000 DPO preference pairs.
  - `[OPEN]` Decision Q-F: are these the right targets? See decisions section.

### 2.6 Preference-pair construction

- [ ] For each verified example, the verified completion is the "chosen" response.
- [ ] Generate the "rejected" response by deliberate corruption. Vary the corruption type uniformly:
  - 40%: corrupt the final answer (off by an order of magnitude, or arithmetic error)
  - 30%: corrupt one cited line item (swap for a similarly-named wrong line)
  - 20%: remove the citation entirely
  - 10%: keep correct content but use generic non-financial language ("the number is...")
- [ ] Rationale: this teaches the model to disprefer multiple distinct failure modes, not just one.

### 2.7 Final example shape (the format the model is trained on)

```
<system>
You answer numerical questions about excerpts from SEC filings.
Cite the source line item, show your computation, then give the final answer.
</system>

<user>
{question}

Source ({ticker} {form} period ending {period_end}, section {section_id}):
{section_text}
</user>

<assistant>
<cited_line_item>{verbatim_quote_1}</cited_line_item>
<cited_line_item>{verbatim_quote_2}</cited_line_item>
<computation>
{step_1}
{step_2}
</computation>
<answer>{final_answer}</answer>
</assistant>
```

### 2.8 Training setup

- [ ] QLoRA configuration:
  - Base: Gemma 3 1B in 4-bit NF4 via `bitsandbytes`
  - Adapters: LoRA rank 16, alpha 32, dropout 0.05
  - Target modules: attention projections (q, k, v, o) and MLP gates (gate, up, down)
  - Optimizer: paged AdamW 8-bit
- [ ] Context length: start at 8K. If loss is stable and memory has headroom, increase to 16K.
- [ ] Use the same SFT and from-scratch DPO implementations from Phase 1.

### 2.9 Three-arm ablation (the headline experiment)

Identical Phase 2 finance training data and hyperparameters across arms; the only thing that varies is the **starting checkpoint**.

- [ ] **Arm A — Base + few-shot.** No training. Evaluate Gemma 3 1B (instruction-tuned) on the finance test set with 3 few-shot examples in the prompt.
- [ ] **Arm B — Base → finance.** Apply finance SFT + DPO starting from the base instruction-tuned model.
- [ ] **Arm C — Base → math → finance.** Apply finance SFT + DPO starting from the Phase 1 SFT+DPO-best checkpoint.

### 2.10 Evaluation harness (extends Phase 1 harness)

- [ ] Test set: held-out filing sections (held out at the **company level**, not the example level — to ensure the model isn't memorizing line-item phrasings). Recommendation: hold out 5 of the 30 companies entirely.
- [ ] Per-question metrics:
  - Final-answer correctness (numeric match within tolerance)
  - Citation correctness (cited line item appears verbatim in source)
  - Computation correctness (steps re-execute to the claimed answer)
  - Hallucination indicator: a number is in the answer that does not appear in the source
- [ ] Per-arm metrics with bootstrapped 95% confidence intervals.
- [ ] Paired bootstrap for arm-vs-arm comparisons.

---

## Phase 3 — Writeup and consolidation (≈1 week)

- [ ] Repository README with reproduction instructions.
- [ ] Loss-curve plots, ablation tables, three-arm comparison plot.
- [ ] One-page technical report.
- [ ] Optional: blog-post-style narrative.

---

## Open decision points (to resolve via grilling)

| ID | Decision | Where it lives in the plan |
|----|----------|---------------------------|
| ~~Q-A~~ | ~~Write our own SFT trainer or use Hugging Face `Trainer`~~ → **DECIDED**: write our own from scratch (2026-05-05) | 1.2 |
| Q-B | How to handle DPO prompts with all-correct or all-incorrect SFT samples | 1.4 |
| Q-C | Test-set sample size for statistical power on ~5-percentage-point gains | 1.6 |
| Q-D | Exact company list for the Phase 2 corpus | 2.1 |
| Q-E | Teacher model for distillation (Claude vs. GPT-5 vs. both) | 2.3 |
| Q-F | Target counts for Phase 2 SFT and DPO datasets (5K / 2K?) | 2.5 |

These are resolved one at a time below as the grilling continues. Each resolution gets dated.
