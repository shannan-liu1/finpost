# 0002. Phase 1 data loading (GSM8K, MATH)

- **Status:** Done (verified 2026-05-05; see Amendments 1 and 2)
- **Created:** 2026-05-05
- **Owner:** Shannan
- **Estimated time:** ~1.5 hours
- **Depends on:** [`repo-skeleton`](../repo-skeleton/PRD.md)

## Goal

Load the GSM8K and MATH benchmarks from Hugging Face, normalize both into a single common record schema, and expose summary statistics. After this PRD, downstream training code can consume both datasets through a uniform interface without needing to know either dataset's idiosyncrasies.

## Scope

**In scope:**
- A common `Example` schema (Python `dataclass` or `TypedDict`) with the fields needed for Supervised Fine-Tuning: `id`, `source`, `difficulty`, `prompt`, `response`, `answer`, plus a parsed `final_answer` token.
- A loader for GSM8K that downloads via `datasets.load_dataset("gsm8k", "main")` and returns `Example` records with `####`-sentinel parsing.
- A loader for MATH that downloads the equivalent dataset and returns `Example` records with `\boxed{...}` parsing. Difficulty levels (1–5) are preserved.
- A small command-line interface (`python -m finpost.data.cli`) that prints, per dataset:
  - Train and test counts.
  - Token-length distribution of the full `prompt + response` text (mean, median, 95th percentile, max), tokenized with the Gemma 3 1B tokenizer.
  - One full example printed to console for visual inspection.
- A pytest test file verifying the schema is satisfied for a sample of records from each dataset.

**Out of scope:**
- Tokenization for training (we only tokenize for length stats here; the trainer handles training-time tokenization).
- Caching to disk in custom formats (Hugging Face `datasets` already caches).
- Any LaTeX equivalence checking (that's the grader's job, in a later PRD).
- Deduplication across train/test (we trust the official splits in this PRD; cross-check is a separate workstream).

## Deliverables

```
src/finpost/data/
├── __init__.py
├── schema.py                  # Example dataclass and validation
├── gsm8k.py                   # load_gsm8k() -> list[Example]
├── math_dataset.py            # load_math() -> list[Example]
└── cli.py                     # `python -m finpost.data.cli --dataset gsm8k|math`

tests/
└── test_data_schema.py        # pytest tests covering schema validation for both loaders
```

## Acceptance criteria

1. `python -m finpost.data.cli --dataset gsm8k --split train` prints train count (~7,473), length stats with units, and one example.
2. `python -m finpost.data.cli --dataset gsm8k --split test` prints test count (~1,319) and stats.
3. `python -m finpost.data.cli --dataset math --split train` prints train count (~7,500), length stats, difficulty distribution, and one example.
4. `python -m finpost.data.cli --dataset math --split test` prints test count (~5,000) and stats.
5. `pytest tests/test_data_schema.py -v` passes.
6. Every record returned by both loaders has a non-empty `prompt`, non-empty `response`, and a `final_answer` that was successfully parsed (not `None`).

## Notes / open questions

- The MATH dataset has been redistributed under multiple Hugging Face dataset IDs over the years (`hendrycks/math`, `lighteval/MATH`, etc.). Use the most current canonical one as of execution; document the exact ID in the loader.
- The GSM8K canonical answer parser strips commas from numbers (e.g., `1,200` → `1200`). Match that convention.
- For MATH, use the published normalizer from `lm-evaluation-harness` rather than writing our own (this PRD does not yet integrate the normalizer; it just preserves the raw `\boxed{...}` content as `final_answer`).
- Tokenizer download requires Hugging Face authentication for gated Gemma weights. Document this in the PRD's exit summary.

## Amendment 1 — 2026-05-05

Implementation deviated from the original scope in three places, all noted here for the record:

1. **Schema implementation:** `Pydantic` (`BaseModel`) rather than the `dataclass` / `TypedDict` mentioned in scope. Decision recorded in conversation; chosen for runtime validation guarantees.
2. **MATH dataset source:** `DigitalLearningGmbH/MATH-lighteval` (parquet-only mirror) rather than `lighteval/MATH` (which ships a Python loader script). Reason: supply-chain hardening — see `SECURITY.md`.
3. **GSM8K loading approach:** dropped the planned `revision="refs/convert/parquet"` pin after verifying it flattens multi-config datasets to a single `default` config (losing the `main` vs `socratic` distinction). Security guarantee comes from `trust_remote_code=False` instead, which already disables script execution. See `SECURITY.md`.

**Open issue at amendment time:** the MATH `\boxed{...}` parser does not handle the LaTeX-valid `\boxed N` form (no braces, single token argument). At least one MATH train example uses this form: `... our answer is $\boxed 2$`. Acceptance criteria 3 cannot be marked passed until this is fixed.

## Amendment 2 — 2026-05-05

Resolved the open issue from Amendment 1 plus discovered and addressed two additional MATH per-record data-quality issues during the full-load verification.

**Parser changes (`src/finpost/data/math_dataset.py`):**

1. Vendored `last_boxed_only_string` and `remove_boxed` verbatim from [hendrycks/math](https://github.com/hendrycks/math) (MIT licensed; attributed in source). The vendored functions handle the canonical `\boxed{...}` and `\fbox{...}` brace forms with proper nesting.
2. Extended `remove_boxed` to also strip `\fbox{` (the upstream library has a known gap here — see `EleutherAI/lm-evaluation-harness#3116`).
3. Added a small `_parse_no_brace_boxed` extension to handle `\boxed N` (no-brace LaTeX form). This is project code, not vendored; documented as a divergence from the canonical parser.
4. `parse_math_difficulty` now returns `int | None`, with `None` for the dataset's known `Level ?` unknown-marker. Other non-integer values still raise.

**Loader change (`load_math`):** switched from strict (raise on first failure) to skip-and-report. Failed records are counted by category and the totals are printed when any are skipped.

**Discovered failure rates (full scan, 2026-05-05):**

- MATH train (7500 records): 2 skipped (0.03%), both `\boxed{}` empty-braces in count problems. Author typo — intended answer was `0`. Skipping is conservative until we have an answer-equivalence normalizer (planned for the grading PRD).
- MATH train: 2 records with `Level ?` marker — loaded successfully with `difficulty=None`.
- MATH test (5000 records): 0 failures.
- GSM8K train (7473) and test (1319): 0 failures.

**Acceptance criteria status:**

1. ✓ GSM8K train CLI run prints count, length stats, sample example.
2. ✓ GSM8K test CLI run prints count, length stats.
3. ✓ MATH train CLI run prints count, length stats, difficulty + category distributions, sample example.
4. ✓ MATH test CLI run prints count, length stats.
5. ✓ `pytest tests/test_data_schema.py -v` passes (along with all other test files).
6. ✓ Every returned record has non-empty fields (Pydantic enforces this; loader skips records that would violate it, with logged counts).

## Amendment 3 - 2026-05-09

The original scope mentioned Gemma token-length statistics because Gemma was the initial Phase 1 base model. ADR-0001 supersedes that model choice. Current and future length stats for Phase 1 should use `Qwen/Qwen2.5-0.5B` unless an issue explicitly requests a different tokenizer comparison.

This does not change the loader contract: loaders still return normalized text examples, and training-time tokenization remains owned by the SFT trainer workstream.
