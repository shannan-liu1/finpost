# 01 - Source registry and answer extractors

- **Status:** Not Started
- **Ready for agent:** yes
- **Depends on:** none

## Goal

Build the domain-agnostic source registry that the eval CLI will consume. Define the `EvalSource` dataclass and register the two Phase 1 entries: `gsm8k` and `math`. Each entry knows how to load eval examples, extract a final answer from a model generation, and score that answer against the gold.

## Scope

**In scope:**
- `src/finpost/evals/__init__.py` — empty marker file (the package will be populated by issues 01 and 02).
- `src/finpost/evals/sources.py` — defines `EvalSource` dataclass and a `REGISTRY` dict keyed by source name. Initial entries: `gsm8k`, `math`.
- `tests/test_eval_sources.py` — answer extractor tests (positive + negative cases) and source-registry lookup tests.
- Reuse the existing `load_gsm8k` and `load_math` loaders from `src/finpost/data/`. Do not duplicate dataset loading.

**Out of scope:**
- The CLI itself (issue 02).
- Generation or model code (issue 02).
- Notebooks (issue 03).
- Anything related to cost tracking or output writing.

## EvalSource contract

```python
@dataclass(frozen=True)
class EvalSource:
    name: str
    load_examples: Callable[[], list[Example]]   # returns the test split
    extract_answer: Callable[[str], str | None]  # parses model generation; returns None on parse failure
    score: Callable[[str | None, str], bool]     # (predicted, gold) -> is_correct
    default_max_new_tokens: int                  # generation budget hint
```

The dataclass is frozen so registry entries cannot be mutated at runtime. `load_examples` is wrapped in a thunk so the registry import does not trigger dataset downloads.

## Answer extractor rules

- **GSM8K** answer pattern: the gold convention is the final line `#### <number>`. The extractor should:
  - Look for the *last* `#### <number>` in the generation.
  - Return the number as a normalized string (strip whitespace, strip leading `$`, strip trailing periods, strip commas — but do not convert to float; the score comparison is string-based after normalization).
  - Return `None` if no `####` marker is found.

- **MATH** answer pattern: the gold convention is `\boxed{<answer>}`. The extractor should:
  - Find the last `\boxed{...}` in the generation, respecting balanced braces (the contents may contain nested braces, e.g., `\frac{1}{2}`).
  - Return the inner contents as a normalized string (strip whitespace, strip wrapping `$` if present).
  - Return `None` if no `\boxed{...}` is found.

- **Score function** for both sources: simple string equality after normalization. Predicted `None` is always incorrect.

## Acceptance criteria

1. `python -c "from finpost.evals.sources import REGISTRY; print(sorted(REGISTRY))"` prints `['gsm8k', 'math']`.
2. `REGISTRY['gsm8k'].extract_answer("blah blah\n#### 42")` returns `"42"`.
3. `REGISTRY['gsm8k'].extract_answer("no marker here")` returns `None`.
4. `REGISTRY['math'].extract_answer("answer is \\boxed{\\frac{1}{2}}")` returns `"\\frac{1}{2}"`.
5. `REGISTRY['math'].extract_answer("no boxed here")` returns `None`.
6. `REGISTRY['gsm8k'].score("42", "42")` returns `True`; `REGISTRY['gsm8k'].score("42", "43")` returns `False`; `REGISTRY['gsm8k'].score(None, "42")` returns `False`.
7. `REGISTRY['gsm8k'].default_max_new_tokens == 256`; `REGISTRY['math'].default_max_new_tokens == 768`.
8. `pytest tests/test_eval_sources.py -v` passes.
9. `pytest` (whole suite) still passes — no regressions to existing 102 tests.

## Notes / open questions

- The codebase already has `from finpost.data.gsm8k import load_gsm8k` and `from finpost.data.math_dataset import load_math`. Use those — do not re-implement. Inspect their signatures first.
- Keep all functions small and pedagogically clear. The user's standing preference is that every line of code be intelligible without library magic. Comments are welcome where they explain *why*, not what.
- The answer extractors are pure functions (no I/O, no state). They should be straightforward to test exhaustively.
- The `MATH` brace-matching is the only tricky bit — write the brace counter explicitly, do not use a regex.
