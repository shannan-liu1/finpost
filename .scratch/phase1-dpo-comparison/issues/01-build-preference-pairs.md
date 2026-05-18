# 01 - Build fixed offline DPO preference pairs

- **Status:** Not Started
- **Ready for agent:** yes
- **Depends on:** first Qwen SFT checkpoint for production data; code can start before it exists

## Parent

`.scratch/phase1-dpo-comparison/PRD.md`

## What to build

Build the offline preference-pair pipeline for Direct Preference Optimization:
sample multiple completions from the best SFT checkpoint on held-out training
prompts, grade them with the existing final-answer verifier path, and emit a
frozen pair dataset plus a manifest that proves there is no test leakage.

## Acceptance criteria

- [ ] `scripts/build_dpo_pairs.py` accepts checkpoint, sources, prompt count,
      samples per prompt, sampling parameters, seed, and output directory.
- [ ] Output includes `completions.jsonl`, `pairs.jsonl`, and `manifest.json`.
- [ ] Every pair records prompt id, source, chosen response, rejected response,
      chosen/rejected grades, source checkpoint id, sampling params, seed, and
      verifier version.
- [ ] GSM8K/MATH test prompts are never eligible for DPO pair generation.
- [ ] All-correct and all-incorrect prompt groups produce zero pairs and are
      counted explicitly in the manifest.
- [ ] Pair construction is deterministic for fixed completions and seed.

## Verification

- `.\.venv\Scripts\python.exe -m pytest tests/test_preference_data.py`
- A tiny dry run writes all three artifacts under `results/dpo_pairs/smoke/`
  using a tiny model or fixture completions.

## Blocked by

Production pair generation is blocked by a real SFT checkpoint. The builder,
schema, tests, and dry-run path can start immediately.
