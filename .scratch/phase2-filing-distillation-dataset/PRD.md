# Phase 2 filing distillation dataset

- **Status:** Not Started
- **Created:** 2026-05-09
- **Owner:** Shannan
- **Estimated time:** ~1 week for survey/design, ~2 weeks for first dataset build after Phase 1 comparison
- **Depends on:** [`phase1-dpo-comparison`](../phase1-dpo-comparison/PRD.md)

## Goal

Create the data plan for the 10-K/10-Q numerical reasoning experiment: first survey open-source finance QA datasets, then decide whether to adapt an existing dataset, generate teacher-distilled filing examples, or use a hybrid.

This workstream starts after Phase 1 answers whether SFT + DPO beats pure SFT on math.

## Scope

**In scope:**
- Survey open-source financial QA and SEC filing datasets.
- Decide which datasets are suitable for SFT, DPO, evaluation, or only inspiration.
- Define the filing-excerpt example schema for numerical extraction and numerical reasoning.
- Define programmatic verification for citations, arithmetic, and answer grounding.
- Define where LLM-as-judge is allowed and where it is not.
- Define how teacher-generated examples become SFT data and DPO preference pairs.

**Out of scope:**
- Running teacher generation.
- Training on finance data.
- GRPO.
- Using an LLM judge as the only source of truth for numerical correctness.

## Candidate external datasets to evaluate

- [`Tim-Pinecone/sec-10k-qa`](https://huggingface.co/datasets/Tim-Pinecone/sec-10k-qa): SEC 10-K retrieval QA dataset, 20 companies x 5 years, 950 QA pairs.
- [FinQA](https://finqasite.github.io/): financial numerical reasoning over reports, 8K QA pairs with annotated reasoning programs.
- [TAT-QA](https://github.com/NExTplusplus/TAT-QA): hybrid table/text financial annual-report QA, 16K+ questions over real financial reports.
- [`nvidia/Nemotron-SpecializedDomains-Finance-v1`](https://huggingface.co/datasets/nvidia/Nemotron-SpecializedDomains-Finance-v1): synthetic finance QA grounded in SEC filings.
- [`thinkwee/DDRBench_10K`](https://huggingface.co/datasets/thinkwee/DDRBench_10K): structured 10-K financial database for deeper data-research tasks.

These candidates are not automatically accepted. Each must pass licensing, schema, leakage, and task-fit checks.

## Deliverables

- `.scratch/phase2-filing-distillation-dataset/issues/01-open-source-dataset-survey.md`
- `.scratch/phase2-filing-distillation-dataset/issues/02-filing-example-schema-and-verifier.md`
- `.scratch/phase2-filing-distillation-dataset/issues/03-teacher-generation-and-judge-rubric.md`
- `.scratch/phase2-filing-distillation-dataset/issues/04-assemble-sft-and-dpo-data-plan.md`

Future code/doc deliverables:

```
docs/primers/filing-distillation.md
src/finpost/finance/schema.py
src/finpost/finance/verify.py
scripts/survey_finance_datasets.py
tests/test_finance_verify.py
```

## Acceptance criteria

1. Dataset survey compares at least three external datasets on license, size, source documents, schema, numerical reasoning support, citation support, and fit for this project.
2. The chosen Phase 2 data path explicitly states: existing dataset only, teacher generation only, or hybrid.
3. Filing-excerpt schema distinguishes numerical extraction from numerical reasoning.
4. Verifier contract includes citation presence, numeric final-answer checking, arithmetic re-execution where possible, and unanswerable rejection.
5. LLM-as-judge rubric is limited to semantic checks such as answerability, citation relevance, and explanation faithfulness.
6. SFT and DPO data assembly rules are written before any teacher-generation run starts.
7. Train/test leakage policy is written before any dataset is loaded.

## Notes / open questions

- Open-source 10-K datasets exist, but many are retrieval-oriented rather than numerical-reasoning oriented.
- FinQA and TAT-QA are likely stronger for numerical reasoning; SEC 10-K QA datasets are likely stronger for filing-domain grounding and retrieval.
- The most likely final path is hybrid: use existing datasets for baseline/eval inspiration, then generate verified filing-excerpt examples for the exact target capability.
