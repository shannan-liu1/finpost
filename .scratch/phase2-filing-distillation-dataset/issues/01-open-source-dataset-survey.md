# 01 - Open-source finance dataset survey

- **Status:** Not Started
- **Ready for agent:** no
- **Depends on:** none

## Goal

Find and evaluate existing open-source financial QA / 10-K datasets before generating new teacher-distilled data.

## Scope

**In scope:** dataset search, license check, schema check, task-fit check, leakage risk, recommendation.

**Out of scope:** loading data into training code.

## Acceptance criteria

- Survey includes at least: `Tim-Pinecone/sec-10k-qa`, FinQA, TAT-QA, `nvidia/Nemotron-SpecializedDomains-Finance-v1`, and `thinkwee/DDRBench_10K`.
- Each candidate has license, size, source-document type, available fields, and task fit recorded.
- Final recommendation says whether to use, adapt, evaluate against, or reject each dataset.
- No dataset is selected until licensing and train/test leakage risks are documented.

## Notes / open questions

- Seed links for the survey:
  - `https://huggingface.co/datasets/Tim-Pinecone/sec-10k-qa`
  - `https://finqasite.github.io/`
  - `https://github.com/NExTplusplus/TAT-QA`
  - `https://huggingface.co/datasets/nvidia/Nemotron-SpecializedDomains-Finance-v1`
  - `https://huggingface.co/datasets/thinkwee/DDRBench_10K`
- Prior quick search found promising candidates, but this issue owns the canonical repo decision.
