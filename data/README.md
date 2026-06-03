# Data

## What is this data?

This repo uses **FinChain v2**, a generated financial reasoning dataset for
post-training small language models.

The generated dataset has three splits:

| Split | Rows | Purpose |
|---|---:|---|
| train | 2320 | Train the model |
| validation | 870 | Select checkpoints and monitor training |
| test | 870 | Report final held-out results |

Each row is a JSON object with a financial word problem, worked solution, final
answer, template metadata, generation seed, and split label. Common fields
include `prompt`, `problem`, `solution`, `reasoning`, `answer`,
`final_answer`, `domain`, `template_file`, `template_name`, `seed`, and `split`.

The split is **template-disjoint**: each upstream problem template appears in
exactly one split. The model can see the same financial topics across splits,
but not the same problem template structure.

## Where are the files?

The generated split files are included here:

```text
data/finchain_v2_template_disjoint/
  train.jsonl
  validation.jsonl
  test.jsonl
  manifest.json
```

Run the audit command below before training if you modify or replace them.

## How was it generated?

The data is generated from the upstream
[mbzuai-nlp/finchain](https://github.com/mbzuai-nlp/finchain) template
repository.

The builder does three things:

1. Reads an audited template catalog from `data/archive/finchain_pilot/`.
2. Loads the pinned upstream FinChain template functions.
3. Renders fresh examples under fixed seeds, then writes template-disjoint
   train/validation/test JSONLs.

The exact seeds, upstream revision, row counts, split counts, and checksums are
recorded in
`results/preflight/finchain_v2_generation_provenance.json`.

## Can a fresh clone use it?

Yes. A fresh clone has the generated train/validation/test JSONLs needed for
training and evaluation.

## Can a fresh clone regenerate it from scratch?

Not from tracked files alone.

You need two external inputs:

1. `data/archive/finchain_pilot/{train,validation,test}.jsonl`
   - Audited catalog of the upstream templates and metadata.
   - This is an intermediate source catalog, not the final training split.
   - It is ignored and not shipped in this repo.
2. `/workspace/finchain`
   - A local checkout of `mbzuai-nlp/finchain`.
   - Use revision `146eaa8225bf867fe7386c2d4727b59e03170235`.

Without both inputs, you can use and audit the committed JSONLs, but you cannot
regenerate the split from scratch.

## Recreate the data

Build:

```bash
python scripts/build_finchain_template_disjoint_splits.py \
  --source-dir data/archive/finchain_pilot \
  --upstream-root /workspace/finchain \
  --output-dir data/finchain_v2_template_disjoint
```

Verify:

```bash
python scripts/audit_finchain_template_disjoint_manifest.py \
  --data-dir data/finchain_v2_template_disjoint \
  --out results/preflight/finchain_v2_manifest_audit.json
```

The audit checks row counts, split integrity, template overlap, seed overlap,
and SHA-256 checksums. If it fails, do not train on the data.

## License

Code and documentation in this repo are MIT licensed. The generated FinChain v2
rows are derived from upstream FinChain template assets, so treat them as
subject to upstream FinChain terms. Qwen model weights and Hugging Face-hosted
artifacts remain under their own licenses and terms.
