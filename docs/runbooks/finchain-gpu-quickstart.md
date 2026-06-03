# FinChain GPU Quickstart

This is the shortest safe path for reproducing the public FinChain workflow on a
fresh GPU environment.

## 1. Prepare the Environment

```bash
cd /workspace
git clone https://github.com/shannan-liu1/finpost.git
cd finpost
python -m pip install -e ".[dev,rlvr,chaineval]"
bash scripts/repair_cuda_stack.sh
python scripts/check_cuda_stack.py
```

Use `HF_HOME=/workspace/hf-cache` and keep checkpoints/evals under the ignored
`results/` subfolders. Authenticate before any upload:

```bash
wandb login
huggingface-cli login
```

## 2. Place FinChain Data

The public repo does not commit the generated FinChain rows. Before preflight or
training, place these files under `data/finchain_v2_template_disjoint/`:

- `train.jsonl`
- `validation.jsonl`
- `test.jsonl`
- `manifest.json`

Then run:

```bash
python scripts/audit_finchain_template_disjoint_manifest.py \
  --data-dir data/finchain_v2_template_disjoint \
  --out artifacts/preflight/finchain_v2_manifest_audit.json
python scripts/gpu_preflight.py \
  --out artifacts/preflight/preflight_report.json \
  --timeout-sec 900
```

Stop if either command fails. Do not start training with mismatched checksums or
missing split files.

## 3. Notebook Order

Run notebooks in this order:

1. `notebooks/01_sft_ablation.ipynb`
2. `notebooks/02_dpo.ipynb`
3. `notebooks/03_grpo_rlvr.ipynb`
4. `notebooks/04_opd_gkd.ipynb`

SFT is the upstream policy for DPO, GRPO/RLVR, and OPD/GKD. Validation selects
checkpoints. Test is for final reporting only.

## 4. Expected Outputs

| Notebook | Primary output | Public HF target |
|---|---|---|
| SFT | selected SFT checkpoint and validation/test evals | `qwen25-1p5b-finchain-v2-sft-selected`, `qwen25-1p5b-finchain-v2-sft-candidates` |
| DPO | selected DPO checkpoint and evals | `qwen25-1p5b-finchain-v2-dpo-selected`, `qwen25-1p5b-finchain-v2-dpo-candidates` |
| GRPO/RLVR | selected GRPO checkpoint and evals | `qwen25-1p5b-finchain-v2-grpo-selected`, `qwen25-1p5b-finchain-v2-grpo-candidates` |
| OPD/GKD | selected GKD checkpoint and evals | `qwen25-1p5b-finchain-v2-gkd-selected` |

Use `docs/hf_checkpoints.md` as the reference for public repo IDs.

## 5. Stop Rules

- Do not use test results to choose a checkpoint.
- Do not push a method that fails its validation gate.
- Do not push candidate repos unless the notebook explicitly says the repo is
  verified in `docs/hf_checkpoints.md`.
- Do not commit raw data, checkpoints, full eval dumps, logs, caches, or tokens.

Compact public result summaries belong under `results/public_evals/`. Full run
artifacts stay ignored or live on Hugging Face.
