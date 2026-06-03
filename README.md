# FinPost

FinPost compares four fine-tuning methods — SFT, DPO, GRPO/RLVR, and OPD/GKD — on multi-step financial reasoning with Qwen2.5-1.5B.

The main contribution is the evaluation harness: a template-disjoint FinChain split, deterministic answer grading, repeatable training workflows, and validation-based checkpoint selection. The goal was to test each method under comparable conditions and document what improved, what failed, and why.

## Results

Held-out FinChain v2 test split, n=870. Checkpoints were selected on validation accuracy; test was not touched until final reporting.

| Run | Exact answer accuracy | Interpretation |
|---|---:|---|
| Base Qwen2.5-1.5B | 16.55% | Reference model |
| SFT | 17.93% | Lift over base, with stronger reasoning scores |
| DPO | 11.49% | Regressed below base — see note |
| GRPO/RLVR | 18.51% | Best exact accuracy and step metrics among verifier-reward methods |
| OPD/GKD | 19.43% | Highest raw exact-answer accuracy; weaker step scores than SFT and GRPO |

**DPO regression:** preference pairs were sampled from the SFT policy, which was not strong enough to produce consistently informative chosen/rejected contrasts. A follow-up should generate pairs from a stronger teacher model — the same setup that made OPD/GKD the best performer here.

**OPD/GKD caveat:** the higher exact-answer accuracy came with weaker Step F1 than SFT and GRPO, meaning the reasoning traces were less well-formed. Read the 19.43% as a directional result, not a clean win.

## Where To Look

```text
notebooks/              Training and eval workflows for each method
configs/finchain/       Method-specific training configs
src/finpost/            Data loaders, answer verifiers, training utilities, and eval logic
scripts/                Training, eval, preflight, and Hugging Face helpers
results/public_evals/   Compact public result summaries
results/preflight/      Dataset provenance, checksums, and dry-run verification
docs/                   Experiment logs, checkpoint locations, metrics, and troubleshooting
```

## Run Checks

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -e ".[dev,rlvr,chaineval]"

python -m pytest tests
python scripts/smoke_finchain_notebooks.py --out results/preflight/notebook_smoke.json
```

A passing test suite means the data loaders, verifiers, and eval logic are intact. Full training requires the FinChain JSONLs under `data/finchain_v2_template_disjoint/` and GPU access — see [`data/README.md`](data/README.md) for how to get or regenerate those files.

Model weights and eval artifacts are on Hugging Face. See [`docs/hf_checkpoints.md`](docs/hf_checkpoints.md) for locations, [`docs/hf_public_metrics.md`](docs/hf_public_metrics.md) for full metrics, and [`docs/experiment_registry.md`](docs/experiment_registry.md) for the method-to-evidence map.
