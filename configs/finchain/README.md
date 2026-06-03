# FinChain Configs

Current method configs are grouped by training stage:

```text
configs/finchain/
|-- sft/
|-- dpo/
|-- gkd/
`-- grpo/
```

Run order for the public notebooks is:

1. SFT: `notebooks/01_sft_ablation.ipynb`
2. DPO: `notebooks/02_dpo.ipynb`
3. GRPO/RLVR: `notebooks/03_grpo_rlvr.ipynb`
4. OPD/GKD: `notebooks/04_opd_gkd.ipynb`

All methods expect the FinChain v2 split files to be available locally under `data/finchain_v2_template_disjoint/`. See `data/README.md` for how to get or regenerate those files.
