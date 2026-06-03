# Configs

Config files are grouped by current FinChain workflow.

## Current FinChain Configs

| Method | Path | Purpose |
|---|---|---|
| SFT | `configs/finchain/sft/qwen25_1_5b_sft.yaml` | Full Qwen2.5-1.5B FinChain SFT run |
| SFT (local smoke test) | `configs/finchain/sft/local_tiny_gpt2_sft.yaml` | Fast local plumbing check with a tiny model |
| DPO | `configs/finchain/dpo/finchain_qwen25_1_5b.yaml` | DPO training config; sets the policy and reference checkpoint to optimize |
| OPD/GKD | `configs/finchain/gkd/finchain_qwen25_1_5b.yaml` | Full OPD/GKD student run |
| OPD/GKD (local smoke test) | `configs/finchain/gkd/finchain_qwen25_1_5b_canary.yaml` | Fast local smoke run |
| GRPO/RLVR | `configs/finchain/grpo/qwen25_1_5b_grpo.yaml` | Full GRPO/RLVR run |

The GPU notebooks keep their launch settings inline so each notebook is self-contained in a fresh accelerator environment. These YAMLs make the same run structure readable and reusable outside notebooks. If a notebook launch cell overrides a YAML value, treat the notebook cell as the authoritative record.
