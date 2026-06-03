# Hugging Face Checkpoints

Checked on 2026-06-02 against the public Hugging Face API for `shannan-liu1`. `verified_public` means the repo appeared in the public author listing or returned HTTP 200 from the HF API.

## FinChain v2 Repos

| Experiment | Method | Public HF Repo | Notes |
|---|---|---|---|
| FinChain v2 SFT selected | SFT | `shannan-liu1/qwen25-1p5b-finchain-v2-sft-selected` | Selected model weights |
| FinChain v2 SFT candidates/evals | SFT | `shannan-liu1/qwen25-1p5b-finchain-v2-sft-candidates` | Candidate checkpoints and validation/test eval artifacts |
| FinChain v2 DPO selected | DPO | `shannan-liu1/qwen25-1p5b-finchain-v2-dpo-selected` | Selected model weights |
| FinChain v2 DPO candidates/evals | DPO | `shannan-liu1/qwen25-1p5b-finchain-v2-dpo-candidates` | Candidate checkpoints and validation/test eval artifacts |
| FinChain v2 GRPO/RLVR selected | GRPO/RLVR | `shannan-liu1/qwen25-1p5b-finchain-v2-grpo-selected` | Selected weights |
| FinChain v2 GRPO/RLVR candidates/evals | GRPO/RLVR | `shannan-liu1/qwen25-1p5b-finchain-v2-grpo-candidates` | Candidate checkpoints and validation/test eval artifacts |
| FinChain v2 OPD/GKD selected/evals | OPD/GKD | `shannan-liu1/qwen25-1p5b-finchain-v2-gkd-selected` | Selected weights and validation/test eval artifacts |

## Earlier FinChain Experiments

These repos are from earlier FinChain exploratory runs and are not part of the reported FinChain v2 results.

| Experiment | Public HF Repo | Notes |
|---|---|---|
| Pilot FinChain SFT | `shannan-liu1/qwen25-1p5b-finchain-sft` | Pilot run before v2 split |

See `docs/hf_public_metrics.md` for metric and cost-artifact detail.
