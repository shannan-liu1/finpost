# OPD Study Guide

Status: pair-construction backend implemented; RunPod bridge notebook and TRL
OnlineDPO launch script implemented.

## 1. Plain-English Idea

On-Policy Distillation is teacher/student learning where the teacher signal is
created from the current policy's own rollouts.

The loop is:

1. The student/current policy generates answers.
2. A verifier grades those answers.
3. The verifier signal acts like a teacher.
4. The student updates.
5. The next round samples from the updated student.

This is why OPD is not just DPO. DPO can use a fixed offline preference file.
OPD is about where the training signal comes from and when it is refreshed.

## 2. Relation To DPO

There are two valid OPD implementations:

```text
Bridge OPD:
  current policy -> rollouts -> verifier -> chosen/rejected pairs -> DPO loss

Online OPD:
  current policy -> live rollouts -> verifier/reward -> online trainer update
```

The bridge path reuses the DPO data contract because it is cheap and legible.
That is an implementation bridge, not a conceptual identity.

## 3. Backend Contract

The transparent backend lives in `src/finpost/posttraining/opd.py`:

- `OPDRollout`: one current-policy completion plus verifier reward.
- `build_opd_pairs`: groups rollouts by prompt and emits best-vs-worst pairs.
- `OPDPair.to_dpo_preference_example`: converts bridge pairs into the tested
  DPO data contract.

The industry adapter lives in `src/finpost/posttraining/finchain_rlvr.py`:

- `build_finchain_prompt_rows`: prompt rows for TRL-style trainers.
- `finchain_binary_rewards`: verifier reward function for online trainers.

## 4. Adaptive Sampling And Weighting

The first adaptive rule:

```text
success_rate >= 0.8 -> easy       -> sample less / weight 0.25
success_rate <= 0.2 -> hard       -> sample cautiously / weight 0.50
otherwise           -> ambiguous  -> sample more / weight 1.00
```

Interpretation:

- Easy prompts already work, so they should not dominate compute.
- Hard prompts may be impossible at the current model size or budget.
- Ambiguous prompts are the high-value zone: the policy sometimes solves them,
  so extra rollouts can expose learnable differences.

If using TRL OnlineDPO, treat this primarily as an adaptive sampling rule. Only
add per-example loss weights when the trainer supports them cleanly.

## 5. RunPod Notebook And Scripts

- Notebook: `notebooks/finchain_11_opd_runpod.ipynb`
- Bridge scripts: `scripts/build_dpo_pairs.py`, `scripts/merge_dpo_pair_shards.py`
- Industry online script: `scripts/train_finchain_trl_online_dpo.py`

Bridge canary:

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/build_dpo_pairs.py \
  --model-checkpoint results/checkpoints/qwen25-1p5b-finchain-sft-hf \
  --sources finchain \
  --out-dir results/finchain_opd/round_001/pairs \
  --heldout-train-n 512 \
  --samples-per-prompt 6 \
  --generation-batch-size 32 \
  --max-new-tokens-finchain 512 \
  --max-pairs-per-prompt 4
```

TRL OnlineDPO canary:

```bash
python scripts/train_finchain_trl_online_dpo.py \
  --model Qwen/Qwen2.5-1.5B \
  --train-n 256 \
  --max-steps 20 \
  --output-dir results/checkpoints/qwen25-1p5b-finchain-online-dpo-canary
```

Two-GPU OnlineDPO:

```bash
accelerate launch --num_processes 2 scripts/train_finchain_trl_online_dpo.py \
  --model Qwen/Qwen2.5-1.5B \
  --train-n 2000 \
  --max-steps 300 \
  --output-dir results/checkpoints/qwen25-1p5b-finchain-online-dpo-2gpu
```

## 6. Efficiency Reasoning

OPD's cost center is rollout generation. Use rollout parallelism for the bridge
path before trying distributed training. For the online path, use Accelerate
only after a one-GPU canary proves the reward function, data path, and logging
work.

Start with binary final-answer reward:

```text
reward = 1.0 if parsed final answer is correct else 0.0
```

Do not add shaped rewards until the binary reward's failure modes are visible.

## 7. Failure Modes

- OPD degenerates into DPO if you never refresh rollouts.
- Ambiguous prompts vanish if sampling temperature is too low.
- Pair counts explode if every positive is paired with every negative.
- Reward rises but held-out accuracy does not, indicating parser shortcuts.

## 8. Industry Package Mapping

- TRL `OnlineDPOTrainer`: closest fit to the teacher/student online loop.
- Axolotl: useful if you want config-driven DPO/ORPO-style experiments more
  than a custom online loop.
- OpenRLHF / verl: useful when OPD becomes a distributed actor/reference/reward
  system rather than a notebook experiment.
