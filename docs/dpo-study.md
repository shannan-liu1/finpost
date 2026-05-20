# DPO Study Guide

Status: repo-native offline DPO trainer implemented; FinChain pair generation
and rollout sharding are implemented; industry TRL/Axolotl DPO is documented as
the scale-up path.

## 1. Plain-English Idea

Direct Preference Optimization trains a model from preference pairs:

```text
prompt
chosen answer
rejected answer
```

The policy should assign higher probability to the chosen answer than to the
rejected answer. A frozen reference model stays in the objective so the policy
does not win by drifting arbitrarily far from the starting model.

For this repo, DPO is an offline comparator:

1. Generate completions from an SFT checkpoint.
2. Score them with the FinChain verifier.
3. Write fixed chosen/rejected pairs.
4. Train on those pairs.

That is not OPD. OPD refreshes signal from the current policy as the policy
changes.

## 2. The Math

For one pair, define the policy and reference margins:

```text
policy_margin = log pi_theta(chosen | prompt) - log pi_theta(rejected | prompt)
ref_margin    = log pi_ref(chosen | prompt)   - log pi_ref(rejected | prompt)
```

DPO optimizes:

```text
loss = -log sigmoid(beta * (policy_margin - ref_margin))
```

If the policy improves the chosen-vs-rejected margin more than the reference
does, the term inside the sigmoid is positive and the loss falls.

`beta` controls how strongly the objective reacts to preference-margin changes.
Larger beta means stronger pressure to separate chosen from rejected; too large
can overfit noisy pairs.

## 3. Weighted DPO Loss

Weighted DPO is not a new algorithm. It is the same per-pair DPO loss with a
per-example scalar before the batch reduction:

```text
plain:    mean(loss_i)
weighted: sum(w_i * loss_i) / sum(w_i)
```

Use it when some pairs are more trustworthy or more useful. In this repo, it is
reasonable for OPD bridge pairs if ambiguous prompts should dominate training.
It should not be used to blur DPO and OPD conceptually:

- DPO describes the pairwise loss.
- OPD describes how training signal is produced online from the current policy.

## 4. Repo Implementation Map

- `scripts/build_dpo_pairs.py` generates offline verifier-labeled pairs.
- `scripts/merge_dpo_pair_shards.py` merges multi-GPU rollout shards.
- `src/finpost/training/preference_data.py` loads and tokenizes pair JSONL.
- `src/finpost/training/dpo.py` implements sequence log-probs and the DPO loss.
- `src/finpost/training/dpo_train.py` implements the repo-native DPO trainer.
- `experiments/dpo/finchain_qwen25_1_5b.yaml` is the FinChain 1.5B RunPod config.
- `notebooks/finchain_10_dpo_runpod.ipynb` is the operator notebook.

The most important efficiency feature is cached frozen-reference log-probs:
the trainer can run one policy forward per batch instead of both policy and
reference forwards on every step.

## 5. RunPod Recipe

Single-GPU pair generation:

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/build_dpo_pairs.py \
  --model-checkpoint results/checkpoints/qwen25-1p5b-finchain-sft-hf \
  --sources finchain \
  --out-dir results/finchain_pairs/run_001/single_gpu \
  --heldout-train-n 256 \
  --samples-per-prompt 4 \
  --generation-batch-size 16 \
  --max-new-tokens-finchain 512 \
  --max-pairs-per-prompt 4
```

Two-GPU rollout parallelism:

```bash
export CKPT=results/checkpoints/qwen25-1p5b-finchain-sft-hf
export OUT=results/finchain_pairs/run_001

CUDA_VISIBLE_DEVICES=0 python scripts/build_dpo_pairs.py --model-checkpoint $CKPT --sources finchain --out-dir $OUT/shards/shard-00-of-02 --heldout-train-n 2000 --samples-per-prompt 8 --generation-batch-size 64 --max-new-tokens-finchain 768 --max-pairs-per-prompt 8 --shard-id 0 --num-shards 2 &
CUDA_VISIBLE_DEVICES=1 python scripts/build_dpo_pairs.py --model-checkpoint $CKPT --sources finchain --out-dir $OUT/shards/shard-01-of-02 --heldout-train-n 2000 --samples-per-prompt 8 --generation-batch-size 64 --max-new-tokens-finchain 768 --max-pairs-per-prompt 8 --shard-id 1 --num-shards 2 &
wait

python scripts/merge_dpo_pair_shards.py \
  --shard-dirs $OUT/shards/shard-00-of-02 $OUT/shards/shard-01-of-02 \
  --out-dir $OUT/merged
```

Train:

```bash
WANDB_MODE=offline python -m finpost.training.dpo_train \
  --config experiments/dpo/finchain_qwen25_1_5b.yaml \
  --device cuda \
  --max-steps 20
```

Remove `--max-steps 20` only after the canary writes a checkpoint and the loss
is finite.

## 6. Industry Package Mapping

- Hugging Face TRL `DPOTrainer`: standard library interface for DPO once the
  pair dataset contract is understood.
- Axolotl: good when you want config-driven LoRA/QLoRA DPO with fewer custom
  launch details.
- OpenRLHF and verl: more relevant when pairwise DPO becomes part of a larger
  distributed RLHF/RLVR system.

The resume value is highest if you can explain both levels: the repo-native
loss and cache mechanics, then the production trainer call that replaces the
hand-built loop.
