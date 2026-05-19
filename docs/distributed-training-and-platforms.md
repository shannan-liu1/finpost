# Distributed training and GPU platform guide

Status: active learning guide as of 2026-05-19.

This document is the practical primer for the part of post-training roles that
often sounds mysterious from the outside: multi-GPU training, sharding,
distributed launchers, and cloud GPU platforms.

The goal is not to become a cluster engineer overnight. The goal is to learn
enough to run, debug, and explain a small distributed post-training experiment
without pretending that every workload needs a cluster.

## 1. Why multi-GPU matters for post-training roles

Many post-training jobs ask for distributed experience because production
training is limited by memory, throughput, or both.

Examples:

- Full fine-tuning a model can exceed one GPU's memory because parameters,
  gradients, optimizer states, and activations all consume memory.
- DPO/OPD can be memory-hungry because policy and reference models may both be
  present.
- GRPO can be sampling-hungry because each prompt needs a group of completions.
- Large rollout jobs can be throughput-bound even when training itself is small.

The interview trap is thinking "multi-GPU" means one thing. It does not. It can
mean data parallel training, parameter sharding, optimizer sharding, tensor
parallel inference, multi-node networking, sharded checkpoints, or simply
parallel rollout workers.

## 2. The core vocabulary

### Rank

One training process in a distributed job. Usually one rank owns one GPU.

### World size

The total number of ranks. A single machine with 4 GPUs usually runs world size
4.

### Process group

The communication group used by ranks to exchange gradients, parameters, or
other tensors.

### All-reduce

The operation that combines gradients across ranks and gives every rank the
same averaged result.

### Distributed sampler

A sampler that gives each rank a different slice of the dataset so GPUs do not
train on the same examples.

### Sharding

Splitting model state across GPUs so no one GPU stores everything.

## 3. DDP versus FSDP versus DeepSpeed ZeRO

### DistributedDataParallel

DDP replicates the model on every GPU. Each rank sees different data, computes
gradients, and synchronizes gradients with the other ranks.

Use DDP when:

- the model already fits on one GPU,
- you want more throughput,
- you can increase global batch size cleanly,
- checkpointing simplicity matters.

Do not use DDP to solve memory if the model does not fit on one GPU. DDP
replicates the model; it does not shard it.

### Fully Sharded Data Parallel

FSDP shards model parameters, gradients, and optimizer states across ranks. Each
GPU stores only part of the training state and gathers what it needs during
forward/backward.

Use FSDP when:

- full fine-tuning or large adapter training is memory-bound,
- optimizer state is the bottleneck,
- you need to learn sharded checkpoint semantics,
- you are ready to debug distributed state.

FSDP is the most educational sharding path for this repo because PyTorch and
Hugging Face Accelerate expose it directly.

### DeepSpeed ZeRO

DeepSpeed ZeRO is another family of state-sharding techniques:

- Stage 1 shards optimizer states.
- Stage 2 shards optimizer states and gradients.
- Stage 3 shards optimizer states, gradients, and parameters.

Use ZeRO when:

- a framework stack already expects DeepSpeed,
- you want mature configuration-driven scaling,
- you need ZeRO-2/3 behavior and are comfortable with DeepSpeed's checkpointing
  model.

For this project, learn FSDP first and map it to ZeRO after. Hugging Face
Accelerate documents the mapping: FSDP `FULL_SHARD` corresponds to ZeRO-3,
`SHARD_GRAD_OP` corresponds to ZeRO-2, and `NO_SHARD` corresponds to normal DDP.

## 4. What should finpost learn first?

Do not distributed-train the main FinChain model first.

The right ladder:

1. Run the single-GPU 3B/4B LoRA path.
2. Run `notebooks/finchain_06_distributed_training_lab.ipynb` on a toy model.
3. Prove DDP equivalence: one GPU global batch N versus two GPUs local batch N/2.
4. Prove FSDP mechanics on a tiny transformer.
5. Add Accelerate config files only after the toy lab works.
6. Move the FinChain SFT/OPD loop to distributed only if the bottleneck is real.

Why:

- If the single-GPU experiment is unclear, multi-GPU makes it unclear faster.
- Distributed bugs often look like ML bugs: divergent loss, hangs, broken
  checkpoint restore, duplicate examples, wrong effective batch.
- The learning value comes from isolating the distributed concept, not from
  immediately mixing it with FinChain, LoRA, DPO, and GRPO.

## 5. Multi-GPU options for this repo

### Option A: parallel rollout workers

This is the easiest useful scaling path.

Run multiple GPUs or machines that each generate completions for a shard of the
prompt set. Merge the rollout JSONL files and verify centrally.

Best for:

- OPD,
- GRPO data collection,
- model bake-offs,
- avoiding distributed-training complexity while still using multiple GPUs.

Why it is underrated:

- Rollouts are embarrassingly parallel.
- Failed workers can be retried without corrupting optimizer state.
- It teaches cost accounting and cluster orchestration without forcing FSDP.

### Option B: DDP adapter training

Use DDP when the model plus adapters fit on each GPU and you want throughput.

Best for:

- LoRA SFT,
- LoRA DPO/OPD if policy/reference memory fits,
- experiments where global batch scaling is the main goal.

Risk:

- Global batch changes learning dynamics. If local batch is 4 on 4 GPUs, global
  batch is 16 before gradient accumulation. The learning rate and number of
  optimizer steps need to be interpreted accordingly.

### Option C: FSDP full fine-tuning

Use FSDP when memory is the actual problem.

Best for:

- full fine-tuning a 3B/4B model,
- learning sharded checkpoints,
- understanding optimizer-state memory.

Risk:

- It is easy to lose time to launch configs, checkpoint merging, and rank-local
  logging.

### Option D: tensor/pipeline parallelism

Mostly out of scope for now.

Use only when a single layer or activation path is too large for one GPU, or when
you are serving/training models much larger than the current 3B/4B target.

## 6. The notebook design for distributed learning

The distributed notebook should be an instructional lab, not a production
launcher.

Cells:

1. Inspect GPUs, CUDA, NCCL, and environment variables.
2. Explain rank/world-size with a drawing and printed examples.
3. Run a CPU-only fake "rank split" to show how a distributed sampler works.
4. If multiple GPUs are available, launch a tiny DDP script.
5. Compare single-process and DDP loss/weights.
6. Show what would change under FSDP.
7. Print a checklist for moving the real trainer to Accelerate/FSDP.

Visible outputs to include:

- rank id,
- local rank,
- world size,
- examples seen per rank,
- effective global batch,
- per-rank loss,
- gradient synchronization sanity check,
- checkpoint paths.

## 7. Cloud platform recommendations

### RunPod

Best use:

- notebook-first single-GPU work,
- A40/L40S/RTX 6000 Ada experiments,
- quick iteration with JupyterLab,
- cheap canaries.

Strengths:

- broad GPU catalog,
- low friction for one-off pods,
- instant clusters now exist for multi-node jobs,
- good fit for "rent a 48GB GPU for a few hours."

Weaknesses:

- availability can be frustrating,
- pod state and volumes require discipline,
- spot/interruptible instances are risky for training,
- multi-GPU is possible but adds platform and launch complexity.

Recommendation:

- Keep RunPod as the notebook surface, but do not make it the only platform in
  the study.

### Lambda Cloud

Best use:

- more conventional GPU VMs,
- stable boxes when available,
- cluster-shaped experiments if you can tolerate commitment/capacity friction.

Strengths:

- clearer cloud posture,
- strong GPU focus,
- cluster products.

Weaknesses:

- availability can be limited,
- some cluster pricing/commitment paths are less flexible than a cheap pod.

Recommendation:

- Use Lambda when you want a cleaner VM or cluster story for a serious
  distributed test.

### Modal

Best use:

- batch rollout jobs,
- reproducible containerized experiments,
- job-shaped workloads rather than long interactive notebooks.

Strengths:

- good developer ergonomics,
- easy GPU function declaration,
- supports multi-GPU containers for several GPU types,
- strong fit for repeatable rollout workers.

Weaknesses:

- less natural for long exploratory notebook sessions,
- persistent interactive state is not the core model.

Recommendation:

- Consider Modal for rollout generation once the prompt/verifier/cache contract
  is stable.

### Vast.ai

Best use:

- cheapest possible GPU experiments,
- non-sensitive work,
- flexible marketplace exploration.

Strengths:

- broad marketplace,
- often low prices,
- API-driven filtering.

Weaknesses:

- host quality varies,
- storage/network details matter,
- debugging platform issues can eat the savings.

Recommendation:

- Use Vast for cheap inference/rollout experiments after the runbook is robust,
  not for first-time distributed training.

## 8. Platform decision matrix

| Need | First choice | Why |
| --- | --- | --- |
| Human notebook learning | RunPod 48GB pod | easiest interactive Jupyter surface |
| Cheapest non-sensitive rollouts | Vast.ai | marketplace pricing, tolerate variability |
| Reproducible rollout jobs | Modal | containerized job ergonomics |
| Stable single GPU VM | Lambda or RunPod secure cloud | less marketplace variability |
| First multi-GPU toy lab | single-node 2x/4x GPU on RunPod or Lambda | avoids multi-node networking first |
| Serious multi-node test | RunPod Instant Clusters or Lambda cluster | needs high-speed networking and Slurm/launcher discipline |

## 9. What to say in interviews

Good:

> I learned distributed training by separating the concepts. First I used a
> single-GPU LoRA workflow. Then I built a toy distributed notebook to show DDP
> gradient synchronization and FSDP state sharding. Only after that would I move
> the FinChain trainer to multi-GPU, because otherwise I would not know whether a
> failure came from RLVR, data, or distributed state.

Better:

> For this project, the first multi-GPU win is probably rollout parallelism, not
> FSDP. OPD and GRPO need many sampled completions; rollouts shard cleanly across
> GPUs and merge through a verifier. I would use FSDP only when training memory,
> not sampling throughput, is the actual bottleneck.

## 10. Sources

- PyTorch DistributedDataParallel: https://docs.pytorch.org/docs/main/generated/torch.nn.parallel.DistributedDataParallel.html
- Hugging Face Accelerate FSDP: https://huggingface.co/docs/accelerate/main/en/usage_guides/fsdp
- DeepSpeed ZeRO: https://www.deepspeed.ai/tutorials/zero/
- RunPod Pods: https://docs.runpod.io/pods
- RunPod Instant Clusters: https://docs.runpod.io/instant-clusters
- Lambda GPU Cloud Pricing: https://lambda.ai/service/gpu-cloud/pricing
- Modal GPU docs: https://modal.com/docs/reference/modal.gpu
- Vast.ai: https://vast.ai/
