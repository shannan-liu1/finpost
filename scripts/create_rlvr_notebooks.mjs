import { existsSync, writeFileSync } from "node:fs";

function markdown(source) {
  return {
    cell_type: "markdown",
    metadata: {},
    source: source.trim().split("\n").map((line) => `${line}\n`),
  };
}

function code(source) {
  return {
    cell_type: "code",
    execution_count: null,
    metadata: {},
    outputs: [],
    source: source.trim().split("\n").map((line) => `${line}\n`),
  };
}

function notebook(cells) {
  return {
    cells,
    metadata: {
      kernelspec: {
        display_name: "Python 3",
        language: "python",
        name: "python3",
      },
      language_info: {
        name: "python",
        version: "3.11",
      },
    },
    nbformat: 4,
    nbformat_minor: 5,
  };
}

const setupCell = code(`
from pathlib import Path
import importlib
import json
import os
import platform
import subprocess
import sys
import time

PROJECT_ROOT = Path.cwd()
if not (PROJECT_ROOT / "pyproject.toml").exists():
    PROJECT_ROOT = Path("/workspace/finpost") if Path("/workspace/finpost").exists() else PROJECT_ROOT

RESULTS_DIR = PROJECT_ROOT / "results" / "finchain_rlvr"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

def progress(title, detail=None):
    stamp = time.strftime("%H:%M:%S")
    print(f"[{stamp}] {title}")
    if detail:
        print(detail)

def check_module(name):
    try:
        return importlib.import_module(name), None
    except Exception as exc:
        return None, exc

def run_cmd(cmd, *, check=False):
    progress("running command", cmd)
    completed = subprocess.run(cmd, shell=True, text=True, capture_output=True)
    if completed.stdout:
        print(completed.stdout)
    if completed.stderr:
        print(completed.stderr)
    if check and completed.returncode != 0:
        raise RuntimeError(f"command failed with exit {completed.returncode}: {cmd}")
    return completed

def append_cost_event(stage, **payload):
    path = RESULTS_DIR / "cost_ledger.jsonl"
    row = {"stage": stage, "time": time.strftime("%Y-%m-%dT%H:%M:%S"), **payload}
    with path.open("a", encoding="utf-8") as fp:
        fp.write(json.dumps(row, sort_keys=True) + "\\n")
    print(json.dumps(row, indent=2, sort_keys=True))
    return row

progress("project root", str(PROJECT_ROOT))
progress("python", sys.version.split()[0])
progress("platform", platform.platform())
`);

const gpuCell = code(`
progress("GPU preflight")
try:
    import torch
    print("torch:", torch.__version__)
    print("cuda available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        for idx in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(idx)
            print(f"gpu {idx}: {props.name}, vram={props.total_memory / 1e9:.1f} GB")
except Exception as exc:
    print("torch check failed:", repr(exc))

run_cmd("nvidia-smi")
`);

const dependencyCell = code(`
progress("repo dependency preflight")
for module_name in [
    "finpost.training.trainer",
    "finpost.training.dpo",
    "finpost.training.preference_data",
]:
    module, exc = check_module(module_name)
    print(module_name, "OK" if module else f"MISSING: {exc}")

for planned in [
    "finpost.data.finchain",
    "finpost.evals.finchain_metrics",
    "finpost.posttraining.rollout_cache",
    "finpost.posttraining.verifier",
    "finpost.posttraining.opd",
    "finpost.posttraining.grpo",
]:
    module, exc = check_module(planned)
    print(planned, "OK" if module else "planned, not implemented yet")
`);

const ledgerCell = code(`
append_cost_event(
    stage="notebook_preflight",
    notebook=Path(globals().get("__vsc_ipynb_file__", "unknown")).name,
    gpu_count=0,
    notes="preflight cell executed; update this ledger in each expensive stage",
)
`);

const NOTEBOOKS = [
  {
    path: "notebooks/finchain_00_dataset_and_verifier.ipynb",
    title: "FinChain 00 - Dataset and verifier lab",
    problem: "Before training, prove that examples load, prompts render, gold answers verify, and corrupted answers fail.",
    cells: [
      markdown(`# FinChain 00 - Dataset and verifier lab

This is the first RLVR notebook. Its job is to make the reward substrate visible before any GPU spend.

Motivating question: if a model gives a fluent finance answer, how do we know it is actually correct?

This notebook should eventually show: a raw FinChain example, the prompt we send to the model, the executable chain or answer verifier, a gold pass, and several intentional failures.`),
      setupCell,
      gpuCell,
      dependencyCell,
      code(`
progress("planned loader contract")
print("Expected module: finpost.data.finchain")
print("Expected functions: load_finchain_split, render_finchain_prompt")
print("Expected verifier: verify_finchain_completion")
print()
print("Until those functions exist, use this notebook as the implementation checklist.")
`),
      code(`
sample = {
    "question": "Revenue increased from 12.4 to 14.1. What was the growth rate?",
    "chain": "(14.1 - 12.4) / 12.4",
    "answer": 0.1370967742,
}
print(json.dumps(sample, indent=2))
print("Manual verifier sketch:", abs(eval(sample["chain"]) - sample["answer"]) < 1e-9)
`),
      ledgerCell,
      markdown(`## Exit gate

- Gold examples verify.
- Corrupted final answers fail.
- Parse failures are separate from reasoning failures.
- The notebook displays examples by topic/template before training starts.`),
    ],
  },
  {
    path: "notebooks/finchain_00_model_bakeoff.ipynb",
    title: "FinChain 00 - Model bake-off",
    problem: "Pick one serious model empirically before training.",
    cells: [
      markdown(`# FinChain 00 - Model bake-off

This notebook compares candidate models on a small FinChain subset before committing to a training run.

The point is not to crown a leaderboard winner. The point is to choose the model with the best mix of parseability, baseline accuracy, speed, memory footprint, and tooling friction.`),
      setupCell,
      gpuCell,
      dependencyCell,
      code(`
CANDIDATES = [
    "Qwen/Qwen3-4B-Base",
    "Qwen/Qwen2.5-3B-Base",
    "Qwen/Qwen2.5-0.5B",
]
for idx, model_id in enumerate(CANDIDATES, start=1):
    print(f"{idx}. {model_id}")
`),
      code(`
progress("bake-off metrics contract")
columns = [
    "model_id",
    "examples",
    "accuracy",
    "parseability",
    "avg_output_tokens",
    "tokens_per_sec",
    "peak_vram_gb",
    "failure_modes",
]
print("\\n".join(columns))
`),
      ledgerCell,
      markdown(`## Exit gate

Pick one main model and one canary. Do not continue with a model zoo.`),
    ],
  },
  {
    path: "notebooks/finchain_01_sft_lora.ipynb",
    title: "FinChain 01 - LoRA SFT",
    problem: "Train the finance SFT anchor without rewriting the existing SFT notebooks.",
    cells: [
      markdown(`# FinChain 01 - LoRA SFT

Existing SFT notebooks remain the Phase 1 evidence. This notebook is the future FinChain SFT surface.

SFT is not the headline RLVR method. It teaches the model the task interface: finance vocabulary, reasoning trace style, and parseable final answers.`),
      setupCell,
      gpuCell,
      dependencyCell,
      code(`
SFT_CONFIG = {
    "model_id": "Qwen/Qwen3-4B-Base",
    "adapter": "LoRA or QLoRA",
    "max_seq_len": 2048,
    "early_dry_run_steps": 10,
    "notes": "Do not launch the full run until the FinChain verifier notebook passes.",
}
print(json.dumps(SFT_CONFIG, indent=2))
`),
      code(`
progress("existing SFT notebooks kept as-is")
for path in [
    PROJECT_ROOT / "notebooks" / "sft_phase1_runpod_ablation_2000.ipynb",
    PROJECT_ROOT / "notebooks" / "sft_phase1_multitask.ipynb",
]:
    print(path.name, "exists" if path.exists() else "missing")
`),
      ledgerCell,
      markdown(`## Exit gate

- A 10-step dry run completes.
- Loss moves without NaNs.
- Eval parseability improves or stays stable.
- Cost ledger records runtime and GPU type.`),
    ],
  },
  {
    path: "notebooks/finchain_02_rollouts_and_buckets.ipynb",
    title: "FinChain 02 - Rollouts and buckets",
    problem: "Build the shared substrate for rejection SFT, OPD, and GRPO.",
    cells: [
      markdown(`# FinChain 02 - Rollouts and buckets

This notebook is the center of the RLVR workflow. It samples completions, verifies them, assigns difficulty buckets, and writes the rollout cache.

The key idea: spend most additional sampling on ambiguous prompts, not on prompts the model always gets right or always gets wrong.`),
      setupCell,
      gpuCell,
      dependencyCell,
      code(`
def bucket_from_p_correct(p):
    if p >= 0.8:
        return "easy", 0, 0.25
    if p <= 0.2:
        return "hard", 0, 0.5
    return "ambiguous", 12, 1.0

for p in [0.0, 0.25, 0.5, 0.75, 1.0]:
    print(p, "->", bucket_from_p_correct(p))
`),
      code(`
ROLLOUT_SCHEMA = {
    "prompt_id": "string",
    "model_revision": "string",
    "sampling_hash": "string",
    "completion": "string",
    "parsed_answer": "string | number | null",
    "verified": "bool",
    "reward": "float",
    "failure_reason": "string | null",
}
print(json.dumps(ROLLOUT_SCHEMA, indent=2))
`),
      ledgerCell,
      markdown(`## Exit gate

- Rollouts are cached by model revision and sampling parameters.
- Bucket proportions are visible.
- Parse failures and wrong-answer failures are counted separately.`),
    ],
  },
  {
    path: "notebooks/finchain_03_rejection_sft_and_opd.ipynb",
    title: "FinChain 03 - Rejection SFT and OPD",
    problem: "Compare verified-positive self-training against on-policy pairwise preference learning.",
    cells: [
      markdown(`# FinChain 03 - Rejection SFT and OPD

This notebook tests whether the model benefits more from verified positives alone or from chosen/rejected pairs sampled from its own current behavior.

OPD is the bridge from DPO to RLVR: DPO-style loss, but on-policy verifier-labeled pairs.`),
      setupCell,
      gpuCell,
      dependencyCell,
      code(`
progress("OPD pair construction sketch")
rollouts = [
    {"prompt_id": "p1", "completion": "correct chain", "verified": True, "bucket": "ambiguous"},
    {"prompt_id": "p1", "completion": "wrong chain", "verified": False, "bucket": "ambiguous"},
]
chosen = [r for r in rollouts if r["verified"]]
rejected = [r for r in rollouts if not r["verified"]]
print("chosen", len(chosen), "rejected", len(rejected), "pairs", min(len(chosen), len(rejected)))
`),
      code(`
METHOD_TABLE = [
    {"method": "SFT", "uses_rollouts": False, "uses_pairs": False, "required": True},
    {"method": "Rejection SFT", "uses_rollouts": True, "uses_pairs": False, "required": True},
    {"method": "Uniform OPD", "uses_rollouts": True, "uses_pairs": True, "required": "ablation"},
    {"method": "Adaptive OPD", "uses_rollouts": True, "uses_pairs": True, "required": True},
]
print(json.dumps(METHOD_TABLE, indent=2))
`),
      ledgerCell,
      markdown(`## Exit gate

- Pair counts by bucket are visible.
- Rejection SFT and OPD share the same rollout cache.
- OPD reports accuracy, parseability, KL proxy, and cost.`),
    ],
  },
  {
    path: "notebooks/finchain_04_grpo.ipynb",
    title: "FinChain 04 - GRPO",
    problem: "Run one controlled grouped RLVR update.",
    cells: [
      markdown(`# FinChain 04 - GRPO

This is the headline RLVR notebook.

GRPO samples a group of completions for each prompt, scores them with the verifier, normalizes rewards within the group, and applies a KL-controlled update. The first run should be boring and controlled: one model, one K, one reward, one KL coefficient, one budget.`),
      setupCell,
      gpuCell,
      dependencyCell,
      code(`
progress("GRPO reward normalization sketch")
rewards = [1.0, 0.0, 1.0, 0.0]
mean = sum(rewards) / len(rewards)
var = sum((r - mean) ** 2 for r in rewards) / len(rewards)
std = var ** 0.5 or 1.0
advantages = [(r - mean) / std for r in rewards]
print("rewards:", rewards)
print("advantages:", [round(a, 3) for a in advantages])
`),
      code(`
GRPO_CONSTRAINTS = {
    "samples_per_prompt": 4,
    "reward": "binary correctness first; shaped reward only after binary works",
    "kl_control": "track against SFT reference",
    "no_grid": True,
}
print(json.dumps(GRPO_CONSTRAINTS, indent=2))
`),
      ledgerCell,
      markdown(`## Exit gate

- Group rewards and normalized advantages are visible.
- KL/reference drift is tracked.
- Failure examples are saved.
- The writeup says whether GRPO improved reasoning, improved format, or reward-hacked.`),
    ],
  },
  {
    path: "notebooks/finchain_05_transfer_and_writeup.ipynb",
    title: "FinChain 05 - Transfer and writeup",
    problem: "Test whether FinChain improvements survive messier finance tasks.",
    cells: [
      markdown(`# FinChain 05 - Transfer and writeup

This notebook prevents the project from mistaking a symbolic wind tunnel for real finance competence.

It compares Base, SFT, Rejection SFT, OPD, and GRPO on held-out FinChain plus a FinQA transfer subset.`),
      setupCell,
      gpuCell,
      dependencyCell,
      code(`
FINAL_COLUMNS = [
    "method",
    "model",
    "train_examples",
    "rollout_tokens",
    "gpu_hours",
    "dollars",
    "parseability",
    "finchain_acc",
    "finqa_acc",
    "notes",
]
print(" | ".join(FINAL_COLUMNS))
`),
      code(`
progress("writeup questions")
questions = [
    "What improved?",
    "What got worse?",
    "Did FinChain transfer to FinQA?",
    "Did any method reward-hack the verifier?",
    "Which result was cheapest per point of accuracy?",
]
for q in questions:
    print("-", q)
`),
      ledgerCell,
      markdown(`## Exit gate

- Final table includes cost.
- Transfer result is reported honestly.
- Failure modes are more prominent than aggregate accuracy if transfer is weak.`),
    ],
  },
  {
    path: "notebooks/finchain_06_distributed_training_lab.ipynb",
    title: "FinChain 06 - Distributed training lab",
    problem: "Learn DDP/FSDP concepts before making the real trainer distributed.",
    cells: [
      markdown(`# FinChain 06 - Distributed training lab

This notebook is intentionally separate from the main RLVR training notebooks.

The purpose is to learn distributed vocabulary and failure modes: rank, world size, process groups, distributed samplers, DDP, FSDP, ZeRO, sharded checkpoints, and when rollout parallelism is enough.`),
      setupCell,
      gpuCell,
      code(`
progress("distributed environment variables")
for key in ["RANK", "LOCAL_RANK", "WORLD_SIZE", "MASTER_ADDR", "MASTER_PORT", "CUDA_VISIBLE_DEVICES"]:
    print(f"{key}={os.environ.get(key)}")
`),
      code(`
progress("sampler intuition without launching distributed processes")
examples = list(range(12))
world_size = 3
for rank in range(world_size):
    shard = examples[rank::world_size]
    print(f"rank {rank} sees {shard}")
`),
      code(`
progress("when to use which scaling mode")
rows = [
    ("parallel rollouts", "sampling throughput bottleneck", "first multi-GPU win for OPD/GRPO"),
    ("DDP", "model fits on each GPU", "throughput and global batch scaling"),
    ("FSDP", "model/optimizer state does not fit", "state sharding and sharded checkpoints"),
    ("tensor parallel", "single layer/model too large", "mostly out of scope for 3B/4B LoRA"),
]
for row in rows:
    print(f"{row[0]:18} | {row[1]:36} | {row[2]}")
`),
      ledgerCell,
      markdown(`## Exit gate

- You can explain rank and world size.
- You can explain why DDP does not reduce model memory.
- You can explain why FSDP/ZeRO reduce per-GPU training state.
- You know why rollout parallelism may be more useful than distributed training for the first RLVR scaling experiment.`),
    ],
  },
];

for (const spec of NOTEBOOKS) {
  if (existsSync(spec.path)) {
    console.log(`Skipping existing ${spec.path}`);
    continue;
  }
  const nb = notebook(spec.cells);
  writeFileSync(spec.path, `${JSON.stringify(nb, null, 2)}\n`, "utf8");
  console.log(`Created ${spec.path}`);
}
