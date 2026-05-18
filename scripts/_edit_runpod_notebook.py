"""One-shot editor for sft_phase1_runpod_ablation_2000.ipynb.

Applies the five edits from .scratch/runpod-canary-and-runbook/issues/02-notebook-edits.md:
  1. Expand pre-flight markdown (cell index 0).
  2. Change `max(50, ...)` to `max(5, ...)` in hyperparameters cell (index 6).
  3. Insert canary cell + monitoring cell after YAML generation (after index 8).
  4. Insert per-arm convert+cleanup cell after each arm training cell.
  5. Insert monitoring cell after arm 3 convert+cleanup.
  6. Repurpose verify cell (was index 18) to verify HF dirs.
  7. Delete the batch convert-all cell (was index 20).

Run once: ``python scripts/_edit_runpod_notebook.py``.

This script is a one-shot — once the notebook is edited, the script's
job is done. It's committed alongside the edit so the diff is auditable
and the procedure is replayable (e.g. if the underlying notebook is
restored from git and the edits need to be re-applied).
"""

from __future__ import annotations

import json
from pathlib import Path

NOTEBOOK = Path("notebooks/sft_phase1_runpod_ablation_2000.ipynb")


def code_cell(source: str) -> dict:
    """Build a code cell dict matching the notebook's existing shape."""
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source.splitlines(keepends=True),
    }


def markdown_cell(source: str) -> dict:
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": source.splitlines(keepends=True),
    }


# ------- new cell contents ---------------------------------------------------

PREFLIGHT_MD = """## Before you run a single cell in this notebook — terminal pre-flight

Open a JupyterLab terminal (**File > New > Terminal**) and run the commands below in order. Each guards against a failure mode that has actually cost money on prior runs. Do this BEFORE clicking any cell in the notebook.

### 1. Confirm the pod has enough disk

```bash
df -h /workspace
```

Look at the **Avail** column on the `/workspace` row. You need at least **40 GB free**. If you have less, terminate the pod and re-deploy with a larger Container Disk / Volume Disk size. Trying to train with less will hit "No space left on device" mid-run and you'll have to start over.

### 2. Confirm the GPU is the one you asked for, and nothing else is using it

```bash
nvidia-smi
```

The **Product Name** row should say `NVIDIA RTX 6000 Ada Generation`, `NVIDIA A40`, or `NVIDIA RTX A6000` (any 48 GB chip). The **Memory-Usage** column should show ~0 MB used and ~48 GB total. If a process is listed under the "Processes" section using GPU memory and you haven't started anything yet, it's a leftover from a previous session — kill it before launching:

```bash
nvidia-smi --query-compute-apps=pid --format=csv,noheader
kill -9 <pid_from_above>
```

### 3. Pull the latest code

```bash
cd /workspace/finpost
git status   # if anything shows as 'deleted:', run: git restore .
git pull
```

### 4. Install the project (first time on this pod only)

```bash
pip install -e ".[dev]"
```

This takes 8–15 minutes. Long silences during the "Installing collected packages" phase are normal — do not interrupt it.

### 5. Confirm the install worked

```bash
python -c "import finpost; print(finpost.__file__)"
```

Expected output: `/workspace/finpost/src/finpost/__init__.py`. If you instead see `ModuleNotFoundError: No module named 'finpost'` despite the pip install succeeding, the editable install's PEP 660 hook didn't get written. Manual fix:

```bash
echo "/workspace/finpost/src" > /usr/local/lib/python3.11/dist-packages/finpost.pth
python -c "import finpost; print(finpost.__file__)"   # retry; should now print the path
```

### 6. Confirm torch sees the GPU

```bash
python -c "import torch; print('cuda:', torch.cuda.is_available())"
```

Expected output: `cuda: True`. If you see `cuda: False`, the installed torch wheel was built for a newer CUDA than the pod's driver supports. Downgrade torch to a CUDA 12.4 wheel:

```bash
pip install "torch==2.4.1+cu124" --index-url https://download.pytorch.org/whl/cu124 --force-reinstall --no-deps
python -c "import torch; print('cuda:', torch.cuda.is_available())"   # retry
```

### 7. Sanity-check one more time

```bash
df -h /workspace && nvidia-smi
```

Confirm disk and GPU still look healthy. You're now ready to run the notebook cells below in order.

---

Full troubleshooting table in [`docs/runbooks/runpod-end-to-end.md`](../docs/runbooks/runpod-end-to-end.md) (or the standalone `runpod-end-to-end.html` for offline reading). Subsequent kernel restarts on the same pod do not need re-installation.

---
"""

CANARY_CELL = """# Step 3.5 — 50-step canary on the combined arm
#
# This cell is the cheapest safety check we have. It writes a one-off YAML
# that mirrors the production recipe but runs for only 50 optimizer steps,
# then subprocess-launches the trainer against it. Pass criterion is
# subprocess exit code 0 — the trainer raises RuntimeError on non-finite
# loss (see `_check_finite_loss` in src/finpost/training/_guards.py), so a
# clean exit means no NaN appeared in 50 steps at production hyperparams.
#
# Cost: ~75 seconds of GPU time, ~$0.01. Worth it every run.
#
# IMPORTANT: the canary overrides BOTH `max_steps` AND `warmup_steps`.
# Inheriting the production `warmup_steps=200` would trip pydantic's
# `warmup_steps < max_steps` validator and produce a misleading config
# load error before the model ever starts training.

import subprocess

CANARY_STEPS = 50
canary_warmup = max(5, CANARY_STEPS // 10)

canary_cfg = build_config('combined', ARMS['combined'])
canary_cfg['training']['max_steps'] = CANARY_STEPS
canary_cfg['training']['warmup_steps'] = canary_warmup
canary_cfg['training']['val_every_n_steps'] = CANARY_STEPS  # one val pass at the end
canary_cfg['training']['checkpoint_every_n_steps'] = CANARY_STEPS  # one ckpt at end
canary_cfg['checkpointing']['save_dir'] = 'results/checkpoints/qwen-canary-50s-a40'
canary_cfg['checkpointing']['retention_last_n'] = 1
canary_cfg['logging']['run_name'] = 'qwen-canary-50s-a40'

canary_yaml = Path(f'{EXPERIMENTS_DIR}/canary_50_a40.yaml')
canary_yaml.write_text(yaml.safe_dump(canary_cfg, sort_keys=False), encoding='utf-8')
print(f'canary config: {canary_yaml}')
print(f'canary steps:  {CANARY_STEPS}   canary warmup: {canary_warmup}')
print()

result = subprocess.run(
    ['python', '-m', 'finpost.training.train', '--config', str(canary_yaml), '--device', 'cuda'],
    capture_output=True,
    text=True,
    check=False,
)
print('--- canary stdout (tail) ---')
print('\\n'.join(result.stdout.splitlines()[-30:]))
if result.returncode != 0:
    print('--- canary stderr (tail) ---')
    print('\\n'.join(result.stderr.splitlines()[-30:]))
    print()
    print('\\n✗ CANARY FAILED — DO NOT launch full 2000-step run')
    print('   inspect stderr above; common causes:')
    print('     - Non-finite loss: trainer logic regression or hardware/SDPA issue')
    print('     - OOM: lower per_device_batch_size in hyperparams cell, retry canary')
    print('     - Config error: verify warmup_steps < max_steps in canary YAML')
    raise RuntimeError(f'Canary subprocess returned {result.returncode}; aborting.')

print()
print('\\n✓ CANARY PASSED — safe to launch full 2000-step run\\n')
"""

POST_CANARY_MONITOR = """# Disk + GPU snapshot after the canary, before launching the full run.
# Healthy values: /workspace has ≥30 GB still free; nvidia-smi shows ~0 MB
# used (the canary subprocess freed its allocations on exit). If GPU shows
# residual memory, run `nvidia-smi --query-compute-apps=pid --format=csv,noheader`
# in a terminal and `kill -9 <pid>` before launching arm 1.
!df -h /workspace; nvidia-smi
"""


def make_convert_cleanup_cell(arm_name: str) -> str:
    """Per-arm convert→cleanup: convert each step to HF, then rm raw step-* dirs."""
    return f"""# Convert {arm_name}'s intermediate checkpoints to HF format, then delete the
# raw step-* directories to free disk before the next arm trains. Each raw
# checkpoint is ~3 GB (weights + optimizer state); the HF copy is ~1 GB
# (weights only). We drop optimizer state because we don't need to RESUME
# from these steps — we only need to EVAL them. Cuts peak disk from ~48 GB
# to ~24 GB.
import shutil

arm_name = '{arm_name}'
run_name = f'qwen-{{arm_name}}-{{ABLATION_STEPS}}s-a40'

for step in INTERMEDIATE_STEPS:
    src = f'{{CHECKPOINTS_DIR}}/{{run_name}}/step-{{step:08d}}'
    dst = f'{{CHECKPOINTS_DIR}}/{{run_name}}-step{{step}}-hf'
    print(f'\\n[{{arm_name}}] converting step-{{step}}:')
    print(f'  from: {{src}}')
    print(f'  to:   {{dst}}')
    !python scripts/convert_checkpoint_to_hf.py \\
        --checkpoint-dir {{src}} \\
        --out-dir {{dst}} \\
        --dtype {{DTYPE}}
    hf_paths[(arm_name, step)] = dst

# All 4 conversions for this arm done — safe to drop the raw checkpoints.
for step in INTERMEDIATE_STEPS:
    raw = Path(f'{{CHECKPOINTS_DIR}}/{{run_name}}/step-{{step:08d}}')
    if raw.exists():
        shutil.rmtree(raw)
        print(f'[{{arm_name}}] deleted raw {{raw}}')

print(f'\\n[{{arm_name}}] disk after cleanup:')
!df -h /workspace
"""


HF_PATHS_INIT_CELL = """# Initialise the (arm, step) -> HF path mapping. Populated by the per-arm
# convert+cleanup cells below. Eval reads from this mapping at the end.
hf_paths: dict[tuple[str, int], str] = {}
"""


VERIFY_HF_CELL = """# Verify the HF-converted directory landed for every (arm, step). The
# per-arm convert+cleanup cells have already deleted the raw step-* dirs,
# so we verify the HF copies (the input to eval) instead of the raw
# checkpoints. Any missing dir here is a hard failure — eval can't run
# without it.
missing: list[str] = []
for arm_name in ARMS.keys():
    run_name = f'qwen-{arm_name}-{ABLATION_STEPS}s-a40'
    print(f'\\n{arm_name}:')
    for step in INTERMEDIATE_STEPS:
        hf_dir = Path(f'{CHECKPOINTS_DIR}/{run_name}-step{step}-hf')
        ok = hf_dir.exists()
        marker = 'OK ' if ok else 'MISSING'
        print(f'  step-{step:>5} hf: {marker}  {hf_dir}')
        if not ok:
            missing.append(str(hf_dir))
        else:
            files = sorted(p.name for p in hf_dir.iterdir())
            print(f'    files: {files}')

if missing:
    raise FileNotFoundError(f'Missing HF checkpoint dirs: {missing}')
"""


VERIFY_HF_MD = """## Step 5 — Verify the converted HF checkpoints landed

The per-arm convert+cleanup cells above have already deleted the raw `step-*` directories, so verifying raw checkpoint existence no longer applies — we verify the HF-converted copies (which eval reads from) instead.

Every arm should have four HF-format directories on disk: `<run_name>-step{500,1000,1500,2000}-hf`. Any missing here is a hard failure rather than a warning.
"""


POST_ARM3_MONITOR = """# Disk + GPU snapshot after all three arms have trained, converted, and
# cleaned up. Healthy values: /workspace has ≥30 GB still free; nvidia-smi
# shows ~0 MB used. If GPU shows residual memory, kill the offending
# process before launching the eval cells below.
!df -h /workspace; nvidia-smi
"""


# ------- edit logic ----------------------------------------------------------

def main() -> None:
    nb = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    cells = nb["cells"]

    # 1. Replace cell 0 (pre-flight markdown).
    assert cells[0]["cell_type"] == "markdown" and "Before running this notebook" in "".join(cells[0]["source"])
    cells[0] = markdown_cell(PREFLIGHT_MD)
    cells[0]["id"] = "8368cc33"  # preserve original ID so git diff is minimal

    # 2. Change `max(50, ...)` to `max(5, ...)` in hyperparameter cell (index 6).
    hp_src = "".join(cells[6]["source"])
    assert "max(50, ABLATION_STEPS // 10)" in hp_src, "hyperparameters cell not at expected index 6"
    cells[6]["source"] = hp_src.replace(
        "max(50, ABLATION_STEPS // 10)",
        "max(5, ABLATION_STEPS // 10)",
    ).splitlines(keepends=True)

    # 3. Insert hf_paths init cell + canary cell + post-canary monitor cell
    #    after cell 8 (YAML generation), before cell 9 ("Step 4 — Train each arm").
    #    Order matters: init mapping FIRST so per-arm cells can append to it.
    insertion_point_after_yaml = 9  # cells.insert(9, ...) puts new cell at index 9
    cells.insert(insertion_point_after_yaml, code_cell(HF_PATHS_INIT_CELL))
    cells.insert(insertion_point_after_yaml + 1, markdown_cell(
        "## Step 3.5 — Canary: 50-step smoke test before paying for the full run\n\n"
        "If this cell raises, do NOT continue to the training cells below — the recipe is broken on this hardware. "
        "Cost: ~$0.01. Saves you from a multi-arm production run that would die partway through."
    ))
    cells.insert(insertion_point_after_yaml + 2, code_cell(CANARY_CELL))
    cells.insert(insertion_point_after_yaml + 3, code_cell(POST_CANARY_MONITOR))

    # After those 4 inserts, indices have shifted by +4. Re-locate the arm training cells.
    # Original indices for arm 1/2/3 training: 12, 14, 16. Now: 16, 18, 20.
    # Find them by content rather than relying on the shift math.
    arm_cells: list[tuple[int, str]] = []
    for i, c in enumerate(cells):
        if c["cell_type"] != "code":
            continue
        src = "".join(c["source"])
        if "yaml_paths['gsm8k_only']" in src and "finpost.training.train" in src:
            arm_cells.append((i, "gsm8k_only"))
        elif "yaml_paths['math_only']" in src and "finpost.training.train" in src:
            arm_cells.append((i, "math_only"))
        elif "yaml_paths['combined']" in src and "finpost.training.train" in src:
            arm_cells.append((i, "combined"))
    assert len(arm_cells) == 3, f"expected 3 arm cells, found {arm_cells}"

    # 4. Insert per-arm convert+cleanup after each arm training cell.
    #    Work from highest index to lowest so earlier inserts don't shift later targets.
    for arm_idx, arm_name in sorted(arm_cells, key=lambda x: -x[0]):
        cells.insert(arm_idx + 1, code_cell(make_convert_cleanup_cell(arm_name)))

    # After arm-3 convert+cleanup (which is now 2 cells past the original arm-3 index plus
    # earlier shifts), insert the post-arm3 monitor cell.
    # Find it: the last cell that includes `arm_name = 'combined'` in its source.
    last_combined_idx = -1
    for i, c in enumerate(cells):
        if c["cell_type"] == "code" and "arm_name = 'combined'" in "".join(c["source"]):
            last_combined_idx = i
    assert last_combined_idx >= 0, "could not find post-combined convert+cleanup cell"
    cells.insert(last_combined_idx + 1, code_cell(POST_ARM3_MONITOR))

    # 5. Repurpose the old verify-raw-checkpoints cell to verify HF dirs.
    #    Find by content: it has 'Verify every intermediate checkpoint exists for every arm'.
    for i, c in enumerate(cells):
        if c["cell_type"] == "code" and "Verify every intermediate checkpoint exists" in "".join(c["source"]):
            cells[i] = code_cell(VERIFY_HF_CELL)
            # Also update the markdown header above it (Step 5 — Verify intermediate checkpoints landed).
            for j in range(i - 1, max(i - 5, -1), -1):
                if cells[j]["cell_type"] == "markdown" and "Step 5" in "".join(cells[j]["source"]):
                    cells[j] = markdown_cell(VERIFY_HF_MD)
                    break
            break

    # 6. Delete the old batch convert-all cell. Find by content + its markdown header.
    convert_all_idx = -1
    convert_all_md_idx = -1
    for i, c in enumerate(cells):
        if c["cell_type"] == "code" and "Convert every intermediate checkpoint per arm to HF format" in "".join(c["source"]):
            convert_all_idx = i
            # Its preceding markdown is "Step 6 — Convert every intermediate checkpoint to HF format"
            for j in range(i - 1, max(i - 3, -1), -1):
                if cells[j]["cell_type"] == "markdown" and "Step 6" in "".join(cells[j]["source"]) and "Convert every intermediate" in "".join(cells[j]["source"]):
                    convert_all_md_idx = j
                    break
            break
    assert convert_all_idx >= 0, "could not find original Step 6 convert-all cell"
    # Delete higher index first so the lower one stays at the same position.
    del cells[convert_all_idx]
    if convert_all_md_idx >= 0:
        del cells[convert_all_md_idx]

    # Write back, preserving the rest of the notebook metadata exactly.
    NOTEBOOK.write_text(
        json.dumps(nb, indent=1, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"Notebook updated. Total cells: {len(cells)} (was 31).")


if __name__ == "__main__":
    main()
