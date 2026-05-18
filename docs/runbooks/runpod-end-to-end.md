# RunPod end-to-end runbook — Phase 1 Qwen 0.5B SFT ablation

This is the single document you read while running `notebooks/sft_phase1_runpod_ablation_2000.ipynb` on a fresh RunPod pod. Pod creation through eval download. Every command is shown, every expected output is described, every failure mode that has actually cost money on prior runs is flagged inline.

Designed to be read side-by-side with the notebook. Open the HTML version of this file locally (`docs/runbooks/runpod-end-to-end.html`) and the notebook on the pod; work down both in parallel.

---

## Section 1 — What you'll do and what it costs

You will:

1. Create a 48 GB GPU pod on RunPod (RTX 6000 Ada, RTX A6000, or A40 — any 48 GB chip).
2. Open JupyterLab in your browser.
3. Run a sequence of terminal commands to install the project.
4. Open the notebook and run cells in order.
5. The notebook runs a 50-step canary first (~1 minute, ~$0.01) — if it fails, you stop and investigate.
6. If the canary passes, the notebook runs three 2000-step SFT arms (~45–90 min total, ~$0.33–$0.66).
7. The notebook converts checkpoints to Hugging Face format and deletes the raw weights (~12 minutes, ~$0.09).
8. The notebook runs the eval CLI across all 12 checkpoints on GSM8K + MATH (~30–45 min, ~$0.22–$0.33).
9. You download the eval results via the JupyterLab file browser.
10. You stop the pod.

**Total cost: ~$0.75–$1.10 on RTX 6000 / A40.** Cheaper on A6000 (~$0.49/hr) or A40 (~$0.44/hr); a bit more on RTX 6000 Ada (~$0.79/hr) but the faster chip cuts total wall time.

---

## Section 2 — Before you click anything on RunPod

What to have ready before logging into RunPod:

- **Weights & Biases API key:** OPTIONAL. The notebook auto-detects: if `WANDB_API_KEY` is set in the environment, wandb writes runs to your wandb.ai account in real time; if not, it falls back to OFFLINE mode (logs land in `wandb/offline-run-*/` and never leave the pod). Three ways to set the key when you want online logging:
  1. **RunPod pod environment variable** (recommended for pod runs). When creating the pod in Section 3, scroll down on the deploy form to the "Environment Variables" field. Click "Add Variable", set Key=`WANDB_API_KEY`, Value=`<your key from https://wandb.ai/authorize>`. Persists for the pod's lifetime; never appears in any file or commit.
  2. **Local `.env` file** (recommended for local dev runs of this notebook on a laptop). Create `.env` at the repo root with `WANDB_API_KEY=<your key>`. The file is gitignored. You'd need to manually `export $(cat .env | xargs)` in your shell before launching Jupyter — the notebook itself does NOT auto-load `.env`.
  3. **Manual `export` in the JupyterLab terminal** before opening the notebook: `export WANDB_API_KEY=<your key>`. Works but you re-do it for every pod.

  The notebook's wandb-mode cell prints a one-line confirmation of which mode it picked, with the first/last 4 characters of the key (masked) so you can verify it loaded from the right source.
- **Hugging Face account / token:** OPTIONAL. Only required if you want to push checkpoints to HF Hub at the end via the (commented-out) `hf upload` block in the last cell. Not needed for a normal run.
- **A local browser:** for the RunPod console and JupyterLab. Any modern browser works.
- **A local terminal:** only needed if you want to SCP files instead of using JupyterLab's file browser. JupyterLab's file-browser download is the easier path; SCP is the fallback.

---

## Section 3 — Pod creation, step by step

### 3.1 Log in

Go to <https://www.runpod.io>, sign in.

### 3.2 Go to Pods → Deploy

Click the **Pods** tab in the left sidebar, then click **Deploy** (or **GPU Pods → Deploy** depending on RunPod's current UI).

### 3.3 Pick a GPU

You want a chip with **48 GB VRAM**. RunPod groups GPUs by name. Look for, in this preference order:

1. **RTX 6000 Ada** — fastest, ~$0.79/hr. Newest architecture, ~1.5–1.8× A40 throughput.
2. **RTX A6000** — cheapest of the three, ~$0.49/hr. Roughly A40-equivalent throughput.
3. **A40** — middle option, ~$0.44/hr. The original target chip for this recipe.

**Avoid:** RTX 3090 (24 GB VRAM is too tight for the bs=16 recipe). If you can't find any 48 GB chip, see the "If you can't get 48 GB" sidebar below.

Click the chip you want; RunPod opens a deploy form on the right.

### 3.4 Pick a template

Search for a PyTorch template — anything with **PyTorch 2.4+** on **CUDA 12.x**. RunPod's "PyTorch 2.4.0" or "PyTorch 2.5.0" templates are both fine. The torch version doesn't have to match what we eventually use — we'll adjust it inside the pod.

**Avoid:** any template specifying CUDA 13 or a torch wheel built for it. The pod's NVIDIA driver may not support CUDA 13, and re-installing torch is the most common bootstrap failure (covered in Section 5).

### 3.5 Set disk sizes

This is the step that prevents the "no space left on device" error during training.

- **Container Disk:** set to **at least 50 GB**. The default is sometimes 20 GB which is not enough.
- **Volume Disk:** if shown, set to **at least 50 GB**. The volume persists across pod stops; the container disk does not.

The training run plus the HF cache plus the eval artifacts peak at ~30 GB of disk in use. 50 GB gives you ~20 GB of headroom for log files, kernel files, the venv, etc.

### 3.6 Set the deploy options and deploy

Leave the rest of the defaults alone unless you have a reason to change them. Click **Deploy** (or **Deploy On-Demand** — avoid Spot/Interruptible for this workflow; a mid-run interruption is more expensive than the spot discount saves).

RunPod will show a deployment progress bar. The pod takes 30–90 seconds to become available.

### 3.7 Open the pod's controls

When the pod's status turns green ("Running"), you'll see a card for it in **My Pods**. The card has buttons:
- **Connect** — opens a popup with JupyterLab and SSH connection options.
- **Logs** — useful only if Connect fails to open JupyterLab.
- **Stop / Terminate** — covered in Section 8.

---

### Sidebar — If you can't get 48 GB

Every 48 GB chip is reserved? Two options:

1. **Wait and re-check.** RunPod's chip availability rotates. Try again in 30 minutes.
2. **Fall back to RTX 3090 (24 GB).** The recipe must change. In the hyperparameters cell of the notebook, change:
   - `PER_DEVICE_BATCH_SIZE = 16` → `PER_DEVICE_BATCH_SIZE = 8`
   - `GRAD_ACCUM_STEPS = 1` → `GRAD_ACCUM_STEPS = 2`
   The effective batch stays at 16, training dynamics are identical, each step takes ~2.5–3× as long. Total time: ~75–90 min for the training portion instead of ~45.

Anything smaller than a 24 GB chip (T4, K80, V100 16 GB) cannot run this recipe at all.

---

## Section 4 — Connecting to the pod

### 4.1 Use JupyterLab, not the web terminal

Click **Connect** on the pod card. You'll see options like "Start Web Terminal" and "Connect to Jupyter Lab".

**Use JupyterLab.** Reasons (from prior experience):
- Drag-and-drop file uploads / downloads work in the file browser.
- Copy-paste behaves like a normal browser (Ctrl+C / Ctrl+V).
- The web terminal uses xterm.js, which has flaky copy-paste and no file transfer at all.

### 4.2 If the JupyterLab tab opens

You'll see a JupyterLab interface. The file browser on the left shows `/workspace/`. If the `finpost` directory is already there, great — continue to Section 5. If not, you'll clone it in Section 5.

### 4.3 If the JupyterLab tab fails

Sometimes the JupyterLab service hasn't fully started by the time RunPod says the pod is "Running". Wait 30 seconds, click Connect again. If it still doesn't load, check the pod logs from the pod card.

---

## Section 5 — Terminal pre-flight (before clicking any notebook cell)

In JupyterLab, open a terminal: **File > New > Terminal**. All commands below run there. Run them in order. Each guards against a real failure that has cost money on prior runs.

### 5.1 Confirm the pod has enough disk

```bash
df -h /workspace
```

Look at the **Avail** column on the `/workspace` row:

```
Filesystem      Size  Used Avail Use% Mounted on
overlay         100G   12G   88G  12% /workspace
```

"88G Avail" → you're good. You need at least **40 GB free**.

**If less than 40 GB:** stop now. Use the pod controls to **Terminate** the pod (you'll lose all state), then go back to Section 3 and create a new pod with a larger Container Disk / Volume Disk. Trying to train with less disk will hit "No space left on device" mid-run.

### 5.2 Confirm the GPU is the one you asked for

```bash
nvidia-smi
```

You'll see a table. Look for:

- **Product Name** (top): should say `NVIDIA RTX 6000 Ada Generation`, `NVIDIA RTX A6000`, or `NVIDIA A40`.
- **Memory-Usage** (top right): should show ~0 MiB used / ~48 GB total (e.g., `0MiB / 49140MiB`).
- **Processes** (bottom): should say "No running processes found" or be empty.

If the **Processes** section lists a Python or other process using GPU memory, it's a leftover from a previous session. Kill it:

```bash
nvidia-smi --query-compute-apps=pid --format=csv,noheader
# prints e.g.:
# 12345
kill -9 12345
nvidia-smi   # re-verify Processes is now empty
```

If the **Product Name** is not what you ordered or shows less than 48 GB, you got the wrong chip. Stop, terminate, redeploy.

### 5.3 Pull the latest code

```bash
cd /workspace
ls
```

If you see a `finpost` directory, the repo is already there. Update it:

```bash
cd /workspace/finpost
git status
```

If you see a long list of `deleted:` entries, the working tree got damaged on pod start (this happens sometimes; cause unclear). Restore it:

```bash
git restore .
git status   # must now print "nothing to commit, working tree clean"
```

Then pull:

```bash
git pull
```

If `/workspace/finpost` doesn't exist (first-time pod), clone it:

```bash
cd /workspace
git clone https://github.com/shannan-liu1/finpost.git
cd finpost
```

### 5.4 Install the project as an editable package

```bash
pip install -e ".[dev]"
```

This is a ~3 GB download (torch + CUDA wheels). Expect **8–15 minutes**. The "Installing collected packages" phase goes silent for several minutes as wheels link — **silence is not a hang**. If you want to verify pip is still alive, open a second terminal and run:

```bash
ps aux | grep -E "pip|python" | grep -v grep
```

If `pip` is in the process list, it's working. Wait for the `Successfully installed ...` line.

### 5.5 Confirm the install actually exposes `finpost`

This is the step most likely to fail silently.

```bash
python -c "import finpost; print(finpost.__file__)"
```

**Expected output:** `/workspace/finpost/src/finpost/__init__.py`

**If you instead see `ModuleNotFoundError: No module named 'finpost'`** despite the pip install succeeding, the editable install's PEP 660 import hook wasn't written. This is a known pip / setuptools edge case. Diagnose:

```bash
pip show finpost   # should print version info, confirming pip thinks it's installed
ls /usr/local/lib/python3.11/dist-packages/ | grep -i finpost
```

If you see **only** `finpost-0.0.1.dist-info/` and no `__editable__.finpost-*.pth` or similar, the hook is missing. Write the path file manually:

```bash
echo "/workspace/finpost/src" > /usr/local/lib/python3.11/dist-packages/finpost.pth
python -c "import finpost; print(finpost.__file__)"   # should now print the path
```

Don't spend time chasing the underlying pip/PEP 660 bug — the manual `.pth` is the canonical fallback.

### 5.6 Confirm torch sees the GPU

```bash
python -c "import torch; print('cuda:', torch.cuda.is_available()); print('device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none')"
```

**Expected output:**

```
cuda: True
device: NVIDIA RTX 6000 Ada Generation   (or A40 / RTX A6000)
```

**If you see `cuda: False`**, the torch wheel pip installed was built for a newer CUDA than the pod's driver supports. `pyproject.toml` pins `torch>=2.5`, which makes pip pull a CUDA 13 wheel; many RunPod pods ship with NVIDIA drivers supporting only CUDA 12.x. Downgrade torch:

```bash
pip install "torch==2.4.1+cu124" --index-url https://download.pytorch.org/whl/cu124 --force-reinstall --no-deps
python -c "import torch; print('cuda:', torch.cuda.is_available())"   # should now print True
```

`--no-deps` keeps the rest of the install intact. The warning about mismatched `torchaudio` / `torchvision` is safe to ignore (neither is used by this notebook).

### 5.7 Final sanity check

```bash
df -h /workspace && nvidia-smi
```

Confirm both are still healthy. You're now ready to open the notebook.

---

## Section 6 — Notebook execution, cell by cell

Open `notebooks/sft_phase1_runpod_ablation_2000.ipynb` from the JupyterLab file browser. You'll click **Run** on each cell in order from top to bottom. **Wait for each cell to finish before running the next one.** A cell is finished when the `[*]` next to it turns into a number like `[7]`.

### Cell 1 — Pre-flight markdown

Read-only (markdown). Mirrors Section 5 above. Skim it.

### Cells 2–3 — Title and Step 1 markdown

Read-only. Tells you what to expect.

### Cell 4 — `!nvidia-smi` + `!df -h`

Re-runs the sanity check from inside the notebook. Same expectations as 5.1 and 5.2 above. If anything is off, stop here.

### Cell 5 — `import os`, `import torch`, `import finpost`

Imports + sets `os.chdir(REPO_ROOT)`. Should print:

```
cwd: /workspace/finpost
... finpost: 0.0.1
torch: <some version>
cuda: True
device: NVIDIA RTX 6000 Ada Generation
```

If any of these is wrong, stop and go back to Section 5.

### Cell 6 — Step 2 markdown

Read-only.

### Cell 7 — Hyperparameter cell

Sets `ABLATION_STEPS = 2000`, `MAX_SEQ_LEN = 1024`, etc., and computes `warmup_steps = max(5, ABLATION_STEPS // 10) = 200`. Should print a block of variable names and values. No GPU work here.

### Cell 8 — Step 3 markdown

Read-only.

### Cell 9 — Generate the three YAML configs

Writes `experiments/runpod_a40/{gsm8k_only,math_only,combined}_2000_a40.yaml`. Should print `wrote experiments/runpod_a40/...` three times, then the contents of the `combined` config. No GPU work.

### Cell 10 — Initialise `hf_paths` mapping

One-liner: `hf_paths = {}`. Populated by the per-arm convert cells later.

### Cell 11 — Step 3.5 markdown (canary intro)

Read-only. Tells you the next cell is the gate.

### Cell 12 — **CANARY CELL** (the gate)

This is the cell that protects you from paying for a doomed full run.

What it does:
- Writes a `canary_50_a40.yaml` (50 steps, warmup_steps=5, otherwise identical to production).
- Subprocess-runs the trainer against it.
- Checks the subprocess exit code.

**Expected runtime:** ~1–2 minutes. **Expected output (last lines):**

```
✓ CANARY PASSED — safe to launch full 2000-step run
```

**If you see `✗ CANARY FAILED`:** the cell will raise a `RuntimeError` and Jupyter will halt. **Do NOT run the cells below.** Inspect the stderr printed above the failure message. Common causes:

| What you see in stderr | What it means | What to do |
|---|---|---|
| `RuntimeError: Non-finite loss at step N` | NaN or inf in training. Trainer logic, mask, or hardware issue. | File a bug; do not retry by just re-running — the cause is structural. |
| `torch.cuda.OutOfMemoryError` | GPU ran out of VRAM at this batch size. | Lower `PER_DEVICE_BATCH_SIZE` in cell 7 (try 8). Re-run from cell 7. |
| `ValidationError: warmup_steps ... must be < max_steps` | The canary's warmup override didn't apply. | Verify cell 12 source still computes `canary_warmup = max(5, CANARY_STEPS // 10)`. |
| Anything else | New failure mode. | Read stderr carefully; the message usually points at the cause. |

### Cell 13 — Post-canary monitor (`!df -h /workspace; !nvidia-smi`)

Should show ~30+ GB free disk and ~0 MB GPU memory used (canary released its allocations on subprocess exit). If GPU memory is still in use, see Section 7.

### Cell 14 — Step 4 markdown

Read-only.

### Cell 15 — WANDB mode auto-detect

Inspects `os.environ['WANDB_API_KEY']`. If set, switches wandb to **online** (your runs appear at wandb.ai in real time). If not set, falls back to **offline** (runs written to `wandb/offline-run-*/` on the pod; you can manually sync them later or just ignore them). Prints a one-line confirmation with the first/last 4 chars of the key (masked) when online so you can verify the right key loaded.

If you set the key in RunPod's pod env vars (recommended), it should already be visible — the cell's output should say `WANDB_MODE = online`. If you forgot, just run `export WANDB_API_KEY=<your key>` in a terminal, restart the JupyterLab kernel, and re-run from cell 4.

### Cells 16–17 — Arm 1: gsm8k_only train, then convert+cleanup

Cell 16: `!python -m finpost.training.train --config <gsm8k_only_yaml>`. Runs for ~15–30 min. You'll see training progress lines, periodic validation losses, checkpoint saves at steps 500/1000/1500/2000.

**Expected output (last lines):** the trainer's "Training complete" message and the path of the final checkpoint.

**If the cell raises:** stop. The same failure-cause table as the canary applies. Note that the canary should have caught these, so a failure here is unexpected — most likely a transient OOM if disk or GPU memory got fragmented.

Cell 17: converts the 4 intermediate checkpoints to HF format, then `rm -rf`s the raw `step-*` directories. Saves ~8 GB of disk. Should print 4 `[gsm8k_only] converting step-NNN:` blocks, 4 `[gsm8k_only] deleted raw ...` lines, and a final `df -h` line.

### Cells 18–19 — Arm 2: math_only train, then convert+cleanup

Same pattern as arm 1.

### Cells 20–21 — Arm 3: combined train, then convert+cleanup

Same pattern as arms 1 and 2.

### Cell 22 — Post-arm3 monitor

Should show ~25+ GB free disk and ~0 MB GPU memory used. The interleaved cleanup has kept peak disk under ~25 GB the whole time.

### Cells 23–24 — Step 5 markdown + verify HF dirs

Cell 24 loops over (arm, step) and prints `OK` or `MISSING` for each. Should be all `OK`. If anything is `MISSING`, the convert+cleanup for that arm failed silently — stop and inspect.

### Cells 25–26 — Step 7 markdown + eval

Cell 26 runs `!python -m finpost.evals.eval_exact ...` with 12 checkpoints × 2 sources. Runtime: ~30–45 min. You'll see batched generation progress, then accuracy + cost summaries.

**If the cell raises OOM:** the eval CLI has internal `_generate_chunk_with_oom_fallback` that halves batch size on OOM. If you see "OOM halving batch from 128 to 64" in the output, that's the fallback working — let it continue. If it fails again at 64, lower `BATCH_SIZE_GSM8K` / `BATCH_SIZE_MATH` in cell 7 and re-run only this cell.

### Cells 27–29 — Step 8 markdown + display headline numbers + trajectory table

Loads and prints the eval summary CSV and cost JSON. No GPU work — just file reads.

### Cell 30 — Step 9 markdown

Read-only.

### Cell 31 — Tarball the eval results

`!tar -czf ... && !ls -lah ...`. Creates `results/evals/ablation_a40_run_1.tar.gz`. Should print the file size (~few MB).

### Cells 32–33 — Step 10 (optional HF push)

Commented out by default. Skip unless you specifically want the checkpoints on HF Hub.

### Cell 34 — Final markdown

Read-only. Tells you to download the tarball and stop the pod (Section 8).

---

## Section 7 — Mid-flow monitoring

The two monitoring cells (post-canary and post-arm3) print disk and GPU snapshots. Here's how to read them.

### `df -h /workspace`

Healthy = ≥25 GB **Avail**. If it drops below 10 GB you're in danger of "no space left" during the next operation. Investigate before continuing.

### `nvidia-smi`

Healthy = the **Memory-Usage** column shows ~0 MiB used (the previous subprocess released its allocations on exit). The **Processes** section should be empty.

If Memory-Usage shows non-zero use but Processes is empty: that's "lingering allocation," usually a CUDA context that didn't fully release. Open a terminal and run:

```bash
nvidia-smi --query-compute-apps=pid --format=csv,noheader
```

If it returns a PID, `kill -9 <pid>`. If it returns nothing but the memory is still showing as used, restart the JupyterLab kernel: **Kernel > Restart Kernel**. The next cell will need to re-import torch but otherwise picks up where it left off (the trained checkpoints on disk are untouched by a kernel restart).

---

## Section 8 — After eval: download and stop

### 8.1 Download the eval tarball

In JupyterLab's file browser (left side), navigate to `results/evals/`. Right-click `ablation_a40_run_1.tar.gz` and click **Download**. The file is small (~few MB) — downloads in seconds.

If you also want individual files (CSVs, per-row details), navigate into `results/evals/ablation_a40_run_1/` and download whatever you want.

### 8.2 Optional: push checkpoints to HF Hub

If you want to keep the trained checkpoints accessible without keeping the pod alive, run:

```bash
hf auth login   # in a terminal; paste your Write token
```

Then uncomment cell 33 of the notebook and run it. ~12 GB to upload (4 ckpts × 3 arms × ~1 GB HF format).

### 8.3 Stop or terminate the pod

In the RunPod console, on your pod's card:

- **Stop Pod** — billing pauses, volume persists. Use this if you'll be back within a few days. You pay ~$0.10/day for the stopped volume.
- **Terminate Pod** — frees everything, including the venv and checkpoints. Use this only after you've downloaded everything you care about.

**Verify your cost** from `cost_summary.json` in the eval output. That's the authoritative number; the estimates in Section 1 are planning numbers only.

---

## Section 9 — Troubleshooting appendix

Index of failure modes covered inline above, with pointer to where they live:

| Symptom | Where to read | One-line fix |
|---|---|---|
| "No space left on device" during training | Section 5.1, Section 7 | Confirm ≥40 GB free before launching; if not, redeploy with bigger volume. |
| `cuda: False` after install | Section 5.6 | Downgrade torch to `2.4.1+cu124`. |
| `ModuleNotFoundError: No module named 'finpost'` despite pip install | Section 5.5 | Write `/usr/local/lib/python3.11/dist-packages/finpost.pth` by hand. |
| `git status` shows huge list of `deleted:` files | Section 5.3 | `git restore .`. |
| `pip install` hangs at "Installing collected packages" | Section 5.4 | Wait — silence is not a hang. Verify via `ps aux` from a second terminal. |
| Canary cell raises `RuntimeError: Non-finite loss` | Section 6, Cell 12 | Structural bug. Do not retry blind. File issue. |
| Canary cell raises `torch.cuda.OutOfMemoryError` | Section 6, Cell 12 | Lower `PER_DEVICE_BATCH_SIZE` in cell 7. |
| Canary cell raises `ValidationError: warmup_steps ... must be < max_steps` | Section 6, Cell 12 | Cell source corruption — verify the canary computes its own `warmup_steps`. |
| Training cell raises OOM mid-run after canary passed | Section 7 | Likely GPU-fragmentation. Kernel restart, then re-run the failing arm's training cell. |
| Eval cell raises OOM | Section 6, Cell 26 | Internal fallback halves batch automatically; if it still fails, lower `BATCH_SIZE_GSM8K` / `BATCH_SIZE_MATH` in cell 7 and re-run only the eval cell. |
| `nvidia-smi` shows memory used but no process | Section 7 | Kernel restart. |
| Verify-HF cell prints `MISSING` for some checkpoint | Section 6, Cell 24 | Convert+cleanup for that arm failed silently. Re-run that arm's convert cell. |
| JupyterLab won't open | Section 4.3 | Wait 30s, retry Connect, then check pod Logs. |

---

## Appendix — What this runbook intentionally does NOT cover

- **Multi-chip recipes other than 48 GB.** RTX 3090 fallback is mentioned in the Section 3 sidebar; anything below 24 GB cannot run this recipe.
- **Mid-run pod crash recovery.** Out of scope. The canary + interleaved cleanup is the primary defense; if a pod genuinely dies mid-run, treat it as a restart from the top.
- **Phase 2 / DPO / GRPO workflows.** This runbook is for the Phase 1 SFT ablation only. Other workstreams have their own runbooks under `docs/runbooks/`.
- **The from-scratch trainer's internals.** Documented in `src/finpost/training/` source files and in the trainer's own docstrings.

---

*Last updated: 2026-05-18. If you hit a failure mode not covered here, add it.*
