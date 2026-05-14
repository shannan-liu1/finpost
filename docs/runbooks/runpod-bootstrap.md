# RunPod bootstrap

First-time setup on a fresh RunPod pod before any training or eval notebook runs. Every step here corresponds to a failure mode hit during a real session — skip a step and you will probably hit the failure it guards against.

The pod is assumed to have:
- `/workspace/` as a persistent volume
- the `finpost` repo cloned at `/workspace/finpost` (or cloneable there)
- JupyterLab running on port 8888

## Order of operations

```
0. Open a JupyterLab terminal
1. Update or clone the repo
2. Restore the working tree if files are missing
3. Install finpost as an editable package
4. Verify the editable install actually exposes finpost
5. Verify torch + CUDA agree with the pod's NVIDIA driver
6. Open the experiment notebook
```

## 0. Open a JupyterLab terminal

`File > New > Terminal`. All steps below run in that shell, not in a notebook cell.

## 1. Update or clone the repo

```bash
cd /workspace
git clone https://github.com/shannan-liu1/finpost.git   # first time only
cd finpost
git pull
```

## 2. Restore the working tree if files are missing

Pods sometimes start with tracked files **deleted from the working tree** (root cause unclear — possibly cleanup hooks in the base image; happens even when the pod's volume is the same as a previous session). The deletions are unstaged, so `git pull` succeeds while the source tree is half-empty.

```bash
git status
```

If you see a long list of `deleted:` entries — anything from `src/finpost/__init__.py` to `tests/test_*.py` — restore them before continuing:

```bash
git restore .
git status   # must now print "nothing to commit, working tree clean" (plus a few untracked dirs)
```

The only acceptable leftover untracked entries are JupyterLab's `.ipynb_checkpoints/` and notebook-generated dirs like `experiments/runpod_a40/`. If anything tracked is still missing, run `git checkout HEAD -- <path>` directly.

## 3. Install finpost as an editable package

```bash
pip install -e ".[dev]"
```

This download is large — ~3 GB of torch + CUDA wheels. Expect 8–15 min on a typical RunPod connection. The pip output will go silent for several minutes during the `Installing collected packages: ...` phase as wheels unpack and link. **Silence is not a hang.** Verify pip is still alive from a second terminal:

```bash
ps aux | grep -E "pip|python" | grep -v grep
```

If `pip` is in the process list, let it cook. When the `Successfully installed ...` line appears, move on.

## 4. Verify the editable install actually exposes finpost

This is the step most likely to bite. Pip can land the metadata (`pip show finpost` succeeds) **without writing the PEP 660 import hook** that makes `import finpost` resolve. The two are separate writes inside `pip install -e`, and the second one occasionally gets skipped.

Test:

```bash
python -c "import finpost; print(finpost.__file__)"
```

**If it prints `/workspace/finpost/src/finpost/__init__.py` → you're done with this step.**

**If it raises `ModuleNotFoundError: No module named 'finpost'` despite `pip show finpost` succeeding**, the import hook is missing. Diagnose:

```bash
ls /usr/local/lib/python3.11/dist-packages/ | grep -i finpost
```

If you see **only** `finpost-0.0.1.dist-info/` and no `__editable__.finpost-*.pth` or `__editable___finpost_*_finder.py`, the hook was never written. Fix by adding the path file by hand — what pip was supposed to write:

```bash
echo "/workspace/finpost/src" > /usr/local/lib/python3.11/dist-packages/finpost.pth
python -c "import finpost; print(finpost.__file__)"   # should print the path now
```

Do not waste time chasing the underlying pip / setuptools / PEP 660 interaction on the pod. The manual `.pth` is the canonical fallback and persists across kernel restarts on this pod.

## 5. Verify torch + CUDA agree with the pod's NVIDIA driver

`pyproject.toml` pins `torch>=2.5`. `pip install -e ".[dev]"` therefore uninstalls the pod's pre-shipped `torch 2.4.1+cu124` and installs `torch 2.12.0 + CUDA 13 toolkit`. RunPod A40 pods typically ship with an NVIDIA driver supporting CUDA 12.x — **CUDA 13 wheels may not initialize on that driver.**

Test:

```bash
python -c "import torch; print('cuda:', torch.cuda.is_available()); print('device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none')"
```

**If `cuda: True` and the device prints (e.g., `NVIDIA A40`) → continue.**

**If `cuda: False`** → the CUDA 13 wheel can't talk to the driver. Downgrade torch to match what the driver supports:

```bash
pip install "torch==2.4.1+cu124" --index-url https://download.pytorch.org/whl/cu124 --force-reinstall --no-deps
python -c "import torch; print('cuda:', torch.cuda.is_available())"
```

`--no-deps` keeps the rest of the install intact. Re-test. Once `cuda: True`, restart any open notebook kernels so they pick up the working torch.

Note: this downgrade leaves `torchaudio` / `torchvision` mismatched if they were ever installed (pip will warn during `--force-reinstall`). They are not used by finpost SFT or DPO, so the warning is safe to ignore.

## 6. Open the experiment notebook

Open `notebooks/sft_phase1_runpod_ablation_2000.ipynb` (or the relevant experiment notebook). The notebook's own first code cell runs `git pull` and `import finpost` — both should now succeed. Run cells in order.

If you are coming back to a running notebook after a kernel restart, you do **not** need to re-run steps 1–5; the pod state persists. Steps 4 and 5 only need a re-check if you re-installed packages.

## Troubleshooting cheatsheet

| Symptom | Most likely cause | Fix |
|---|---|---|
| `git pull` shows huge list of `deleted:` files in `git status` | Working-tree damage on pod start | `git restore .` |
| `pip install` hangs at "Installing collected packages" | Normal — pip is silently linking wheels | Wait. Verify via `ps aux` from a second terminal. |
| `pip show finpost` works, `import finpost` fails | PEP 660 import hook never written | `echo "/workspace/finpost/src" > /usr/local/lib/python3.11/dist-packages/finpost.pth` |
| `torch.cuda.is_available()` returns `False` after install | CUDA 13 wheel vs CUDA 12.x driver | Downgrade torch to `2.4.1+cu124` |
| Notebook cell errors with `ModuleNotFoundError` after a fresh kernel | Kernel started before the `.pth` file existed | Restart the kernel; the `.pth` is read on Python startup |
