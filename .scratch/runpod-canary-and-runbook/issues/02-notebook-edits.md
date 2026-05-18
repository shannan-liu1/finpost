# 02 - RunPod notebook: pre-flight expansion, canary, interleaved cleanup, monitoring

- **Status:** Ready
- **Ready for agent:** yes
- **Depends on:** 01-trainer-nan-halt (the canary's pass criterion relies on the trainer raising on NaN)

## Goal

Apply the five edits to `notebooks/sft_phase1_runpod_ablation_2000.ipynb` that turn it from "two-failure-prone-during-2000-step-run" into a "canary-protected, disk-bounded, fully self-documenting" workflow.

## Scope

**In scope (5 edits to one notebook, in place):**

1. **Pre-flight markdown expansion.** Replace the existing cell `8368cc33` content with a grandma-tone, command-by-command terminal pre-flight: `df -h /workspace`, `free -h`, `nvidia-smi`, repo checkout + pull, `pip install -e ".[dev]"`, `import finpost` sanity, `torch.cuda.is_available()` check, both known failure recipes (PEP 660 hook missing → `.pth` workaround; torch CUDA 13 ↔ driver CUDA 12.x → downgrade to `torch==2.4.1+cu124`). Each command preceded by what to look for in the output and what to do if it doesn't look right.

2. **Warmup floor one-liner.** In the hyperparameter cell (currently cell index 6, the one defining `ABLATION_STEPS = 2000` etc.), change:
   ```python
   warmup_steps = max(50, ABLATION_STEPS // 10)
   ```
   to:
   ```python
   warmup_steps = max(5, ABLATION_STEPS // 10)
   ```
   No other lines in that cell change.

3. **Canary cell.** Insert a new code cell after the YAML-generation cell (after cell-8) and before "Step 4 — Train each arm" (before cell-9). The cell:
   - Builds a transient canary YAML for the **combined** arm at `max_steps=50` AND `warmup_steps=5` (everything else identical to production hyperparams). **The canary MUST override `warmup_steps`** — inheriting the production value (200) would fail pydantic's `warmup_steps < max_steps` validator (`config.py:138-147`) and produce a misleading config-load error before training even starts. Compute: `warmup_steps = max(5, 50 // 10) = 5`.
   - Writes it to `experiments/runpod_a40/canary_50_a40.yaml`.
   - Subprocess-runs `python -m finpost.training.train --config <canary_yaml> --device cuda`.
   - Asserts subprocess returncode == 0; raises `RuntimeError` with the captured stderr if non-zero.
   - On success, prints `"\n✓ CANARY PASSED — safe to launch full 2000-step run\n"`.
   - On failure, prints `"\n✗ CANARY FAILED — DO NOT launch full run; inspect stderr above\n"` then raises.

4. **Interleaved per-arm convert+cleanup.** After each of the three existing arm-training cells (`cell-12`, `cell-14`, `cell-16`), insert a new code cell that:
   - Loops over `INTERMEDIATE_STEPS` for that arm.
   - For each step, runs `scripts/convert_checkpoint_to_hf.py` with the same arguments the current cell-20 uses.
   - After all 4 conversions succeed for that arm, runs `shutil.rmtree` on each raw `step-XXXXXXXX/` directory under that arm's checkpoint root.
   - Prints a per-step status line and a final disk-after-cleanup line.

   Repurpose `cell-18` (currently verifies raw checkpoints exist) to verify the HF-converted directories exist instead. Remove `cell-20` (the batch convert-all cell) — its work is now done per-arm.

5. **Monitoring cells (2 new code cells).** After the canary cell, insert `!df -h /workspace` then `!nvidia-smi` (one cell with both, joined by `;`). After the arm-3 convert+cleanup cell, insert another identical monitoring cell. These are the two highest-risk moments: pre-launch and post-train-cluster.

**Out of scope:**
- New notebook variants. Single notebook edited in place.
- Rewriting `convert_checkpoint_to_hf.py`. Per-arm loop calls the existing script unchanged.
- HF Hub push restructure. The optional `hf upload` block at the end stays commented out, as-is.

## Implementation notes

- All edits use `Edit`/`NotebookEdit` on the existing `.ipynb` JSON. Cell IDs should be preserved where cells are edited (not replaced) so git diffs stay legible.
- Canary YAML lives under `experiments/runpod_a40/` next to the three production YAMLs so they're co-located when debugging.
- The canary subprocess is invoked via `subprocess.run([...], capture_output=True, text=True, check=False)` so we can inspect both streams and raise our own error message rather than letting the subprocess raise (which produces less readable Jupyter tracebacks).

## Acceptance criteria

1. `git status` shows only `notebooks/sft_phase1_runpod_ablation_2000.ipynb` modified (no other tracked files).
2. The notebook JSON contains exactly the string `max(5, ABLATION_STEPS // 10)` (exactly once).
3. The notebook JSON contains at least one **code** cell whose source includes `CANARY PASSED` and at least one **code** cell whose source includes `CANARY FAILED`.
4. The notebook JSON contains at least three **code** cells whose source includes `shutil.rmtree` AND at least three **code** cells whose source includes the substring `step-` and `CHECKPOINTS_DIR` (paths constructed via the `CHECKPOINTS_DIR` variable; the literal path `results/checkpoints/` is only declared once in the hyperparameters cell).
5. The notebook JSON contains at least three **code** cells (cell_type == 'code') whose source contains both `df -h /workspace` and `nvidia-smi` — the pre-existing Step 1 sanity cell plus the two new monitoring cells (after canary, after arm 3).
6. `nbformat.read('notebooks/sft_phase1_runpod_ablation_2000.ipynb', as_version=4)` succeeds (notebook is still valid JSON).
7. `jupyter nbconvert --to script notebooks/sft_phase1_runpod_ablation_2000.ipynb --stdout > /tmp/nb.py && python -c "compile(open('/tmp/nb.py').read(), '<nb>', 'exec')"` succeeds (every code cell is valid Python).
