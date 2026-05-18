# RunPod canary + interleaved cleanup + end-to-end runbook

- **Status:** In Progress
- **Created:** 2026-05-18
- **Owner:** Shannan
- **Estimated time:** 3-5 hours
- **Depends on:** `phase1-sft-trainer` (uses the existing trainer; modifies one line of trainer.py)

## Goal

Stop burning RunPod money on failed SFT runs. Two failures have happened so far on the A40 2000-step ablation flow: (1) NaN-loss training silently producing garbage weights over many steps; (2) disk-OOM during the convert-all-then-eval-all stage on tight-volume pods. Both are addressable with small, surgical changes to the existing notebook + a 2-line trainer guard + a clearer pre-flight discipline.

Deliver: a 50-step canary that aborts before the 2000-step run if the recipe is broken on this hardware; an interleaved per-arm convert+cleanup that drops peak disk from ~48 GB to ~24 GB; a trainer-side `RuntimeError` on non-finite loss so failures are loud, not silent; and a single end-to-end HTML runbook the user can read side-by-side with the notebook without context-switching to other docs.

## Scope

**In scope:**
- Replace `max(50, …)` warmup floor with `max(5, …)` in `sft_phase1_runpod_ablation_2000.ipynb` so the canary can use `max_steps=50` without tripping pydantic validation.
- Add 1 canary cell (combined arm, 50 steps, production hyperparams) before arm-1 training. Hard-halt on subprocess exit != 0.
- Restructure the train→convert→cleanup flow into 3 per-arm cells (interleaved) instead of the current train-all → convert-all batch shape. Peak disk: ~24 GB.
- Add 2 monitoring cells (`!df -h /workspace` and `!nvidia-smi`) at the two highest-risk moments: after the canary, after arm 3.
- Expand pre-flight markdown at the top of the notebook with full terminal commands, expected outputs, and failure recipes.
- Add 2-line non-finite-loss guard in `trainer.py` that raises `RuntimeError` mid-training. Add regression test.
- Delete `sft_phase1_runpod_ablation_3000.ipynb` (untracked stale artifact).
- Write standalone HTML runbook at `docs/runbooks/runpod-end-to-end.html` (markdown source at `runpod-end-to-end.md`). End-to-end: pod creation → notebook execution → eval download → pod shutdown. Consolidates relevant pieces of `runpod-bootstrap.md` and `phase1-training-runbook.md` so the user reads one document.

**Out of scope:**
- Replacing the from-scratch trainer with TRL. The from-scratch trainer is the pedagogical core of the project (`user_background.md`); switching to TRL would defeat that, and it doesn't solve the actual problem (which is a missing pre-flight discipline, not trainer correctness — the trainer NaN was the SDPA mask interaction, already fixed in commit 5564f9c).
- Three-arm canary. One arm (combined) catches the same bug classes for 1/3 the cost; user explicitly asked for minimal surface area.
- Restructuring the `convert_checkpoint_to_hf.py` script. The per-arm convert+cleanup is done in the notebook by looping over checkpoint steps and calling the existing script; the script itself is untouched.
- HF Hub push for raw checkpoints. The notebook's optional `hf upload` block at the end remains as it is.
- A new notebook variant (e.g. `_v2`). User explicitly rejected this.

## Deliverables

### Code

- `src/finpost/training/trainer.py`: insert `if not torch.isfinite(loss): raise RuntimeError(...)` immediately after `loss = self._forward_loss(batch)` in `_run_training_loop`. Two lines plus an import if needed (`torch` is already imported). The error message includes the global step and the offending loss value.

### Tests

- `tests/test_trainer_nan_halt.py` (new): asserts that a trainer fed a model whose forward produces a NaN loss raises `RuntimeError` with "Non-finite loss" in the message, at the step the NaN appeared. Red-green: temporarily revert the guard; the test must fail.

### Notebook

- `notebooks/sft_phase1_runpod_ablation_2000.ipynb` (edited in place):
  - Expanded pre-flight markdown at top (cell `8368cc33`).
  - Hyperparameter cell (cell index 6): `warmup_steps = max(5, ABLATION_STEPS // 10)`.
  - New canary cell after YAML generation (between current cells `cell-8` and `cell-9`): subprocess-runs combined arm at `max_steps=50` with the production hyperparams, asserts exit code 0, prints PASS/FAIL.
  - New monitoring cell after the canary: `!df -h /workspace && !nvidia-smi`.
  - Existing arm-N training cells (`cell-12/14/16`): each followed by a NEW per-arm convert+cleanup cell that converts its 4 intermediate checkpoints to HF and `rm -rf`s the raw `step-*` directories for that arm.
  - Existing `cell-18` (verify intermediate checkpoints) repurposed to verify HF dirs exist instead of raw checkpoints.
  - Existing `cell-20` (convert-all): removed (no longer needed; conversion happens per-arm).
  - New monitoring cell after arm 3 completes.
- `notebooks/sft_phase1_runpod_ablation_3000.ipynb`: deleted.

### Documentation

- `docs/runbooks/runpod-end-to-end.md` (markdown source).
- `docs/runbooks/runpod-end-to-end.html` (standalone HTML, generated via `pandoc -s --metadata title="finpost RunPod end-to-end runbook" runpod-end-to-end.md -o runpod-end-to-end.html`).
- Both committed. Source is the authoritative version; HTML is regenerated on edit.

Runbook contents (sections):
1. What this notebook does and what it costs (~$0.50–1.10 on RTX 6000 / A40).
2. Pre-pod: which chip to pick, volume size, image, why.
3. Pod creation: the RunPod console clicks, in order.
4. Connecting via JupyterLab (not web terminal) — why.
5. Terminal pre-flight commands with expected outputs and "if you see X, do Y" failure recipes.
6. Notebook execution: cell-by-cell pass criteria, what to look for, when to halt.
7. Mid-flow monitoring: when and what to inspect.
8. After eval: tarball, download, pod stop vs terminate.

## Acceptance criteria

1. `pytest tests/test_trainer_nan_halt.py -v` passes.
2. `pytest tests/test_loss_dtype.py tests/test_dataset.py -v` still pass (no regression).
3. `git status` shows `notebooks/sft_phase1_runpod_ablation_3000.ipynb` as deleted (not present).
4. `python -c "import json; nb=json.load(open('notebooks/sft_phase1_runpod_ablation_2000.ipynb')); assert any('canary' in ''.join(c['source']).lower() for c in nb['cells'] if c['cell_type']=='code')"` passes (canary cell exists).
5. The notebook contains the string `max(5, ABLATION_STEPS // 10)`.
6. The notebook contains a cell that calls `rm -rf` against a raw `step-*` checkpoint path.
7. `docs/runbooks/runpod-end-to-end.md` exists and is non-empty.
8. `docs/runbooks/runpod-end-to-end.html` exists, opens in a browser, contains all 8 sections listed above.
9. Trainer NaN regression test red-greens when the guard is reverted.

## Notes / open questions

- The pandoc dependency for HTML generation is a developer-side requirement (not a runtime requirement). If `pandoc` is missing locally, the markdown source still lands; the HTML can be regenerated later. The PRD acceptance criteria require the HTML to be committed, so a single `pandoc` call has to land on a machine where it's installed.
- The trainer NaN guard halts training on the first non-finite loss. Some recipes accept transient NaNs during early warmup; our recipe has never observed that and we want to fail loud, so this is the right default for finpost. If a future recipe needs tolerance, the guard can be gated behind a config flag.
- The interleaved cleanup deletes raw checkpoints after HF conversion. This loses the ability to resume training from those steps with bit-identical optimizer state. We don't need that — the run completes end-to-end without resume in normal operation, and any "load weights and fine-tune further" path uses HF-format weights with a fresh optimizer anyway.
- The runbook assumes RTX 6000 or A40 (48 GB VRAM). If the user falls back to 3090, the runbook should be amended (or a sidebar added) — out of scope for v1; documented as a "if you can't get 48 GB" note pointing back to this PRD.
- **Canary coverage gap (accepted):** the canary runs the `combined` arm only. If the user later runs `gsm8k_only` or `math_only` standalone (e.g. re-doing one arm after a partial failure), the canary doesn't cover the dataset-specific packing patterns those arms produce. Accepted because in the normal flow all three arms train back-to-back in one notebook run; the canary protects that flow. If standalone re-runs become routine, parameterize the canary to take an `--arm` flag in a future iteration.
- **HTML hand-written, not pandoc-generated:** local Windows env has no pandoc and the project shouldn't take a pandoc dependency for a single document. Issue 03 specifies hand-written HTML with an embedded `<style>` block, ensuring offline-self-containment without external tooling.
