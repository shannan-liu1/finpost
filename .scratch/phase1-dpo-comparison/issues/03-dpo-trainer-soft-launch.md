# 03 - Add DPO trainer and soft-launch path

- **Status:** Not Started
- **Ready for agent:** yes
- **Depends on:** 02-dpo-loss-and-parity

## Parent

`.scratch/phase1-dpo-comparison/PRD.md`

## What to build

Add the DPO training loop, CLI, config, checkpointing, resume, and offline
tracking path by reusing the SFT trainer infrastructure wherever possible.
Run TinyGPT first, then a Qwen 0.5B canary on the target GPU before any full
DPO study.

## Acceptance criteria

- [ ] `src/finpost/training/dpo_train.py` loads a DPO config and launches
      training from a policy checkpoint plus frozen reference checkpoint.
- [ ] `experiments/dpo/qwen_dpo_baseline.yaml` defines model, pair data,
      training, beta, packing, logging, and checkpointing settings.
- [ ] Checkpoints include policy weights, optimizer state, scheduler state,
      global step, config, pair manifest id, and source SFT checkpoint id.
- [ ] Resume from a DPO checkpoint reproduces continuation loss within
      tolerance on a fixed fixture.
- [ ] TinyGPT DPO soft launch runs end to end on local/CPU.
- [ ] Qwen DPO canary runs for 20-50 steps on RunPod without non-finite loss
      or out-of-memory failure before the full run starts.

## Verification

- `.\.venv\Scripts\python.exe -m pytest tests/test_dpo_train_cli.py tests/test_checkpoint.py`
- RunPod canary command recorded in `docs/dpo-study.html`.

## Blocked by

Requires issue 02. The Qwen canary requires a real SFT checkpoint and 48 GB GPU.
