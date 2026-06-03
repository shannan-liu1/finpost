# Troubleshooting

## GPU Setup

1. Start from the clean repo and install requirements in a fresh virtual environment.
2. Confirm the repo imports from `src/`, not from an old checkout.
3. Run `scripts/check_cuda_stack.py` before training.
4. Run `scripts/gpu_preflight.py` before spending on a full training run.

For the full fresh-environment sequence, use
`docs/runbooks/finchain-gpu-quickstart.md`.

## CUDA And bf16

- `nvidia-smi` only proves that the driver sees the GPU. It does not prove that the active Python environment has a CUDA-enabled PyTorch build.
- Check `torch.cuda.is_available()` from the exact Python environment used by the notebook.
- A40/A100/H100 environments can use bf16-capable paths, but small T4/P100-style environments may require fp16 or fp32 fallback.
- If a notebook fails at model load, verify torch, transformers, CUDA runtime, and available VRAM before changing training logic.

## Hugging Face Uploads

- Use `huggingface-cli login` or `hf auth login` in the runtime terminal before upload cells.
- Use repo IDs under `shannan-liu1`.
- Check `docs/hf_checkpoints.md` before pushing. Reported results should refer only to repos listed there as verified public.
- Do not commit HF tokens, W&B keys, `.env`, or cache directories.

## Common Path Problems

- Configs live under `configs/`, not `experiments/`.
- Local/generated FinChain data should live under `data/finchain_v2_template_disjoint/`.
- Checkpoints and eval dumps should stay under ignored `results/checkpoints/` and `results/evals/`.
- Avoid absolute local paths in committed notebooks and docs. Keep runtime-specific paths configurable.

## Repository ID Drift

If a command or notebook references an older repository namespace, stop and
replace it with the intended `shannan-liu1` repo ID. Then check whether the new
repo is verified in `docs/hf_checkpoints.md`.

## Notebook Order

1. `notebooks/01_sft_ablation.ipynb`
2. `notebooks/02_dpo.ipynb`
3. `notebooks/03_grpo_rlvr.ipynb`
4. `notebooks/04_opd_gkd.ipynb`
Do not use test split results to choose checkpoints. Validation selects; test reports.
