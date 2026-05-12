# 03 - Colab and Kaggle eval notebooks

- **Status:** Not Started
- **Ready for agent:** yes (after issue 02)
- **Depends on:** 02-eval-exact-cli-and-output

## Goal

Create two thin notebook wrappers — one for Google Colab, one for Kaggle — that call the `python -m finpost.evals.eval_exact` CLI built in issue 02 and render summary tables inline. Each notebook handles only the platform-specific setup (paths, install commands, secrets); all eval logic lives in the CLI.

## Scope

**In scope:**
- `notebooks/colab_phase1_eval_and_cost_tracking.ipynb` — Colab variant; uses `/content/drive/MyDrive/finpost_runs/...` paths; mounts Drive; reads Colab Secrets for HF token.
- `notebooks/kaggle_phase1_eval_and_cost_tracking.ipynb` — Kaggle variant; uses `/kaggle/working/...` paths; reads Kaggle Secrets for HF token.
- Both notebooks must NOT have platform-specific paths leaking into the other. Colab notebook has no `/kaggle/working/`; Kaggle notebook has no `/content/drive/`.

**Out of scope:**
- Re-implementing any eval logic. Notebooks are thin wrappers; the CLI does the work.
- Generating plots in separate files. Render inline via matplotlib if useful; the goal is human-readable, not paper-figure-quality.
- Implementing OPD, DPO, training, rollouts, or anything from compute-aware. This is eval only.

## Notebook structure (both variants)

Nine cells. Identical logical structure; only the platform-specific cells differ.

| # | Cell type | Purpose |
|---|---|---|
| 0 | Markdown | Title, link to the PRD, one-paragraph "what this does" |
| 1 | Code | Runtime check: `nvidia-smi`, `torch.cuda.is_available()`, `torch.cuda.get_device_name()` |
| 2 | Code | Paths setup (platform-specific) and Drive mount (Colab only) |
| 3 | Code | `pip install -e .` or `pip install -e ".[dev]"` from the repo |
| 4 | Code | Download / restore the combined SFT checkpoint to a known path |
| 5 | Code | Sanity check: import `finpost.evals.sources`, print `REGISTRY` keys |
| 6 | Code | Call the CLI: `!python -m finpost.evals.eval_exact --checkpoints base=Qwen/Qwen2.5-0.5B combined=<path> --sources gsm8k math --n 500 --seed 42 --out-dir <results_path> ...` |
| 7 | Code | Load `accuracy_summary.json` back, render as a pandas DataFrame (or stdlib table) inline |
| 8 | Code | Load `cost_summary.json`, print elapsed time, generated tokens, tokens/sec, optional cost estimate |

## Platform-specific specifics

### Colab
- Cell 2: `from google.colab import drive; drive.mount('/content/drive')`. Set `REPO_ROOT = '/content/finpost'`, `RESULTS_ROOT = '/content/drive/MyDrive/finpost_runs/results/evals'`, `CHECKPOINT_PATH = '/content/drive/MyDrive/finpost_runs/checkpoints/combined_hf_step_<N>'`.
- Cell 4: assumes the checkpoint already exists in Drive; just verifies the directory is present and prints the contents. Optionally unzips a `.zip` if a zip is present and the directory isn't.
- HF token via Colab Secrets: `from google.colab import userdata; os.environ['HF_TOKEN'] = userdata.get('HF_TOKEN')` (handle the case where the secret is not set).

### Kaggle
- Cell 2: `REPO_ROOT = '/kaggle/working/finpost'`, `RESULTS_ROOT = '/kaggle/working/results/evals'`, `CHECKPOINT_PATH = '/kaggle/working/results/checkpoints/<run_name>'` or a Kaggle Dataset path if the user attached the checkpoint as a dataset.
- Cell 4: assumes the user attached the checkpoint as a Kaggle Dataset (path under `/kaggle/input/<dataset-name>/`) OR built it in a prior cell. Print the directory contents to verify.
- HF token via Kaggle Secrets: `from kaggle_secrets import UserSecretsClient; os.environ['HF_TOKEN'] = UserSecretsClient().get_secret("HF_TOKEN")`.

## Acceptance criteria

1. Both notebook files exist at the paths above and parse as valid JSON (Jupyter notebooks are JSON).
2. The Colab notebook contains no string `/kaggle/working/` or `kaggle_secrets`. The Kaggle notebook contains no string `/content/drive/` or `google.colab`. (Easy grep check.)
3. Both notebooks have exactly nine cells matching the structure above.
4. Cell 6 in both notebooks invokes `python -m finpost.evals.eval_exact` with the same flags shown in issue 02's CLI signature (only the path values differ).
5. The notebooks can be opened in a Jupyter / Colab environment and read top-to-bottom without manual editing — i.e., a fresh user could fill in `CHECKPOINT_PATH` and run.

## Notes / open questions

- Notebook authoring is finicky — use the `nbformat` Python library or write JSON directly. The existing notebooks under `notebooks/` are useful references for cell structure and metadata.
- Do not commit any large output (images, base64 PNGs) to the notebook source. Clear outputs before saving.
- The "render as a pandas DataFrame" can use `pd.read_json` on the summary file. If pandas is not in the project's dependencies, fall back to a stdlib pretty-print — but check `pyproject.toml` first; pandas may already be in the install.
- Existing notebooks `sft_phase1_colab.ipynb` and `sft_phase1_kaggle.ipynb` are the structural templates. Mirror their setup style for consistency.
