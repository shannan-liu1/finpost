# Security policy and supply-chain posture

This project consumes external code (Python packages, Hugging Face dataset loader scripts, model weights) and untrusted external data (filings, teacher-generated content). The defaults below are intentionally cautious. Override them only with explicit reasoning recorded in the relevant PRD or commit message.

## Datasets (Hugging Face)

- Always load datasets through `finpost.safety.safe_load_dataset`, never via `datasets.load_dataset` directly. The wrapper enforces `trust_remote_code=False`.
- The actual security guarantee comes from `trust_remote_code=False`: with remote code disallowed, the `datasets` library will not execute any Python loader script that ships with a dataset, even if one is present. For datasets that ship parquet files alongside a script (most modern ones), the library auto-discovers and reads the parquet directly.
- Prefer datasets that are parquet-only at the source. For datasets that ship both a script and parquet (e.g. `openai/gsm8k`), the `trust_remote_code=False` default is sufficient — we get the parquet path automatically and the script is never executed.
- Pinning to the auto-converted `refs/convert/parquet` branch was originally listed here as an extra precaution. We dropped that approach in PRD 0002 because Hugging Face's auto-conversion flattens multi-config datasets to a single `default` config, which silently loses information (e.g. for GSM8K, the `main` vs `socratic` distinction). The `trust_remote_code=False` guarantee is sufficient on its own.
- Datasets currently approved:
  - `openai/gsm8k`, config `main`. Loaded with `trust_remote_code=False`; the upstream loader script is never executed.
  - `DigitalLearningGmbH/MATH-lighteval` — parquet-only mirror of the Hendrycks MATH dataset, MIT licensed.

## Models (Hugging Face)

- Always load models with `use_safetensors=True`. Refuse `.bin` (PyTorch pickle) format — `pickle` deserialization is a known remote-code-execution vector and the root cause of multiple recent incidents (CVE-2026-25874 in LeRobot is the most public example).
- Pin to canonical organization namespaces. Verify the org slug character-by-character before installing or downloading. Typo-squatted org names ("g00gle/gemma...") are a common attack pattern.
  - Approved repos for this project:
    - `google/gemma-3-1b-it` — instruction-tuned variant. Used as the base for Phase 1 training. **Required.** Verified loading 2026-05-05.
    - `google/gemma-3-1b-pt` — pretrained variant (`-pt` denotes "pretrained"). Optional: only needed for ablations against the non-instruction-tuned base.
  - Note: there is **no** repo at `google/gemma-3-1b` (no suffix). Gemma 3 changed naming convention from earlier Gemma versions; both pretrained and instruction-tuned variants now carry an explicit suffix.

## Authentication and tokens

- Hugging Face tokens live in `~/.cache/huggingface/token` (per-user, outside the repo) and must not be committed.
- API keys for Phase 2 (Anthropic and/or OpenAI) live in `.env` at the repo root, loaded via environment variables. Never inline a key in source.
- The repo `.gitignore` excludes `.env`, `.huggingface/`, `.hf_token`, `.cache/`, `*.key`, `*.pem`, `secrets/`. If you find a way to get a token into a tracked file, treat it as an incident: rotate the token immediately and amend the offending commit out of history before pushing.

## Python packages

- Dependencies are declared in `pyproject.toml` with bounded version ranges. A routine `pip install -e .` should not silently jump major versions.
- Run `pip-audit` periodically and before any compute-spending run to check for known CVEs in installed packages.
- Recent supply chain incidents to be aware of (none currently affect this project's dependencies, listed for context):
  - PyTorch Lightning 2.6.2 and 2.6.3 (April 2026): hidden `_runtime` payload that downloads and executes JavaScript on import. We use `torch` directly, not Lightning.
  - LiteLLM 1.82.7 and 1.82.8 (March 2026): malicious `.pth` file executing on every Python startup. Not a dependency.
  - LeRobot CVE-2026-25874 (April 2026, unpatched at time of policy): unauthenticated RCE via pickle deserialization in the gRPC PolicyServer. Not a dependency.
  - Marimo + HuggingFace Spaces blockchain botnet (April 2026): exploited a marimo pre-auth RCE to deploy NKAbuse via HF Spaces. Not a dependency or platform we use.

## Adding a new dependency or dataset

Before installing or downloading, check:

1. Is the publisher official or well-known? Look at the organization's public history, not just the package name.
2. Has the publisher been involved in a recent incident? Web search.
3. Is there a safer alternative — a parquet mirror of the dataset, a safetensors fork of the model, an official library that wraps a third-party tool?
4. Does the addition expand the trust surface? (Anything that runs Python at install, import, or load time qualifies.)

Record the decision in the PRD or as a commit message. Silent overrides of these defaults are the failure mode this file is meant to prevent.

## Reporting

If something here is wrong or out of date, fix it in the same change that introduces the new behavior. This file is the single source of truth for the project's security posture; if it drifts from reality, the policy stops protecting anything.
